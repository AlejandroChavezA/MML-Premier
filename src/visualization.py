import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class PremierLeagueVisualizer:
    def __init__(self):
        self.data_dir = "../data"
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def load_data(self, season=2023):
        """Cargar datos para visualización"""
        try:
            self.matches_df = pd.read_csv(f"{self.data_dir}/matches_{season}.csv")
            self.standings_df = pd.read_csv(f"{self.data_dir}/standings_{season}.csv")
            return True
        except FileNotFoundError:
            print("No se encontraron archivos de datos")
            return False
    
    def plot_standings_table(self):
        """Gráfico interactivo de la tabla de posiciones"""
        if self.standings_df is None:
            print("Primero carga los datos")
            return
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Pos', 'Equipo', 'PJ', 'PG', 'PE', 'PP', 'PTS', 'GF', 'GC', 'DG'],
                fill_color='lightblue',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[
                    self.standings_df['position'],
                    self.standings_df['team'],
                    self.standings_df['played_games'],
                    self.standings_df['won'],
                    self.standings_df['draw'],
                    self.standings_df['lost'],
                    self.standings_df['points'],
                    self.standings_df['goals_for'],
                    self.standings_df['goals_against'],
                    self.standings_df['goal_difference']
                ],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title='Tabla de Posiciones Premier League',
            height=600
        )
        
        fig.show()
    
    def plot_goals_scatter(self):
        """Gráfico de dispersión de goles a favor vs en contra"""
        if self.standings_df is None:
            print("Primero carga los datos")
            return
        
        fig = px.scatter(
            self.standings_df,
            x='goals_for',
            y='goals_against',
            size='points',
            color='position',
            hover_name='team',
            title='Goles a Favor vs Goles en Contra',
            labels={
                'goals_for': 'Goles a Favor',
                'goals_against': 'Goles en Contra',
                'points': 'Puntos',
                'position': 'Posición'
            }
        )
        
        fig.add_shape(
            type="line",
            x0=self.standings_df['goals_for'].min(),
            y0=self.standings_df['goals_for'].min(),
            x1=self.standings_df['goals_for'].max(),
            y1=self.standings_df['goals_for'].max(),
            line=dict(color="gray", dash="dash")
        )
        
        fig.show()
    
    def plot_goal_trends(self):
        """Tendencias de goles durante la temporada"""
        if self.matches_df is None:
            print("Primero carga los datos")
            return
        
        # Convertir fecha a datetime
        self.matches_df['date'] = pd.to_datetime(self.matches_df['date'])
        self.matches_df = self.matches_df.sort_values('date')
        
        # Calcular goles acumulados por jornada
        self.matches_df['total_goals'] = self.matches_df['home_score'] + self.matches_df['away_score']
        goals_by_round = self.matches_df.groupby('matchday')['total_goals'].mean().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=goals_by_round['matchday'],
            y=goals_by_round['total_goals'],
            mode='lines+markers',
            name='Promedio de Goles',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Evolución de Goles por Jornada',
            xaxis_title='Jornada',
            yaxis_title='Promedio de Goles por Partido',
            hovermode='x unified'
        )
        
        fig.show()
    
    def plot_home_away_performance(self):
        """Comparación rendimiento local vs visitante"""
        if self.matches_df is None:
            print("Primero carga los datos")
            return
        
        # Calcular estadísticas locales y visitantes
        home_performance = self.matches_df.groupby('home_team').agg({
            'home_score': ['sum', 'mean', 'count'],
            'away_score': ['sum']
        }).round(2)
        
        away_performance = self.matches_df.groupby('away_team').agg({
            'away_score': ['sum', 'mean', 'count'],
            'home_score': ['sum']
        }).round(2)
        
        # Simplificar para visualización
        home_stats = self.matches_df.groupby('home_team').apply(
            lambda x: pd.Series({
                'goals_scored': x['home_score'].sum(),
                'goals_conceded': x['away_score'].sum(),
                'wins': (x['home_score'] > x['away_score']).sum(),
                'matches': len(x)
            })
        ).reset_index()
        
        away_stats = self.matches_df.groupby('away_team').apply(
            lambda x: pd.Series({
                'goals_scored': x['away_score'].sum(),
                'goals_conceded': x['home_score'].sum(),
                'wins': (x['away_score'] > x['home_score']).sum(),
                'matches': len(x)
            })
        ).reset_index()
        
        # Crear gráfico comparativo
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Rendimiento Local', 'Rendimiento Visitante'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Top 10 equipos locales
        top_home = home_stats.nlargest(10, 'wins')
        fig.add_trace(
            go.Bar(x=top_home['home_team'], y=top_home['wins'], name='Victorias Local', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Top 10 equipos visitantes
        top_away = away_stats.nlargest(10, 'wins')
        fig.add_trace(
            go.Bar(x=top_away['away_team'], y=top_away['wins'], name='Victorias Visitante', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Comparación Rendimiento Local vs Visitante (Top 10)',
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        
        fig.show()
    
    def plot_results_pie(self):
        """Gráfico de pastel de resultados"""
        if self.matches_df is None:
            print("Primero carga los datos")
            return
        
        # Clasificar resultados
        results = []
        for _, match in self.matches_df.iterrows():
            if match['home_score'] > match['away_score']:
                results.append('Victoria Local')
            elif match['home_score'] < match['away_score']:
                results.append('Victoria Visitante')
            else:
                results.append('Empate')
        
        result_counts = pd.Series(results).value_counts()
        
        fig = px.pie(
            values=result_counts.values,
            names=result_counts.index,
            title='Distribución de Resultados en la Temporada',
            color_discrete_map={
                'Victoria Local': 'lightblue',
                'Victoria Visitante': 'lightcoral',
                'Empate': 'lightgray'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.show()

if __name__ == "__main__":
    visualizer = PremierLeagueVisualizer()
    
    if visualizer.load_data(2023):
        print("Creando visualizaciones...")
        visualizer.plot_standings_table()
        visualizer.plot_goals_scatter()
        visualizer.plot_goal_trends()
        visualizer.plot_home_away_performance()
        visualizer.plot_results_pie()