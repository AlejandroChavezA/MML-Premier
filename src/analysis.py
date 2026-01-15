import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

class PremierLeagueAnalyzer:
    def __init__(self):
        self.data_dir = "../data"
        self.teams_df = None
        self.matches_df = None
        self.standings_df = None
    
    def load_data(self, season=2023):
        """Cargar datos desde archivos CSV"""
        try:
            self.teams_df = pd.read_csv(f"{self.data_dir}/teams.csv")
            self.matches_df = pd.read_csv(f"{self.data_dir}/matches_{season}.csv")
            self.standings_df = pd.read_csv(f"{self.data_dir}/standings_{season}.csv")
            print("Datos cargados exitosamente")
            return True
        except FileNotFoundError as e:
            print(f"No se encontraron archivos de datos: {e}")
            return False
    
    def basic_statistics(self):
        """Estadísticas básicas de la temporada"""
        if self.matches_df is None:
            print("Primero carga los datos con load_data()")
            return
        
        stats = {
            'total_matches': len(self.matches_df),
            'total_teams': len(self.teams_df) if self.teams_df is not None else 20,
            'avg_goals_per_match': 0,
            'home_win_rate': 0,
            'away_win_rate': 0,
            'draw_rate': 0
        }
        
        # Calcular goles promedio
        if 'home_score' in self.matches_df.columns and 'away_score' in self.matches_df.columns:
            total_goals = (self.matches_df['home_score'].fillna(0) + 
                          self.matches_df['away_score'].fillna(0)).sum()
            stats['avg_goals_per_match'] = total_goals / len(self.matches_df)
            
            # Calcular tasas de resultados
            home_wins = len(self.matches_df[self.matches_df['home_score'] > self.matches_df['away_score']])
            away_wins = len(self.matches_df[self.matches_df['home_score'] < self.matches_df['away_score']])
            draws = len(self.matches_df[self.matches_df['home_score'] == self.matches_df['away_score']])
            
            stats['home_win_rate'] = home_wins / len(self.matches_df) * 100
            stats['away_win_rate'] = away_wins / len(self.matches_df) * 100
            stats['draw_rate'] = draws / len(self.matches_df) * 100
        
        return stats
    
    def team_performance_analysis(self):
        """Análisis de rendimiento por equipo"""
        if self.matches_df is None:
            print("Primero carga los datos con load_data()")
            return
        
        # Estadísticas por equipo como local
        home_stats = self.matches_df.groupby('home_team').agg({
            'home_score': ['sum', 'mean', 'count'],
            'away_score': ['sum', 'mean']
        }).round(2)
        
        # Estadísticas por equipo como visitante
        away_stats = self.matches_df.groupby('away_team').agg({
            'away_score': ['sum', 'mean', 'count'],
            'home_score': ['sum', 'mean']
        }).round(2)
        
        return home_stats, away_stats
    
    def goal_distribution(self):
        """Análisis de distribución de goles"""
        if self.matches_df is None:
            print("Primero carga los datos con load_data()")
            return
        
        # Combinar goles de local y visitante
        home_goals = self.matches_df['home_score'].fillna(0)
        away_goals = self.matches_df['away_score'].fillna(0)
        
        goal_stats = {
            'home_goals_mean': home_goals.mean(),
            'away_goals_mean': away_goals.mean(),
            'home_goals_std': home_goals.std(),
            'away_goals_std': away_goals.std(),
            'total_goals_mean': (home_goals + away_goals).mean()
        }
        
        return goal_stats
    
    def create_visualizations(self):
        """Crear visualizaciones básicas"""
        if self.matches_df is None:
            print("Primero carga los datos con load_data()")
            return
        
        # 1. Distribución de goles
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Goles locales
        self.matches_df['home_score'].hist(bins=20, ax=axes[0,0], alpha=0.7)
        axes[0,0].set_title('Distribución de Goles Locales')
        axes[0,0].set_xlabel('Goles')
        axes[0,0].set_ylabel('Frecuencia')
        
        # Goles visitantes
        self.matches_df['away_score'].hist(bins=20, ax=axes[0,1], alpha=0.7, color='orange')
        axes[0,1].set_title('Distribución de Goles Visitantes')
        axes[0,1].set_xlabel('Goles')
        axes[0,1].set_ylabel('Frecuencia')
        
        # Total de goles por partido
        total_goals = self.matches_df['home_score'].fillna(0) + self.matches_df['away_score'].fillna(0)
        total_goals.hist(bins=20, ax=axes[1,0], alpha=0.7, color='green')
        axes[1,0].set_title('Distribución de Total de Goles')
        axes[1,0].set_xlabel('Total de Goles')
        axes[1,0].set_ylabel('Frecuencia')
        
        # Resultados (victoria local, victoria visitante, empate)
        results = []
        for _, match in self.matches_df.iterrows():
            if match['home_score'] > match['away_score']:
                results.append('Victoria Local')
            elif match['home_score'] < match['away_score']:
                results.append('Victoria Visitante')
            else:
                results.append('Empate')
        
        pd.Series(results).value_counts().plot(kind='bar', ax=axes[1,1], color=['blue', 'red', 'gray'])
        axes[1,1].set_title('Resultados de Partidos')
        axes[1,1].set_ylabel('Cantidad')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/basic_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualización guardada como 'basic_analysis.png'")

if __name__ == "__main__":
    analyzer = PremierLeagueAnalyzer()
    
    # Ejemplo de uso
    if analyzer.load_data(2023):
        stats = analyzer.basic_statistics()
        print("\nEstadísticas Básicas:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        analyzer.create_visualizations()