import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class AdvancedFeatureEngineer:
    def __init__(self, data_dir: str = "data/cleaned"):
        self.data_dir = data_dir
        self.matches_2023 = None
        self.matches_2024 = None
        self.matches_2025 = None
        self.teams_df = None
        self.standings_2025 = None
        
    def load_data(self):
        """Cargar todos los datos necesarios"""
        try:
            self.matches_2023 = pd.read_csv(f"{self.data_dir}/matches_2023_cleaned.csv")
            self.matches_2024 = pd.read_csv(f"{self.data_dir}/matches_2024_cleaned.csv")
            self.matches_2025 = pd.read_csv(f"{self.data_dir}/matches_2025_cleaned.csv")
            self.teams_df = pd.read_csv(f"{self.data_dir}/teams_cleaned.csv")
            self.standings_2025 = pd.read_csv(f"{self.data_dir}/standings_2025_cleaned.csv")
            
            # Convertir fechas y remover timezone
            for df in [self.matches_2023, self.matches_2024, self.matches_2025]:
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                
            print("Datos cargados exitosamente")
            return True
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return False
    
    def get_team_standings_stats(self, team: str) -> Dict:
        """Obtener estadísticas actuales de la tabla"""
        if self.standings_2025 is None:
            return self._get_default_standings()
        
        team_row = self.standings_2025[self.standings_2025['team'] == team]
        
        if len(team_row) == 0:
            return self._get_default_standings()
        
        stats = team_row.iloc[0]
        
        return {
            'position': stats['position'],
            'points': stats['points'],
            'played_games': stats['played_games'],
            'goal_difference': stats['goal_difference'],
            'points_per_game': stats['points'] / max(1, stats['played_games']),
            'goal_difference_per_game': stats['goal_difference'] / max(1, stats['played_games'])
        }
    
    def get_recent_form_detailed(self, team: str, date: datetime, last_n_games: int = 5) -> Dict:
        """Forma reciente detallada (últimos N partidos)"""
        # Combinar datos históricos
        historical_matches = pd.concat([self.matches_2023, self.matches_2024])
        
        # Filtrar partidos del equipo antes de la fecha
        team_matches = []
        
        for df in [historical_matches, self.matches_2025]:
            team_mask = ((df['home_team'] == team) | (df['away_team'] == team))
            date_mask = df['date'] < date
            finished_mask = df['status'] == 'FINISHED'
            
            matches = df[team_mask & date_mask & finished_mask].copy()
            team_matches.append(matches)
        
        if not team_matches:
            return self._get_default_form()
            
        all_matches = pd.concat(team_matches).sort_values('date', ascending=False)
        
        # Tomar últimos N partidos
        recent_matches = all_matches.head(last_n_games)
        
        if len(recent_matches) == 0:
            return self._get_default_form()
        
        # Calcular estadísticas detalladas
        stats = {
            'matches_played': len(recent_matches),
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'points': 0,
            'goal_difference': 0,
            'unbeaten_streak': 0,  # Racha de invicto
            'home_specific': {'wins': 0, 'goals_for': 0, 'goals_against': 0},
            'away_specific': {'wins': 0, 'goals_for': 0, 'goals_against': 0}
        }
        
        # Analizar cada partido
        for _, match in recent_matches.iterrows():
            is_home = match['home_team'] == team
            team_score = match['home_score'] if is_home else match['away_score']
            opponent_score = match['away_score'] if is_home else match['home_score']
            
            # Actualizar estadísticas generales
            stats['goals_scored'] += team_score
            stats['goals_conceded'] += opponent_score
            stats['goal_difference'] += (team_score - opponent_score)
            
            # Actualizar específicas local/visitante
            if is_home:
                stats['home_specific']['goals_for'] += team_score
                stats['home_specific']['goals_against'] += opponent_score
            else:
                stats['away_specific']['goals_for'] += team_score
                stats['away_specific']['goals_against'] += opponent_score
            
            # Resultados y puntos
            if team_score > opponent_score:
                stats['wins'] += 1
                stats['points'] += 3
                if is_home:
                    stats['home_specific']['wins'] += 1
                else:
                    stats['away_specific']['wins'] += 1
            elif team_score == opponent_score:
                stats['draws'] += 1
                stats['points'] += 1
            else:
                stats['losses'] += 1
        
        # Calcular racha de invicto
        stats['unbeaten_streak'] = self._calculate_unbeaten_streak(recent_matches, team)
        
        # Calcular tasas
        if stats['matches_played'] > 0:
            stats['win_rate'] = stats['wins'] / stats['matches_played']
            stats['points_per_game'] = stats['points'] / stats['matches_played']
            stats['goals_per_game'] = stats['goals_scored'] / stats['matches_played']
            stats['goals_conceded_per_game'] = stats['goals_conceded'] / stats['matches_played']
            stats['goal_difference_per_game'] = stats['goal_difference'] / stats['matches_played']
        
        return stats
    
    def get_home_away_performance_advanced(self, team: str, venue: str, date: datetime) -> Dict:
        """Rendimiento específico local/visitante avanzado"""
        # Combinar datos históricos
        all_matches = pd.concat([self.matches_2023, self.matches_2024, self.matches_2025])
        
        # Filtrar partidos del equipo en el venue específico antes de la fecha
        if venue == 'home':
            team_mask = all_matches['home_team'] == team
            team_score_col = 'home_score'
            opponent_score_col = 'away_score'
        else:  # away
            team_mask = all_matches['away_team'] == team
            team_score_col = 'away_score'
            opponent_score_col = 'home_score'
        
        date_mask = all_matches['date'] < date
        finished_mask = all_matches['status'] == 'FINISHED'
        
        venue_matches = all_matches[team_mask & date_mask & finished_mask].sort_values('date', ascending=False)
        
        if len(venue_matches) == 0:
            return self._get_default_venue_performance(venue)
        
        # Tomar todos los partidos del venue
        stats = {
            'matches_played': len(venue_matches),
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'points': 0,
            'goal_difference': 0,
            'win_rate': 0.0,
            'points_per_game': 0.0,
            'goals_per_game': 0.0,
            'goals_conceded_per_game': 0.0,
            'goal_difference_per_game': 0.0,
            'clean_sheets': 0,
            'failed_to_score': 0
        }
        
        for _, match in venue_matches.iterrows():
            team_score = match[team_score_col]
            opponent_score = match[opponent_score_col]
            
            # Estadísticas básicas
            stats['goals_scored'] += team_score
            stats['goals_conceded'] += opponent_score
            stats['goal_difference'] += (team_score - opponent_score)
            
            # Clean sheets y no marcar
            if opponent_score == 0:
                stats['clean_sheets'] += 1
            if team_score == 0:
                stats['failed_to_score'] += 1
            
            # Resultados y puntos
            if team_score > opponent_score:
                stats['wins'] += 1
                stats['points'] += 3
            elif team_score == opponent_score:
                stats['draws'] += 1
                stats['points'] += 1
            else:
                stats['losses'] += 1
        
        # Calcular tasas
        if stats['matches_played'] > 0:
            stats['win_rate'] = stats['wins'] / stats['matches_played']
            stats['points_per_game'] = stats['points'] / stats['matches_played']
            stats['goals_per_game'] = stats['goals_scored'] / stats['matches_played']
            stats['goals_conceded_per_game'] = stats['goals_conceded'] / stats['matches_played']
            stats['goal_difference_per_game'] = stats['goal_difference'] / stats['matches_played']
            stats['clean_sheets_rate'] = stats['clean_sheets'] / stats['matches_played']
            stats['scoring_rate'] = (stats['matches_played'] - stats['failed_to_score']) / stats['matches_played']
        
        return stats
    
    def calculate_rest_days(self, team: str, match_date: datetime) -> Dict:
        """Calcular días de descanso para el equipo"""
        # Combinar datos históricos
        all_matches = pd.concat([self.matches_2023, self.matches_2024, self.matches_2025])
        
        # Filtrar partidos del equipo
        team_matches = all_matches[
            ((all_matches['home_team'] == team) | (all_matches['away_team'] == team)) &
            (all_matches['status'] == 'FINISHED')
        ].sort_values('date', ascending=False)
        
        if len(team_matches) == 0:
            return {'rest_days': 7, 'matches_last_2_weeks': 0}  # Default
        
        # Último partido jugado
        last_match = team_matches[team_matches['date'] < match_date].iloc[0]
        days_since_last = (match_date - last_match['date']).days
        
        # Partidos en últimas 2 semanas
        two_weeks_ago = match_date - timedelta(days=14)
        recent_matches = team_matches[
            (team_matches['date'] >= two_weeks_ago) & 
            (team_matches['date'] < match_date)
        ]
        
        return {
            'rest_days': days_since_last,
            'matches_last_2_weeks': len(recent_matches),
            'fatigue_level': min(len(recent_matches) / 2.0, 3.0)  # Nivel de fatiga
        }
    
    def create_advanced_match_features(self, home_team: str, away_team: str, match_date: datetime) -> Dict:
        """Crear características avanzadas para un partido"""
        
        # 🔥 1. Diferencias entre equipos (OBLIGATORIO)
        home_standings = self.get_team_standings_stats(home_team)
        away_standings = self.get_team_standings_stats(away_team)
        
        # 👉 2. Forma reciente (últimos 5 partidos)
        home_form = self.get_recent_form_detailed(home_team, match_date)
        away_form = self.get_recent_form_detailed(away_team, match_date)
        
        # 🔥 3. Home advantage real (no binario)
        home_performance = self.get_home_away_performance_advanced(home_team, 'home', match_date)
        away_performance = self.get_home_away_performance_advanced(away_team, 'away', match_date)
        
        # 🔥 4. Días de descanso
        home_rest = self.calculate_rest_days(home_team, match_date)
        away_rest = self.calculate_rest_days(away_team, match_date)
        
        # Combinar todas las características CLAVE
        features = {
            # 🔥 1. DIFERENCIAS ENTRE EQUIPOS (OBLIGATORIO)
            'points_diff': home_standings['points'] - away_standings['points'],
            'gd_diff': home_standings['goal_difference'] - away_standings['goal_difference'],
            'position_diff': away_standings['position'] - home_standings['position'],
            'points_per_game_diff': home_standings['points_per_game'] - away_standings['points_per_game'],
            'gd_per_game_diff': home_standings['goal_difference_per_game'] - away_standings['goal_difference_per_game'],
            
            # 👉 2. FORMA RECIENTE (últimos 5 partidos)
            'home_last5_points': home_form['points'],
            'away_last5_points': away_form['points'],
            'home_last5_gd': home_form['goal_difference'],
            'away_last5_gd': away_form['goal_difference'],
            'form_diff_last5_points': home_form['points'] - away_form['points'],
            'form_diff_last5_gd': home_form['goal_difference'] - away_form['goal_difference'],
            'form_diff_last5_win_rate': home_form['win_rate'] - away_form['win_rate'],
            
            # 🔥 3. HOME ADVANTAGE REAL (no binario)
            'home_win_rate_home': home_performance['win_rate'],
            'away_win_rate_away': away_performance['win_rate'],
            'home_points_per_game_home': home_performance['points_per_game'],
            'away_points_per_game_away': away_performance['points_per_game'],
            'home_gd_per_game_home': home_performance['goal_difference_per_game'],
            'away_gd_per_game_away': away_performance['goal_difference_per_game'],
            'home_advantage_rate': home_performance['win_rate'] - away_performance['win_rate'],
            'home_advantage_points': home_performance['points_per_game'] - away_performance['points_per_game'],
            
            # 🔥 4. DÍAS DE DESCANSO
            'home_rest_days': home_rest['rest_days'],
            'away_rest_days': away_rest['rest_days'],
            'rest_diff': home_rest['rest_days'] - away_rest['rest_days'],
            'home_fatigue': home_rest['fatigue_level'],
            'away_fatigue': away_rest['fatigue_level'],
            'fatigue_diff': home_rest['fatigue_level'] - away_rest['fatigue_level'],
            
            # 6️⃣ Features secundarias importantes
            'home_unbeaten_streak': home_form['unbeaten_streak'],
            'away_unbeaten_streak': away_form['unbeaten_streak'],
            'home_unbeaten_diff': home_form['unbeaten_streak'] - away_form['unbeaten_streak'],
            
            # Clean sheets y scoring rates
            'home_clean_sheets_rate': home_performance['clean_sheets_rate'],
            'away_clean_sheets_rate': away_performance['clean_sheets_rate'],
            'home_scoring_rate': home_performance['scoring_rate'],
            'away_scoring_rate': away_performance['scoring_rate'],
            
            # Promedios de goles últimos 5
            'home_last5_goals_per_game': home_form['goals_per_game'],
            'away_last5_goals_per_game': away_form['goals_per_game'],
            'home_last5_goals_conceded_per_game': home_form['goals_conceded_per_game'],
            'away_last5_goals_conceded_per_game': away_form['goals_conceded_per_game'],
            
            # Información básica
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date,
            'is_home_advantage': 1  # Siempre 1 para el equipo local
        }
        
        return features
    
    def create_advanced_training_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Crear dataset de entrenamiento con características avanzadas"""
        print("Creando dataset AVANZADO de entrenamiento...")
        
        # Usar partidos finalizados de 2023, 2024 y parte de 2025
        training_matches = pd.concat([
            self.matches_2023[self.matches_2023['status'] == 'FINISHED'],
            self.matches_2024[self.matches_2024['status'] == 'FINISHED'],
            self.matches_2025[self.matches_2025['status'] == 'FINISHED']
        ])
        
        features_list = []
        targets_list = []
        
        for idx, match in training_matches.iterrows():
            if idx % 100 == 0:
                print(f"  Procesando partido {idx + 1}/{len(training_matches)}")
            
            try:
                # Crear características AVANZADAS
                features = self.create_advanced_match_features(
                    match['home_team'],
                    match['away_team'],
                    match['date']
                )
                
                # Crear target (resultado)
                if match['home_score'] > match['away_score']:
                    result = 2  # Victoria local
                elif match['home_score'] < match['away_score']:
                    result = 0  # Victoria visitante
                else:
                    result = 1  # Empate
                
                # Eliminar columnas no numéricas para el modelo
                numeric_features = {k: v for k, v in features.items() 
                                  if k not in ['home_team', 'away_team', 'match_date']}
                
                features_list.append(numeric_features)
                targets_list.append(result)
                
            except Exception as e:
                print(f"  Error procesando partido {idx}: {e}")
                continue
        
        print(f"Dataset AVANZADO creado: {len(features_list)} partidos")
        
        features_df = pd.DataFrame(features_list)
        targets_df = pd.Series(targets_list, name='result')
        
        # Contar características clave
        key_features = [f for f in features_df.columns if any(x in f for x in [
            'points_diff', 'gd_diff', 'position_diff',  # Diferencias
            'last5_points', 'last5_gd', 'form_diff',  # Forma reciente
            'home_win_rate_home', 'away_win_rate_away', 'home_advantage',  # Home advantage
            'rest_days', 'fatigue'  # Descanso
        ])]
        
        print(f"Features CLAVE implementadas: {len(key_features)}")
        print(f"Features totales: {len(features_df.columns)}")
        
        return features_df, targets_df
    
    def _calculate_unbeaten_streak(self, matches: pd.DataFrame, team: str) -> int:
        """Calcular racha de invicto"""
        streak = 0
        
        for _, match in matches.iterrows():
            is_home = match['home_team'] == team
            team_score = match['home_score'] if is_home else match['away_score']
            opponent_score = match['away_score'] if is_home else match['home_score']
            
            if team_score >= opponent_score:  # No perdió
                streak += 1
            else:
                break  # Perdió, terminar racha
        
        return streak
    
    def _get_default_standings(self) -> Dict:
        """Valores por defecto para posiciones de tabla"""
        return {
            'position': 20, 'points': 0, 'played_games': 0,
            'goal_difference': 0, 'points_per_game': 0.0, 'goal_difference_per_game': 0.0
        }
    
    def _get_default_form(self) -> Dict:
        """Valores por defecto para forma de equipo"""
        return {
            'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_scored': 0, 'goals_conceded': 0, 'points': 0,
            'goal_difference': 0, 'win_rate': 0.0, 'points_per_game': 0.0,
            'goals_per_game': 0.0, 'goals_conceded_per_game': 0.0,
            'goal_difference_per_game': 0.0, 'unbeaten_streak': 0,
            'home_specific': {'wins': 0, 'goals_for': 0, 'goals_against': 0},
            'away_specific': {'wins': 0, 'goals_for': 0, 'goals_against': 0}
        }
    
    def _get_default_venue_performance(self, venue: str) -> Dict:
        """Valores por defecto para rendimiento local/visitante"""
        return {
            'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_scored': 0, 'goals_conceded': 0, 'points': 0,
            'goal_difference': 0, 'win_rate': 0.0, 'points_per_game': 0.0,
            'goals_per_game': 0.0, 'goals_conceded_per_game': 0.0,
            'goal_difference_per_game': 0.0, 'clean_sheets': 0,
            'failed_to_score': 0, 'clean_sheets_rate': 0.0, 'scoring_rate': 0.0
        }

if __name__ == "__main__":
    # Prueba del feature engineering avanzado
    fe = AdvancedFeatureEngineer()
    
    if fe.load_data():
        print("Probando feature engineering AVANZADO...")
        
        # Probar con un partido real
        test_features = fe.create_advanced_match_features(
            "Arsenal FC",
            "Manchester City FC", 
            datetime(2025, 1, 15)
        )
        
        print("Características AVANZADAS creadas:")
        key_features = [f for f in test_features.keys() if any(x in f for x in [
            'points_diff', 'gd_diff', 'position_diff',
            'last5_points', 'last5_gd', 'form_diff',
            'home_advantage', 'rest_days'
        ])]
        
        for feature in key_features:
            print(f"  {feature}: {test_features[feature]}")
        
        print(f"Total características: {len(test_features)}")
        print(f"Características CLAVE: {len(key_features)}")