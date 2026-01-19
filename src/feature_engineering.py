import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class FeatureEngineer:
    def __init__(self, data_dir: str = "../data/cleaned"):
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
            
            # Convertir fechas
            for df in [self.matches_2023, self.matches_2024, self.matches_2025]:
                df['date'] = pd.to_datetime(df['date'])
                
            print("Datos cargados exitosamente")
            return True
        except Exception as e:
            print(f" Error cargando datos: {e}")
            return False
    
    def calculate_team_form_detailed(self, team: str, date: datetime, last_n_games: int = 5) -> Dict:
        """Forma detallada con unbeaten streak"""
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
            'clean_sheets': 0,
            'failed_to_score': 0
        }
        
        # Analizar cada partido
        for _, match in recent_matches.iterrows():
            is_home = match['home_team'] == team
            team_score = match['home_score'] if is_home else match['away_score']
            opponent_score = match['away_score'] if is_home else match['home_score']
            
            # Actualizar estadísticas
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
            'failed_to_score': 0,
            'clean_sheets_rate': 0.0,
            'scoring_rate': 0.0
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
            return {'rest_days': 7, 'matches_last_2_weeks': 0, 'fatigue_level': 0.0}  # Default
        
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
    
    def calculate_team_form(self, team: str, date: datetime, last_n_games: int = 5) -> Dict:
        """Calcular forma de un equipo hasta una fecha específica"""
        # Combinar datos históricos (2023, 2024) y partidos de 2025 antes de la fecha
        historical_matches = pd.concat([self.matches_2023, self.matches_2024])
        
        # Filtrar partidos del equipo antes de la fecha
        team_matches = []
        
        # Partidos históricos
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
        
        # Calcular estadísticas
        form_stats = {
            'matches_played': len(recent_matches),
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'points': 0,
            'win_rate': 0.0,
            'goals_per_game': 0.0,
            'goals_conceded_per_game': 0.0,
            'clean_sheets': 0,
            'failed_to_score': 0
        }
        
        for _, match in recent_matches.iterrows():
            is_home = match['home_team'] == team
            team_score = match['home_score'] if is_home else match['away_score']
            opponent_score = match['away_score'] if is_home else match['home_score']
            
            # Resultados
            if team_score > opponent_score:
                form_stats['wins'] += 1
                form_stats['points'] += 3
            elif team_score == opponent_score:
                form_stats['draws'] += 1
                form_stats['points'] += 1
            else:
                form_stats['losses'] += 1
            
            # Goles
            form_stats['goals_scored'] += team_score
            form_stats['goals_conceded'] += opponent_score
            
            # Clean sheets y no marcar
            if opponent_score == 0:
                form_stats['clean_sheets'] += 1
            if team_score == 0:
                form_stats['failed_to_score'] += 1
        
        # Calcular tasas
        if form_stats['matches_played'] > 0:
            form_stats['win_rate'] = form_stats['wins'] / form_stats['matches_played']
            form_stats['goals_per_game'] = form_stats['goals_scored'] / form_stats['matches_played']
            form_stats['goals_conceded_per_game'] = form_stats['goals_conceded'] / form_stats['matches_played']
        
        return form_stats
    
    def get_head_to_head_stats(self, home_team: str, away_team: str, date: datetime, last_n_games: int = 5) -> Dict:
        """Estadísticas head-to-head entre dos equipos"""
        # Combinar datos históricos
        all_matches = pd.concat([self.matches_2023, self.matches_2024, self.matches_2025])
        
        # Filtrar partidos entre los equipos antes de la fecha
        h2h_mask = (
            ((all_matches['home_team'] == home_team) & (all_matches['away_team'] == away_team)) |
            ((all_matches['home_team'] == away_team) & (all_matches['away_team'] == home_team))
        )
        date_mask = all_matches['date'] < date
        finished_mask = all_matches['status'] == 'FINISHED'
        
        h2h_matches = all_matches[h2h_mask & date_mask & finished_mask].sort_values('date', ascending=False)
        
        if len(h2h_matches) == 0:
            return self._get_default_h2h()
        
        # Tomar últimos N partidos
        recent_h2h = h2h_matches.head(last_n_games)
        
        h2h_stats = {
            'matches_played': len(recent_h2h),
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0,
            'home_goals': 0,
            'away_goals': 0,
            'home_win_rate': 0.0,
            'avg_total_goals': 0.0
        }
        
        for _, match in recent_h2h.iterrows():
            is_home_first = match['home_team'] == home_team
            
            if is_home_first:
                home_score = match['home_score']
                away_score = match['away_score']
            else:
                home_score = match['away_score']
                away_score = match['home_score']
            
            # Resultados
            if home_score > away_score:
                h2h_stats['home_wins'] += 1
            elif home_score < away_score:
                h2h_stats['away_wins'] += 1
            else:
                h2h_stats['draws'] += 1
            
            # Goles
            h2h_stats['home_goals'] += home_score
            h2h_stats['away_goals'] += away_score
        
        # Calcular tasas
        if h2h_stats['matches_played'] > 0:
            h2h_stats['home_win_rate'] = h2h_stats['home_wins'] / h2h_stats['matches_played']
            h2h_stats['avg_total_goals'] = (h2h_stats['home_goals'] + h2h_stats['away_goals']) / h2h_stats['matches_played']
        
        return h2h_stats
    
    def get_home_away_performance(self, team: str, venue: str, date: datetime, last_n_games: int = 10) -> Dict:
        """Rendimiento específico local/visitante"""
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
        
        recent_venue = venue_matches.head(last_n_games)
        
        venue_stats = {
            'matches_played': len(recent_venue),
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'points': 0,
            'win_rate': 0.0,
            'goals_per_game': 0.0,
            'goals_conceded_per_game': 0.0,
            'points_per_game': 0.0
        }
        
        for _, match in recent_venue.iterrows():
            team_score = match[team_score_col]
            opponent_score = match[opponent_score_col]
            
            # Resultados
            if team_score > opponent_score:
                venue_stats['wins'] += 1
                venue_stats['points'] += 3
            elif team_score == opponent_score:
                venue_stats['draws'] += 1
                venue_stats['points'] += 1
            else:
                venue_stats['losses'] += 1
            
            # Goles
            venue_stats['goals_scored'] += team_score
            venue_stats['goals_conceded'] += opponent_score
        
        # Calcular tasas
        if venue_stats['matches_played'] > 0:
            venue_stats['win_rate'] = venue_stats['wins'] / venue_stats['matches_played']
            venue_stats['goals_per_game'] = venue_stats['goals_scored'] / venue_stats['matches_played']
            venue_stats['goals_conceded_per_game'] = venue_stats['goals_conceded'] / venue_stats['matches_played']
            venue_stats['points_per_game'] = venue_stats['points'] / venue_stats['matches_played']
        
        return venue_stats
    
    def get_current_standings_position(self, team: str) -> Dict:
        """Posición actual en la tabla"""
        if self.standings_2025 is None:
            return self._get_default_standings()
        
        team_row = self.standings_2025[self.standings_2025['team'] == team]
        
        if len(team_row) == 0:
            return self._get_default_standings()
        
        team_stats = team_row.iloc[0]
        
        return {
            'position': team_stats['position'],
            'points': team_stats['points'],
            'played_games': team_stats['played_games'],
            'win_rate': team_stats['win_rate'],
            'goals_for_per_game': team_stats['goals_for_per_game'],
            'goals_against_per_game': team_stats['goals_against_per_game'],
            'goal_difference': team_stats['goal_difference'],
            'points_per_game': team_stats['points_per_game']
        }
    
    def create_match_features(self, home_team: str, away_team: str, match_date: datetime) -> Dict:
        """Crear todas las características para un partido"""
        # Forma de equipos (versión mejorada)
        home_form = self.calculate_team_form_detailed(home_team, match_date)
        away_form = self.calculate_team_form_detailed(away_team, match_date)
        
        # Head-to-head
        h2h_stats = self.get_head_to_head_stats(home_team, away_team, match_date)
        
        # Rendimiento local/visitante (versión mejorada)
        home_venue = self.get_home_away_performance_advanced(home_team, 'home', match_date)
        away_venue = self.get_home_away_performance_advanced(away_team, 'away', match_date)
        
        # Posiciones en tabla
        home_standings = self.get_current_standings_position(home_team)
        away_standings = self.get_current_standings_position(away_team)
        
        # Días de descanso
        home_rest = self.calculate_rest_days(home_team, match_date)
        away_rest = self.calculate_rest_days(away_team, match_date)
        
        # Combinar todas las características con las CLAVE nuevas
        features = {
            # Características básicas
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date,
            
            # 🔥 1. DIFERENCIAS ENTRE EQUIPOS (OBLIGATORIO)
            'points_diff': home_standings['points'] - away_standings['points'],
            'gd_diff': home_standings['goal_difference'] - away_standings['goal_difference'],
            'position_diff': away_standings['position'] - home_standings['position'],
            'points_per_game_diff': home_standings['points_per_game'] - away_standings['points_per_game'],
            'gd_per_game_diff': (home_standings['goal_difference'] / max(1, home_standings['played_games'])) - 
                              (away_standings['goal_difference'] / max(1, away_standings['played_games'])),
            
            # 👉 2. FORMA RECIENTE (últimos 5 partidos)
            'home_last5_points': home_form['points'],
            'away_last5_points': away_form['points'],
            'home_last5_gd': home_form['goal_difference'],
            'away_last5_gd': away_form['goal_difference'],
            'form_diff_last5_points': home_form['points'] - away_form['points'],
            'form_diff_last5_gd': home_form['goal_difference'] - away_form['goal_difference'],
            'form_diff_last5_win_rate': home_form['win_rate'] - away_form['win_rate'],
            
            # Forma general (mantenemos las originales también)
            'home_form_win_rate': home_form['win_rate'],
            'home_form_points_per_game': home_form['points'] / max(1, home_form['matches_played']),
            'home_form_goals_per_game': home_form['goals_per_game'],
            'home_form_goals_conceded_per_game': home_form['goals_conceded_per_game'],
            'home_form_clean_sheets_rate': home_form['clean_sheets'] / max(1, home_form['matches_played']),
            
            'away_form_win_rate': away_form['win_rate'],
            'away_form_points_per_game': away_form['points'] / max(1, away_form['matches_played']),
            'away_form_goals_per_game': away_form['goals_per_game'],
            'away_form_goals_conceded_per_game': away_form['goals_conceded_per_game'],
            'away_form_clean_sheets_rate': away_form['clean_sheets'] / max(1, away_form['matches_played']),
            
            # 🔥 3. HOME ADVANTAGE REAL (no binario)
            'home_win_rate_home': home_venue['win_rate'],
            'away_win_rate_away': away_venue['win_rate'],
            'home_points_per_game_home': home_venue['points_per_game'],
            'away_points_per_game_away': away_venue['points_per_game'],
            'home_gd_per_game_home': home_venue['goal_difference_per_game'],
            'away_gd_per_game_away': away_venue['goal_difference_per_game'],
            'home_advantage_rate': home_venue['win_rate'] - away_venue['win_rate'],
            'home_advantage_points': home_venue['points_per_game'] - away_venue['points_per_game'],
            
            # Rendimiento específico local/visitante (mantenemos originales)
            'home_venue_win_rate': home_venue['win_rate'],
            'home_venue_points_per_game': home_venue['points_per_game'],
            'home_venue_goals_per_game': home_venue['goals_per_game'],
            'home_venue_goals_conceded_per_game': home_venue['goals_conceded_per_game'],
            
            'away_venue_win_rate': away_venue['win_rate'],
            'away_venue_points_per_game': away_venue['points_per_game'],
            'away_venue_goals_per_game': away_venue['goals_per_game'],
            'away_venue_goals_conceded_per_game': away_venue['goals_conceded_per_game'],
            
            # 🔥 4. DÍAS DE DESCANSO
            'home_rest_days': home_rest['rest_days'],
            'away_rest_days': away_rest['rest_days'],
            'rest_diff': home_rest['rest_days'] - away_rest['rest_days'],
            'home_fatigue': home_rest['fatigue_level'],
            'away_fatigue': away_rest['fatigue_level'],
            'fatigue_diff': home_rest['fatigue_level'] - away_rest['fatigue_level'],
            
            # Features secundarias importantes
            'home_unbeaten_streak': home_form['unbeaten_streak'],
            'away_unbeaten_streak': away_form['unbeaten_streak'],
            'home_unbeaten_diff': home_form['unbeaten_streak'] - away_form['unbeaten_streak'],
            
            # Clean sheets y scoring rates
            'home_clean_sheets_rate': home_venue['clean_sheets_rate'],
            'away_clean_sheets_rate': away_venue['clean_sheets_rate'],
            'home_scoring_rate': home_venue['scoring_rate'],
            'away_scoring_rate': away_venue['scoring_rate'],
            
            # Head-to-head (mantenemos originales)
            'h2h_home_win_rate': h2h_stats['home_win_rate'],
            'h2h_avg_total_goals': h2h_stats['avg_total_goals'],
            'h2h_matches_played': h2h_stats['matches_played'],
            
            # Posiciones en tabla (mantenemos originales)
            'home_standings_position': home_standings['position'],
            'home_standings_points_per_game': home_standings['points_per_game'],
            'home_standings_goal_difference': home_standings['goal_difference'],
            
            'away_standings_position': away_standings['position'],
            'away_standings_points_per_game': away_standings['points_per_game'],
            'away_standings_goal_difference': away_standings['goal_difference'],
            
            # Diferenciales entre equipos (mantenemos algunos originales)
            'form_win_rate_diff': home_form['win_rate'] - away_form['win_rate'],
            'form_goals_diff': home_form['goals_per_game'] - away_form['goals_per_game'],
            'venue_win_rate_diff': home_venue['win_rate'] - away_venue['win_rate'],
            'standings_position_diff': away_standings['position'] - home_standings['position'],
            'points_per_game_diff': home_standings['points_per_game'] - away_standings['points_per_game'],
            
            # Identificador de ventaja local
            'is_home_advantage': 1
        }
        
        return features
    
    def create_training_dataset(self) -> pd.DataFrame:
        """Crear dataset de entrenamiento con todas las características"""
        print("Creando dataset de entrenamiento...")
        
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
                # Crear características MEJORADAS
                features = self.create_match_features(
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
        
        print(f" Dataset creado: {len(features_list)} partidos")
        
        features_df = pd.DataFrame(features_list)
        targets_df = pd.Series(targets_list, name='result')
        
        return features_df, targets_df
    
    def _get_default_form(self) -> Dict:
        """Valores por defecto para forma de equipo"""
        return {
            'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_scored': 0, 'goals_conceded': 0, 'points': 0,
            'win_rate': 0.0, 'goals_per_game': 0.0, 'goals_conceded_per_game': 0.0,
            'clean_sheets': 0, 'failed_to_score': 0
        }
    
    def _get_default_h2h(self) -> Dict:
        """Valores por defecto para head-to-head"""
        return {
            'matches_played': 0, 'home_wins': 0, 'away_wins': 0, 'draws': 0,
            'home_goals': 0, 'away_goals': 0, 'home_win_rate': 0.0, 'avg_total_goals': 0.0
        }
    
    def _get_default_venue_performance(self, venue: str) -> Dict:
        """Valores por defecto para rendimiento local/visitante"""
        return {
            'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_scored': 0, 'goals_conceded': 0, 'points': 0,
            'win_rate': 0.0, 'goals_per_game': 0.0, 'goals_conceded_per_game': 0.0,
            'points_per_game': 0.0
        }
    
    def _get_default_standings(self) -> Dict:
        """Valores por defecto para posiciones de tabla"""
        return {
            'position': 20, 'points': 0, 'played_games': 0,
            'win_rate': 0.0, 'goals_for_per_game': 0.0,
            'goals_against_per_game': 0.0, 'goal_difference': 0,
            'points_per_game': 0.0
        }

if __name__ == "__main__":
    # Prueba del feature engineering
    fe = FeatureEngineer()
    
    if fe.load_data():
        print(" Probando feature engineering...")
        
        # Probar con un partido real
        test_features = fe.create_match_features(
            "Arsenal FC",
            "Manchester City FC", 
            datetime(2025, 1, 15)
        )
        
        print(" Características creadas:")
        for key, value in list(test_features.items())[:10]:
            print(f"  {key}: {value}")
        
        print(f"  Total características: {len(test_features)}")