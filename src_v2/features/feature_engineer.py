"""
Feature Engineer
==============
Crea features para modelos de ML desde datos limpios.

Dependencias:
- pandas, numpy
- Depende de: data.cleaned

Salida:
- DataFrame con features numéricas para modelos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple


class FeatureEngineer:
    """Ingeniero de features para predicción de partidos"""
    
    def __init__(self, data_dir: str = "data/cleaned"):
        self.data_dir = Path(data_dir)
        self.matches_2023: pd.DataFrame = None
        self.matches_2024: pd.DataFrame = None
        self.matches_2025: pd.DataFrame = None
        self.standings_2025: pd.DataFrame = None
        self.teams: pd.DataFrame = None
    
    def load_data(self) -> bool:
        """Cargar todos los datos necesarios"""
        try:
            self.matches_2023 = self._load_csv("matches_2023_cleaned.csv")
            self.matches_2024 = self._load_csv("matches_2024_cleaned.csv")
            self.matches_2025 = self._load_csv("matches_2025_cleaned.csv")
            self.standings_2025 = self._load_csv("standings_2025_cleaned.csv")
            self.teams = self._load_csv("teams_cleaned.csv")
            
            for df in [self.matches_2023, self.matches_2024, self.matches_2025]:
                df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
            
            print(f"✅ Datos cargados: {len(self.matches_2023) + len(self.matches_2024) + len(self.matches_2025)} matches")
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Cargar CSV con validación"""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró: {path}")
        return pd.read_csv(path)
    
    def get_team_form(self, team: str, date: datetime, n_games: int = 5) -> Dict:
        """Calcular forma reciente de un equipo"""
        historical = pd.concat([self.matches_2023, self.matches_2024, self.matches_2025])
        
        team_matches = historical[
            ((historical['home_team'] == team) | (historical['away_team'] == team)) &
            (historical['date'] < date) &
            (historical['status'] == 'FINISHED')
        ].sort_values('date', ascending=False).head(n_games)
        
        if len(team_matches) == 0:
            return self._default_form()
        
        wins = draws = losses = goals_for = goals_against = 0
        
        for _, m in team_matches.iterrows():
            is_home = m['home_team'] == team
            team_goals = m['home_score'] if is_home else m['away_score']
            opp_goals = m['away_score'] if is_home else m['home_score']
            
            goals_for += team_goals
            goals_against += opp_goals
            
            if team_goals > opp_goals:
                wins += 1
            elif team_goals == opp_goals:
                draws += 1
            else:
                losses += 1
        
        return {
            'matches_played': len(team_matches),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'points': wins * 3 + draws,
            'win_rate': wins / len(team_matches),
            'goals_per_game': goals_for / len(team_matches),
            'goals_conceded_per_game': goals_against / len(team_matches),
        }
    
    def get_team_form_detailed(self, team: str, date: datetime, n_games: int = 5) -> Dict:
        """Forma detallada con rachas"""
        form = self.get_team_form(team, date, n_games)
        form.update(self._default_form())  # Fill defaults
        
        historical = pd.concat([self.matches_2023, self.matches_2024, self.matches_2025])
        
        team_matches = historical[
            ((historical['home_team'] == team) | (historical['away_team'] == team)) &
            (historical['date'] < date) &
            (historical['status'] == 'FINISHED')
        ].sort_values('date', ascending=False).head(n_games)
        
        unbeaten = 0
        for _, m in team_matches.iterrows():
            is_home = m['home_team'] == team
            team_goals = m['home_score'] if is_home else m['away_score']
            opp_goals = m['away_score'] if is_home else m['home_score']
            
            if team_goals >= opp_goals:
                unbeaten += 1
            else:
                break
        
        form['unbeaten_streak'] = unbeaten
        form['clean_sheets'] = sum(
            1 for _, m in team_matches.iterrows()
            if (m['away_score'] if m['home_team'] == team else m['home_score']) == 0
        )
        
        return form
    
    def get_venue_performance(self, team: str, venue: str, date: datetime, n_games: int = 10) -> Dict:
        """Rendimiento local/visitante"""
        all_matches = pd.concat([self.matches_2023, self.matches_2024, self.matches_2025])
        
        if venue == 'home':
            team_mask = all_matches['home_team'] == team
            team_score_col = 'home_score'
            opp_score_col = 'away_score'
        else:
            team_mask = all_matches['away_team'] == team
            team_score_col = 'away_score'
            opp_score_col = 'home_score'
        
        venue_matches = all_matches[
            team_mask &
            (all_matches['date'] < date) &
            (all_matches['status'] == 'FINISHED')
        ].sort_values('date', ascending=False).head(n_games)
        
        if len(venue_matches) == 0:
            return self._default_venue(venue)
        
        wins = draws = losses = goals_for = goals_against = 0
        
        for _, m in venue_matches.iterrows():
            goals_for += m[team_score_col]
            goals_against += m[opp_score_col]
            
            if m[team_score_col] > m[opp_score_col]:
                wins += 1
            elif m[team_score_col] == m[opp_score_col]:
                draws += 1
            else:
                losses += 1
        
        return {
            'matches_played': len(venue_matches),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'points': wins * 3 + draws,
            'win_rate': wins / len(venue_matches),
            'goals_per_game': goals_for / len(venue_matches),
            'goals_conceded_per_game': goals_against / len(venue_matches),
            'points_per_game': (wins * 3 + draws) / len(venue_matches),
        }
    
    def get_head_to_head(self, home: str, away: str, date: datetime, n_games: int = 5) -> Dict:
        """Estadísticas head-to-head"""
        all_matches = pd.concat([self.matches_2023, self.matches_2024, self.matches_2025])
        
        h2h = all_matches[
            (((all_matches['home_team'] == home) & (all_matches['away_team'] == away)) |
             ((all_matches['home_team'] == away) & (all_matches['away_team'] == home))) &
            (all_matches['date'] < date) &
            (all_matches['status'] == 'FINISHED')
        ].sort_values('date', ascending=False).head(n_games)
        
        if len(h2h) == 0:
            return {'matches': 0, 'home_wins': 0, 'away_wins': 0, 'draws': 0}
        
        home_wins = away_wins = draws = 0
        
        for _, m in h2h.iterrows():
            if m['home_team'] == home:
                if m['home_score'] > m['away_score']:
                    home_wins += 1
                elif m['home_score'] < m['away_score']:
                    away_wins += 1
                else:
                    draws += 1
            else:
                if m['away_score'] > m['home_score']:
                    home_wins += 1
                elif m['away_score'] < m['home_score']:
                    away_wins += 1
                else:
                    draws += 1
        
        return {'matches': len(h2h), 'home_wins': home_wins, 'away_wins': away_wins, 'draws': draws}
    
    def get_standings(self, team: str) -> Dict:
        """Posición en la tabla"""
        if self.standings_2025 is None:
            return self._default_standings()
        
        row = self.standings_2025[self.standings_2025['team'] == team]
        
        if len(row) == 0:
            return self._default_standings()
        
        s = row.iloc[0]
        return {
            'position': s['position'],
            'points': s['points'],
            'played': s['played_games'],
            'win_rate': s['win_rate'],
            'goals_for_pg': s['goals_for_per_game'],
            'goals_against_pg': s['goals_against_per_game'],
            'gd': s['goal_difference'],
            'points_pg': s['points_per_game'],
        }
    
    def get_rest_days(self, team: str, match_date: datetime) -> Dict:
        """Días de descanso"""
        all_matches = pd.concat([self.matches_2023, self.matches_2024, self.matches_2025])
        
        team_matches = all_matches[
            ((all_matches['home_team'] == team) | (all_matches['away_team'] == team)) &
            (all_matches['status'] == 'FINISHED')
        ].sort_values('date', ascending=False)
        
        if len(team_matches) == 0:
            return {'rest_days': 7, 'matches_2_weeks': 0, 'fatigue': 0}
        
        last_match = team_matches[team_matches['date'] < match_date]
        
        if len(last_match) == 0:
            return {'rest_days': 14, 'matches_2_weeks': 0, 'fatigue': 0}
        
        last_match = last_match.iloc[0]
        rest_days = (match_date - last_match['date']).days
        
        two_weeks_ago = match_date - timedelta(days=14)
        recent = team_matches[
            (team_matches['date'] >= two_weeks_ago) & 
            (team_matches['date'] < match_date)
        ]
        
        return {
            'rest_days': rest_days,
            'matches_2_weeks': len(recent),
            'fatigue': min(len(recent) / 2.0, 3.0),
        }
    
    def create_match_features(self, home: str, away: str, date: datetime) -> Dict:
        """Crear todas las features para un partido"""
        # Get individual stats
        home_form = self.get_team_form_detailed(home, date)
        away_form = self.get_team_form_detailed(away, date)
        
        home_venue = self.get_venue_performance(home, 'home', date)
        away_venue = self.get_venue_performance(away, 'away', date)
        
        home_stand = self.get_standings(home)
        away_stand = self.get_standings(away)
        
        home_rest = self.get_rest_days(home, date)
        away_rest = self.get_rest_days(away, date)
        
        h2h = self.get_head_to_head(home, away, date)
        
        # Build feature dict
        return {
            'home_team': home,
            'away_team': away,
            'date': date,
            
            # Standings differences
            'points_diff': home_stand['points'] - away_stand['points'],
            'gd_diff': home_stand['gd'] - away_stand['gd'],
            'position_diff': away_stand['position'] - home_stand['position'],
            'points_pg_diff': home_stand['points_pg'] - away_stand['points_pg'],
            
            # Form differences
            'form_points_diff': home_form['points'] - away_form['points'],
            'form_gd_diff': (home_form['goals_for'] - home_form['goals_against']) - (away_form['goals_for'] - away_form['goals_against']),
            'form_win_rate_diff': home_form['win_rate'] - away_form['win_rate'],
            
            # Home advantage
            'home_win_rate_home': home_venue['win_rate'],
            'away_win_rate_away': away_venue['win_rate'],
            'home_advantage_rate': home_venue['win_rate'] - away_venue['win_rate'],
            'home_points_pg_home': home_venue['points_per_game'],
            'away_points_pg_away': away_venue['points_per_game'],
            
            # Rest days
            'home_rest_days': home_rest['rest_days'],
            'away_rest_days': away_rest['rest_days'],
            'rest_diff': home_rest['rest_days'] - away_rest['rest_days'],
            'home_fatigue': home_rest['fatigue'],
            'away_fatigue': away_rest['fatigue'],
            
            # H2H
            'h2h_matches': h2h['matches'],
            'h2h_home_win_rate': h2h['home_wins'] / max(1, h2h['matches']),
            
            # Absolute values (for diversity)
            'home_position': home_stand['position'],
            'away_position': away_stand['position'],
            'home_win_rate': home_form['win_rate'],
            'away_win_rate': away_form['win_rate'],
            'home_gpg': home_form['goals_per_game'],
            'away_gpg': away_form['goals_per_game'],
            'home_gcpg': home_form['goals_conceded_per_game'],
            'away_gcpg': away_form['goals_conceded_per_game'],
            'home_unbeaten': home_form['unbeaten_streak'],
            'away_unbeaten': away_form['unbeaten_streak'],
            'home_clean_sheets': home_form['clean_sheets'],
            'away_clean_sheets': away_form['clean_sheets'],
            
            # Placeholder for home advantage (always 1 since we always predict home team)
            'is_home': 1,
        }
    
    def create_training_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Crear dataset para entrenamiento"""
        all_matches = pd.concat([
            self.matches_2023[self.matches_2023['status'] == 'FINISHED'],
            self.matches_2024[self.matches_2024['status'] == 'FINISHED'],
            self.matches_2025[self.matches_2025['status'] == 'FINISHED'],
        ])
        
        features_list = []
        targets = []
        
        for _, m in all_matches.iterrows():
            try:
                feats = self.create_match_features(m['home_team'], m['away_team'], m['date'])
                
                # Target: 0=away, 1=draw, 2=home
                if m['home_score'] > m['away_score']:
                    target = 2
                elif m['home_score'] < m['away_score']:
                    target = 0
                else:
                    target = 1
                
                # Remove non-numeric
                numeric_feats = {k: v for k, v in feats.items() 
                              if k not in ['home_team', 'away_team', 'date']}
                features_list.append(numeric_feats)
                targets.append(target)
                
            except Exception as e:
                continue
        
        print(f"✅ Dataset: {len(features_list)} samples, {len(features_list[0]) if features_list else 0} features")
        
        return pd.DataFrame(features_list), pd.Series(targets, name='result')
    
    def _default_form(self) -> Dict:
        """Default form values"""
        return {
            'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'points': 0,
            'win_rate': 0.0, 'goals_per_game': 0.0, 'goals_conceded_per_game': 0.0,
            'unbeaten_streak': 0, 'clean_sheets': 0,
        }
    
    def _default_venue(self, venue: str) -> Dict:
        """Default venue performance"""
        return {
            'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'points': 0,
            'win_rate': 0.0, 'goals_per_game': 0.0, 'goals_conceded_per_game': 0.0,
            'points_per_game': 0.0,
        }
    
    def _default_standings(self) -> Dict:
        """Default standings"""
        return {
            'position': 20, 'points': 0, 'played': 0,
            'win_rate': 0.0, 'goals_for_pg': 0.0, 'goals_against_pg': 0.0,
            'gd': 0, 'points_pg': 0.0,
        }


def get_feature_engineer(data_dir: str = "data/cleaned") -> FeatureEngineer:
    """Factory function"""
    return FeatureEngineer(data_dir)