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
import re
import glob
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

from season_utils import get_current_season


class FeatureEngineer:
    """Ingeniero de features para predicción de partidos"""
    
    def __init__(self, data_dir: str = "data/cleaned"):
        self.data_dir = Path(data_dir)
        self.matches_by_season: Dict[int, pd.DataFrame] = {}
        self.standings_by_season: Dict[int, pd.DataFrame] = {}
        self.current_season = None
        self.teams: pd.DataFrame = None
    
    def load_data(self) -> bool:
        """Cargar todos los datos necesarios (todas las temporadas disponibles)."""
        try:
            self.matches_by_season = {}
            self.standings_by_season = {}

            for f in glob.glob(str(self.data_dir / "matches_*_cleaned.csv")):
                m = re.search(r"matches_(\d{4})_cleaned\.csv", os.path.basename(f))
                if not m:
                    continue
                year = int(m.group(1))
                df = pd.read_csv(f)
                df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
                df['status'] = df['status'].astype(str).str.strip().str.upper()
                self.matches_by_season[year] = df

            for f in glob.glob(str(self.data_dir / "standings_*_cleaned.csv")):
                m = re.search(r"standings_(\d{4})_cleaned\.csv", os.path.basename(f))
                if not m:
                    continue
                year = int(m.group(1))
                self.standings_by_season[year] = pd.read_csv(f)

            if not self.matches_by_season:
                raise FileNotFoundError("No se encontraron archivos matches_*_cleaned.csv")

            # current_season = ultima temporada CON partidos FINISHED (no la maxima cargada,
            # que podria ser la proxima temporada vacia, ej. 2026 con played_games=0)
            finished_by_season = {
                y: int((df['status'] == 'FINISHED').sum())
                for y, df in self.matches_by_season.items()
            }
            candidates = [y for y, n in finished_by_season.items() if n > 0]
            self.current_season = max(candidates) if candidates else max(self.matches_by_season.keys())

            # Standings de referencia = current_season; si tiene NaN (temporada vacia),
            # usar la ultima standings con datos reales.
            self._standings_ref_season = self.current_season
            if self.current_season in self.standings_by_season:
                ref = self.standings_by_season[self.current_season]
                if ref['points_per_game'].isna().all():
                    finished_standings = [y for y in self.standings_by_season
                                          if not self.standings_by_season[y]['points_per_game'].isna().all()]
                    if finished_standings:
                        self._standings_ref_season = max(finished_standings)

            teams_path = self.data_dir / "teams_cleaned.csv"
            if teams_path.exists():
                self.teams = self._load_csv("teams_cleaned.csv")
            else:
                # Liga sin teams_cleaned.csv propio (ej. Liga MX): derivar la lista
                # de equipos directamente de los partidos ya cargados.
                all_teams = set()
                for df in self.matches_by_season.values():
                    all_teams |= set(df['home_team']) | set(df['away_team'])
                fallback_teams = sorted(all_teams)
                self.teams = pd.DataFrame({'name': fallback_teams, 'name_clean': fallback_teams})

            print(f"✅ Datos cargados: temporadas {sorted(self.matches_by_season)} | "
                  f"actual: {self.current_season}")
            return True
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False

    def _historical_matches(self):
        return pd.concat([
            v for k, v in self.matches_by_season.items() if k != self.current_season
        ]) if len(self.matches_by_season) > 1 else pd.DataFrame()

    def _all_matches(self):
        return pd.concat(list(self.matches_by_season.values()))

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Cargar CSV con validación"""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró: {path}")
        return pd.read_csv(path)
    
    def get_team_form(self, team: str, date: datetime, n_games: int = 5) -> Dict:
        """Calcular forma reciente de un equipo"""
        historical = self._all_matches()
        
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
        # Completar solo las claves faltantes sin pisar los valores reales
        defaults = self._default_form()
        defaults.update(form)
        form = defaults
        
        historical = self._all_matches()
        
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
        all_matches = self._all_matches()
        
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
        all_matches = self._all_matches()
        
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
    
    def _standings_table(self, finished: pd.DataFrame) -> pd.DataFrame:
        """Tabla de posiciones calculada al vuelo desde un set de partidos FINISHED
        (misma logica que _build_standings en los scripts de build de Liga MX)."""
        cols = ['team', 'position', 'played', 'points', 'gd', 'win_rate', 'points_pg', 'gf_pg', 'ga_pg']
        if finished.empty:
            return pd.DataFrame(columns=cols)

        teams = sorted(set(finished['home_team']) | set(finished['away_team']))
        rows = []
        for team in teams:
            home = finished[finished['home_team'] == team]
            away = finished[finished['away_team'] == team]
            played = len(home) + len(away)
            wins = int((home['home_score'] > home['away_score']).sum() + (away['away_score'] > away['home_score']).sum())
            draws = int((home['home_score'] == home['away_score']).sum() + (away['away_score'] == away['home_score']).sum())
            gf = int(home['home_score'].sum() + away['away_score'].sum())
            ga = int(home['away_score'].sum() + away['home_score'].sum())
            points = wins * 3 + draws
            rows.append({
                'team': team, 'played': played, 'points': points, 'gd': gf - ga,
                'win_rate': wins / played if played else 0.0,
                'points_pg': points / played if played else 0.0,
                'gf_pg': gf / played if played else 0.0,
                'ga_pg': ga / played if played else 0.0,
            })

        table = pd.DataFrame(rows)
        table.sort_values(['points', 'gd', 'gf_pg', 'team'], ascending=[False, False, False, True], inplace=True)
        table.reset_index(drop=True, inplace=True)
        table.insert(1, 'position', range(1, len(table) + 1))
        return table

    def get_standings(self, team: str, date: datetime = None, season_year: int = None, phase: str = None) -> Dict:
        """Posicion en la tabla, point-in-time.

        Con `date` + `season_year`: calcula la tabla acumulada SOLO con partidos
        FINISHED de esa temporada (y esa fase, si la liga tiene Apertura/Clausura)
        anteriores a `date` -- asi un partido de 2015 usa la tabla de 2015, no la
        tabla de la temporada actual. Sin esos argumentos (prediccion en vivo de
        un partido futuro, donde no hay "fuga" posible porque de verdad queremos
        el estado mas reciente conocido) usa la temporada de referencia completa,
        igual que antes.
        """
        if date is None or season_year is None:
            ref_season = getattr(self, '_standings_ref_season', self.current_season)
            if ref_season not in self.matches_by_season:
                return self._default_standings()
            season_df = self.matches_by_season[ref_season]
            finished = season_df[season_df['status'] == 'FINISHED']
        else:
            if season_year not in self.matches_by_season:
                return self._default_standings()
            season_df = self.matches_by_season[season_year]
            finished = season_df[(season_df['status'] == 'FINISHED') & (season_df['date'] < date)]
            if phase is not None and 'phase' in season_df.columns:
                finished = finished[finished['phase'] == phase]

        table = self._standings_table(finished)
        row = table[table['team'] == team]
        if row.empty:
            return self._default_standings()

        r = row.iloc[0]
        return {
            'position': int(r['position']),
            'points': int(r['points']),
            'played': int(r['played']),
            'win_rate': float(r['win_rate']),
            'goals_for_pg': float(r['gf_pg']),
            'goals_against_pg': float(r['ga_pg']),
            'gd': int(r['gd']),
            'points_pg': float(r['points_pg']),
        }
    
    def get_rest_days(self, team: str, match_date: datetime) -> Dict:
        """Días de descanso"""
        all_matches = self._all_matches()
        
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
    
    def create_match_features(self, home: str, away: str, date: datetime,
                               season_year: int = None, phase: str = None) -> Dict:
        """Crear todas las features para un partido.

        `season_year`/`phase` son opcionales: al entrenar (build_training_dataset)
        se pasan para que `get_standings` calcule la tabla point-in-time de la
        temporada/fase real del partido. En serving en vivo (prediccion de un
        partido futuro) se dejan en None a proposito -- ver docstring de
        get_standings.
        """
        # Get individual stats
        home_form = self.get_team_form_detailed(home, date)
        away_form = self.get_team_form_detailed(away, date)

        home_venue = self.get_venue_performance(home, 'home', date)
        away_venue = self.get_venue_performance(away, 'away', date)

        home_stand = self.get_standings(home, date=date, season_year=season_year, phase=phase)
        away_stand = self.get_standings(away, date=date, season_year=season_year, phase=phase)
        
        home_rest = self.get_rest_days(home, date)
        away_rest = self.get_rest_days(away, date)
        
        h2h = self.get_head_to_head(home, away, date)
        
        # Build feature dict
        features = {
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
        # Defensa: rellenar cualquier NaN restante con 0 (antes esto estaba
        # despues de un return y nunca se ejecutaba -- ver docs de la sesion).
        for k, v in features.items():
            if isinstance(v, float) and pd.isna(v):
                features[k] = 0.0
        return features

    def create_training_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Crear dataset para entrenamiento"""
        all_matches = pd.concat([
            df[df['status'] == 'FINISHED'].assign(_season_year=year)
            for year, df in self.matches_by_season.items()
        ])

        features_list = []
        targets = []

        for _, m in all_matches.iterrows():
            try:
                phase = m['phase'] if 'phase' in m.index and pd.notna(m['phase']) else None
                feats = self.create_match_features(m['home_team'], m['away_team'], m['date'],
                                                    season_year=int(m['_season_year']), phase=phase)

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
            'goal_difference': 0, 'goal_difference_per_game': 0.0,
            'win_rate': 0.0, 'points_per_game': 0.0,
            'goals_per_game': 0.0, 'goals_conceded_per_game': 0.0,
            'unbeaten_streak': 0, 'clean_sheets': 0,
        }
    
    def _default_venue(self, venue: str) -> Dict:
        """Default venue performance"""
        return {
            'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'points': 0,
            'goal_difference': 0, 'goal_difference_per_game': 0.0,
            'win_rate': 0.0, 'goals_per_game': 0.0, 'goals_conceded_per_game': 0.0,
            'points_per_game': 0.0, 'clean_sheets_rate': 0.0, 'scoring_rate': 0.0,
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
