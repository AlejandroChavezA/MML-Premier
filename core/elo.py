"""Tracker ELO incremental para una sola liga (no es un port -- MML-Mundial's
ELO es específico de selecciones nacionales, ver docs/plan_5_ligas_ligamx.md).

Se instancia siempre a partir de los partidos de UNA liga (un solo data_dir),
nunca compartido entre ligas -- eso es lo que satisface "ELO aislada por liga"
(Paso 4). Se usa para sembrar el prior de Dixon-Coles en equipos con poco
historial (recién ascendidos, temporada nueva), no como señal de predicción
por sí sola en este paso.
"""
from typing import Dict, Optional

import pandas as pd


class EloTracker:
    def __init__(self, k_factor: float = 20.0, home_advantage: float = 60.0,
                 initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))

    def _update_match(self, home_team: str, away_team: str, home_score: float, away_score: float):
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        expected_home = self._expected_score(home_rating + self.home_advantage, away_rating)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        self.ratings[home_team] = home_rating + self.k_factor * (actual_home - expected_home)
        self.ratings[away_team] = away_rating + self.k_factor * ((1.0 - actual_home) - (1.0 - expected_home))

    def fit(self, matches_df: pd.DataFrame) -> "EloTracker":
        """Recorre partidos FINISHED en orden cronológico, actualizando ratings."""
        finished = matches_df[matches_df['status'].astype(str).str.upper() == 'FINISHED'].copy()
        finished['date'] = pd.to_datetime(finished['date'], utc=True).dt.tz_localize(None)
        finished = finished.sort_values('date')

        for _, m in finished.iterrows():
            self._update_match(m['home_team'], m['away_team'], m['home_score'], m['away_score'])

        return self

    def snapshot(self) -> Dict[str, float]:
        return dict(self.ratings)
