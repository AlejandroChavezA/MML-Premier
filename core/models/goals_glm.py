"""Modelo de goles totales / Over-Under del ensamble -- GLM Poisson.

Interfaz paralela a src_v2/models/goals_predictor.py's GoalsPredictor, pero
con sklearn.linear_model.PoissonRegressor (GLM Poisson real) en vez de
RandomForestRegressor -- evita agregar statsmodels como dependencia nueva.
La conversión a probabilidades Over/Under reusa la misma fórmula Poisson-CDF
que GoalsPredictor; solo cambia el estimador de goles esperados upstream.
"""
import pickle
from math import exp, factorial
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from core.eval_utils import chronological_split

THRESHOLDS = [0.5, 1.5, 2.5, 3.5]


def _poisson_over(lamb: float, threshold: float) -> float:
    lamb = max(0.1, lamb)
    k = int(threshold)
    prob_under = sum(exp(-lamb) * (lamb ** i) / factorial(i) for i in range(k))
    return 1 - prob_under


class GoalsGLM:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model: Optional[PoissonRegressor] = None
        self.feature_cols: List[str] = []
        self.performance: Dict = {}

    def train(self, features_df: pd.DataFrame, targets: pd.Series, dates: Optional[pd.Series] = None) -> bool:
        self.feature_cols = features_df.columns.tolist()

        if dates is not None:
            # Split cronológico: ver core/eval_utils.py y OutcomeXGB.train().
            X_train, X_test, y_train, y_test, _ = chronological_split(features_df, targets, dates)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, targets, test_size=0.2, random_state=42
            )

        self.model = PoissonRegressor(alpha=1.0, max_iter=500)
        self.model.fit(X_train.fillna(0), y_train)

        train_pred = self.model.predict(X_train.fillna(0))
        test_pred = self.model.predict(X_test.fillna(0))

        self.performance = {
            'train_mae': float(mean_absolute_error(y_train, train_pred)),
            'test_mae': float(mean_absolute_error(y_test, test_pred)),
        }

        # Igual que OutcomeXGB.train(): el split es solo para medir performance
        # honesta -- el modelo que se guarda se reentrena con todo el dataset.
        if dates is not None:
            self.model.fit(features_df.fillna(0), targets)

        self._save()
        return True

    def _save(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.models_dir / "goals_glm.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.models_dir / "goals_glm_features.pkl", 'wb') as f:
            pickle.dump(self.feature_cols, f)

    def load(self) -> bool:
        model_path = self.models_dir / "goals_glm.pkl"
        if not model_path.exists():
            return False
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.models_dir / "goals_glm_features.pkl", 'rb') as f:
                self.feature_cols = pickle.load(f)
            return True
        except Exception:
            return False

    def predict(self, home_team: str, away_team: str, match_date, feature_engineer) -> Dict:
        if self.model is None:
            return {'error': 'Modelo no cargado'}
        if feature_engineer is None:
            return {'error': 'Feature engineer no proveído'}

        try:
            if hasattr(match_date, 'tzinfo') and match_date.tzinfo:
                match_date = match_date.replace(tzinfo=None)

            features = feature_engineer.create_match_features(home_team, away_team, match_date)
            df = pd.DataFrame([features])
            numeric = df[self.feature_cols].fillna(0)

            expected = max(0.1, float(self.model.predict(numeric)[0]))

            markets = {}
            for thresh in THRESHOLDS:
                over_prob = _poisson_over(expected, thresh)
                markets[f'over_{thresh}'] = {
                    'over_prob': over_prob,
                    'under_prob': 1 - over_prob,
                    'prediction': 'OVER' if over_prob > 0.5 else 'UNDER',
                    'confidence': abs(over_prob - 0.5) * 2,
                }

            return {
                'home_team': home_team,
                'away_team': away_team,
                'expected_goals': expected,
                'markets': markets,
            }
        except Exception as e:
            return {'error': str(e)}

    def get_performance(self) -> Dict:
        return self.performance
