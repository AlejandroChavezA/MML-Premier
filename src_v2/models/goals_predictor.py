"""
Goals Predictor
==============
Predice número de goles y Over/Under.

Dependencias:
- sklearn (RandomForestRegressor)
- pickle, math

Usa:
- features.feature_engineer
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from pathlib import Path
from typing import Dict
from math import exp, factorial


class GoalsPredictor:
    """Predice Over/Under en goles"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.feature_cols = []
        self.performance = {}
    
    def train(self, features_df: pd.DataFrame, targets: pd.Series) -> bool:
        """Entrenar modelo de regresión"""
        print("\n⚽ ENTRENANDO GOALS PREDICTOR")
        print("=" * 50)
        
        self.feature_cols = features_df.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets, test_size=0.2, random_state=42
        )
        
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Avg goals: {y_train.mean():.2f}")
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Test MAE: {test_mae:.2f}")
        
        self._save()
        return True
    
    def _save(self):
        """Guardar modelo"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.models_dir / "goals_predictor.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.models_dir / "goals_features.pkl", 'wb') as f:
            pickle.dump(self.feature_cols, f)
        
        print(f"\n💾 Guardado en {self.models_dir}")
    
    def load(self) -> bool:
        """Cargar modelo"""
        model_path = self.models_dir / "goals_predictor.pkl"
        
        if not model_path.exists():
            return False
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.models_dir / "goals_features.pkl", 'rb') as f:
                self.feature_cols = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error cargando: {e}")
            return False
    
    def predict(self, home_team: str, away_team: str,
                match_date, feature_engineer) -> Dict:
        """Predecir goles y Over/Under"""
        if self.model is None:
            return {'error': 'Modelo no cargado'}
        
        if feature_engineer is None:
            return {'error': 'Feature engineer no proveído'}
        
        try:
            if hasattr(match_date, 'tzinfo') and match_date.tzinfo:
                match_date = match_date.replace(tzinfo=None)
            
            features = feature_engineer.create_match_features(
                home_team, away_team, match_date
            )
            
            df = pd.DataFrame([features])
            numeric = df[self.feature_cols].fillna(0)
            
            expected = self.model.predict(numeric)[0]
            
            thresholds = [0.5, 1.5, 2.5, 3.5]
            markets = {}
            
            for thresh in thresholds:
                over_prob = self._poisson_over(expected, thresh)
                markets[f'over_{thresh}'] = {
                    'over_prob': over_prob,
                    'under_prob': 1 - over_prob,
                    'prediction': 'OVER' if over_prob > 0.5 else 'UNDER',
                    'confidence': abs(over_prob - 0.5) * 2,
                }
            
            return {
                'home': home_team,
                'away': away_team,
                'expected_goals': expected,
                'markets': markets,
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _poisson_over(self, lamb: float, threshold: float) -> float:
        """Calcular P(Over) usando Poisson"""
        lamb = max(0.1, lamb)
        k = int(threshold)
        
        prob_under = sum(
            exp(-lamb) * (lamb ** i) / factorial(i)
            for i in range(k)
        )
        
        return 1 - prob_under


def get_goals_predictor(models_dir: str = "models") -> GoalsPredictor:
    """Factory function"""
    return GoalsPredictor(models_dir)