"""Modelo de resultado (1X2) del ensamble -- XGBoost.

Interfaz paralela a src_v2/models/winner_predictor.py's WinnerPredictor
(train/predict/save/load) pero usando xgboost.XGBClassifier en vez de
GradientBoostingClassifier. Consume el mismo FeatureEngineer.create_match_features()
que WinnerPredictor -- no reinventa el feature engineering (ver
docs/plan_5_ligas_ligamx.md, Paso 3).
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from core.eval_utils import chronological_split

# 0=VISITANTE, 1=EMPATE, 2=LOCAL (mismo mapeo que WinnerPredictor/FeatureEngineer.create_training_dataset)
RESULT_MAP = {0: 'VISITANTE', 1: 'EMPATE', 2: 'LOCAL'}


class OutcomeXGB:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.model: Optional[XGBClassifier] = None
        self.feature_cols: List[str] = []
        self.performance: Dict = {}

    def train(self, features_df: pd.DataFrame, targets: pd.Series, dates: Optional[pd.Series] = None) -> bool:
        self.feature_cols = features_df.columns.tolist()

        if dates is not None:
            # Split cronológico: nunca deja un partido futuro en train con uno
            # pasado en test (ver core/eval_utils.py).
            X_train, X_test, y_train, y_test, _ = chronological_split(features_df, targets, dates)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, targets, test_size=0.2, random_state=42, stratify=targets
            )

        # Regularización fuerte: con ~1-2k partidos y 30+ features, configuraciones
        # menos conservadoras (n_estimators>=100, max_depth>=4) sobreajustan mucho
        # (train~86% / test~47% en el backtest de Premier) -- ver
        # docs/plan_5_ligas_ligamx.md Paso 3 / core/evaluate_ensemble.py.
        self.model = XGBClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.6,
            colsample_bytree=0.6,
            min_child_weight=20,
            reg_alpha=2.0,
            reg_lambda=4.0,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        test_acc = accuracy_score(y_test, self.model.predict(X_test))
        cv = cross_val_score(self.model, X_train, y_train, cv=5)

        self.performance = {
            'train': float(train_acc),
            'test': float(test_acc),
            'cv_mean': float(cv.mean()),
            'cv_std': float(cv.std()),
        }

        # El split de arriba es solo para medir performance honesta -- el modelo
        # que se guarda y se usa en producción se reentrena con TODO el dataset
        # disponible (incluida la parte más reciente que quedó afuera del split),
        # para no predecir la temporada actual con un modelo que nunca vio nada
        # de los últimos años. self.performance queda igual (medido antes de este
        # refit) porque es lo único que se puede evaluar honestamente.
        if dates is not None:
            self.model.fit(features_df, targets)

        self._save()
        return True

    def _save(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.models_dir / "outcome_xgb.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.models_dir / "outcome_xgb_features.pkl", 'wb') as f:
            pickle.dump(self.feature_cols, f)

    def load(self) -> bool:
        model_path = self.models_dir / "outcome_xgb.pkl"
        if not model_path.exists():
            return False
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.models_dir / "outcome_xgb_features.pkl", 'rb') as f:
                self.feature_cols = pickle.load(f)
            return True
        except Exception:
            return False

    def predict_proba(self, home_team: str, away_team: str, match_date, feature_engineer) -> Dict:
        """Probabilidades por resultado para un partido. No decide -- eso lo hace core.ensemble."""
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

            probs = self.model.predict_proba(numeric)[0]
            proba_by_result = {RESULT_MAP[i]: float(p) for i, p in enumerate(probs)}
            best_code = int(probs.argmax())

            return {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_result': RESULT_MAP[best_code],
                'confidence': float(probs[best_code]),
                'probabilities': proba_by_result,
            }
        except Exception as e:
            return {'error': str(e)}

    def get_performance(self) -> Dict:
        return self.performance
