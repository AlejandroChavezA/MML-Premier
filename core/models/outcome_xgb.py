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
from sklearn.calibration import CalibratedClassifierCV
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

    def train(self, features_df: pd.DataFrame, targets: pd.Series, dates: Optional[pd.Series] = None,
              xgb_params: Optional[Dict] = None, sample_weight: Optional[pd.Series] = None) -> bool:
        """`sample_weight` (opcional, requiere `dates`): peso por partido -- p.ej. decay por
        recencia -- alineado por posición con `features_df`/`targets`. Usado por
        experiments/premier_outcome.py; el entrenamiento en vivo (core/train_ensemble.py) no
        lo pasa, así que el comportamiento de producción no cambia.
        """
        self.feature_cols = features_df.columns.tolist()

        sw_train = None
        if dates is not None:
            # Split cronológico: nunca deja un partido futuro en train con uno
            # pasado en test (ver core/eval_utils.py).
            X_train, X_test, y_train, y_test, cutoff_date = chronological_split(features_df, targets, dates)
            if sample_weight is not None:
                dates_r = pd.Series(dates).reset_index(drop=True)
                sw_all = pd.Series(sample_weight).reset_index(drop=True)
                sw_train = sw_all[(dates_r < cutoff_date).values].reset_index(drop=True)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, targets, test_size=0.2, random_state=42, stratify=targets
            )

        # Regularización fuerte por default: con ~1-2k partidos y 30+ features
        # (Premier), configuraciones menos conservadoras (n_estimators>=100,
        # max_depth>=4) sobreajustan mucho (train~86% / test~47% en el backtest
        # de Premier) -- ver docs/plan_5_ligas_ligamx.md Paso 3 /
        # core/evaluate_ensemble.py. Ligas con más historia (ej. Liga MX,
        # ~5k partidos) necesitan menos regularización -- con esta misma config
        # colapsaban a predecir "LOCAL" siempre (ver core/league_config.py
        # OutcomeXGBConfig). Por eso xgb_params viene de LeagueConfig, no
        # hardcodeado acá.
        default_params = dict(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.6,
            colsample_bytree=0.6,
            min_child_weight=20,
            reg_alpha=2.0,
            reg_lambda=4.0,
        )
        params = {**default_params, **(xgb_params or {})}

        def _make_xgb():
            return XGBClassifier(
                **params,
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                random_state=42,
            )

        diagnostic_model = _make_xgb()
        diagnostic_model.fit(X_train, y_train, sample_weight=sw_train)

        train_acc = accuracy_score(y_train, diagnostic_model.predict(X_train))
        test_acc = accuracy_score(y_test, diagnostic_model.predict(X_test))
        cv = cross_val_score(diagnostic_model, X_train, y_train, cv=5)

        self.performance = {
            'train': float(train_acc),
            'test': float(test_acc),
            'cv_mean': float(cv.mean()),
            'cv_std': float(cv.std()),
        }

        # Calibración post-hoc (isotónica, 5-fold): sin esto, las probabilidades
        # de XGBoost quedaban peor calibradas que un prior constante (log-loss
        # 1.20 vs 1.14 en el holdout de Liga MX -- ver evaluate_ensemble.py). La
        # regularización fuerte que evita el overfitting de accuracy no evita que
        # el modelo esté sobre/sub-confiado en cada clase; CalibratedClassifierCV
        # reajusta la forma de las probabilidades sin tocar accuracy en seco.
        # El split de arriba (diagnostic_model) es solo para medir performance
        # honesta -- el modelo final se entrena con TODO el dataset disponible
        # (incluida la parte más reciente que quedó afuera del split), para no
        # predecir la temporada actual con un modelo que nunca vio nada de los
        # últimos años. self.performance queda igual (medido antes de esto)
        # porque es lo único que se puede evaluar honestamente.
        fit_features, fit_targets = (features_df, targets) if dates is not None else (X_train, y_train)
        fit_sample_weight = sample_weight if dates is not None else sw_train
        self.model = CalibratedClassifierCV(_make_xgb(), method='isotonic', cv=5)
        self.model.fit(fit_features, fit_targets, sample_weight=fit_sample_weight)

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
                'raw_features': features,
            }
        except Exception as e:
            return {'error': str(e)}

    def get_performance(self) -> Dict:
        return self.performance
