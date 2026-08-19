"""
Winner Predictor
===============
Predice resultado 1X2: LOCAL / EMPATE / VISITANTE

Ensamble GB (80%) + RF (20%), portado de MML-Mundial/src_v2/models/winner_predictor.py
(sin LogisticRegression: en Mundial su peso final de ensamble terminó en 0, no aporta).

Dependencias:
- sklearn (RandomForestClassifier, GradientBoostingClassifier)
- pickle

Usa:
- features.feature_engineer
- core.eval_utils (split cronológico, opcional vía `dates`)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from core.eval_utils import chronological_split
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.eval_utils import chronological_split


class WinnerPredictor:
    """Predice resultado de partido con un ensamble GB+RF"""

    ENSEMBLE_WEIGHTS = [0.80, 0.20]  # [GB, RF] -- mismos pesos que MML-Mundial

    def __init__(self, models_dir: str = "models", data_dir: str = "data/cleaned"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models: Dict = {}
        self.feature_cols: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.performance: Dict = {}

    def train(self, features_df: pd.DataFrame, targets: pd.Series,
              dates: Optional[pd.Series] = None) -> bool:
        """Entrenar ensamble GB+RF con regularización para evitar overfitting.

        Si se pasa `dates`, usa split cronológico (core.eval_utils.chronological_split)
        para medir performance honesta, y después reentrena sobre el 100% de los datos
        antes de guardar -- el modelo desplegado no debe descartar para siempre el tramo
        más reciente (mismo fix que core/models/outcome_xgb.py y goals_glm.py).
        """
        print("\n🏋️ ENTRENANDO WINNER PREDICTOR (GB+RF)")
        print("=" * 50)

        self.feature_cols = features_df.columns.tolist()

        if dates is not None:
            X_train, X_test, y_train, y_test, _ = chronological_split(features_df, targets, dates)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, targets, test_size=0.2, random_state=42
            )

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        # Escalar features (RF las usa escaladas, GB no las necesita)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Pesos de clase inversamente proporcionales (LOCAL suele dominar)
        class_counts = y_train.value_counts()
        total = len(y_train)
        class_weight_dict = {
            0: total / (3 * class_counts.get(0, 1)),  # VISITANTE
            1: total / (3 * class_counts.get(1, 1)),  # EMPATE
            2: total / (3 * class_counts.get(2, 1)),  # LOCAL
        }
        print(f"  Class weights: {class_weight_dict}")

        gb = GradientBoostingClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.05,
            min_samples_split=20, min_samples_leaf=10,
            subsample=0.7, random_state=42, max_features='sqrt',
        )
        gb.fit(X_train, y_train)

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1,
            class_weight='balanced',
        )
        rf.fit(X_train_scaled, y_train)

        self.models['gb'] = gb
        self.models['rf'] = rf

        y_train_pred = self._ensemble_predict(X_train, X_train_scaled)
        y_test_pred = self._ensemble_predict(X_test, X_test_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # CV solo sobre GB (es el estimador dominante del ensamble, igual que en Mundial)
        cv = cross_val_score(gb, X_train, y_train, cv=5)

        print(f"  Train Acc: {train_acc:.3f}")
        print(f"  Test Acc: {test_acc:.3f}")
        print(f"  CV Mean: {cv.mean():.3f} ± {cv.std():.3f}")

        self.performance = {
            'train': train_acc,
            'test': test_acc,
            'cv_mean': cv.mean(),
            'cv_std': cv.std(),
        }

        # El split de arriba es solo para medir performance honesta -- el modelo que se
        # guarda se reentrena con todo el dataset disponible (ver docstring).
        if dates is not None:
            features_scaled_full = self.scaler.fit_transform(features_df)
            gb.fit(features_df, targets)
            rf.fit(features_scaled_full, targets)

        self._save()
        return True

    def _ensemble_predict_proba(self, X_raw, X_scaled) -> np.ndarray:
        gb_probs = self.models['gb'].predict_proba(X_raw)
        rf_probs = self.models['rf'].predict_proba(X_scaled)
        w = self.ENSEMBLE_WEIGHTS
        return (gb_probs * w[0] + rf_probs * w[1]) / sum(w)

    def _ensemble_predict(self, X_raw, X_scaled) -> np.ndarray:
        return self._ensemble_predict_proba(X_raw, X_scaled).argmax(axis=1)

    def _save(self):
        """Guardar GB + RF + scaler + columnas de features"""
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # winner_predictor.pkl mantiene el nombre histórico -- src/prediction_models.py
        # (gradient_boosting_v2) lo carga directo como un estimador sklearn plano, sin
        # pasar por esta clase. No renombrar sin actualizar ese consumidor.
        with open(self.models_dir / "winner_predictor.pkl", 'wb') as f:
            pickle.dump(self.models['gb'], f)

        with open(self.models_dir / "winner_rf.pkl", 'wb') as f:
            pickle.dump(self.models['rf'], f)

        with open(self.models_dir / "winner_features.pkl", 'wb') as f:
            pickle.dump(self.feature_cols, f)

        with open(self.models_dir / "winner_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"\n💾 Guardado en {self.models_dir}")

    def load(self) -> bool:
        """Cargar GB (requerido) + RF (opcional, fallback a solo-GB si no existe)"""
        gb_path = self.models_dir / "winner_predictor.pkl"
        if not gb_path.exists():
            return False

        try:
            with open(gb_path, 'rb') as f:
                self.models['gb'] = pickle.load(f)

            rf_path = self.models_dir / "winner_rf.pkl"
            if rf_path.exists():
                with open(rf_path, 'rb') as f:
                    self.models['rf'] = pickle.load(f)

            with open(self.models_dir / "winner_features.pkl", 'rb') as f:
                self.feature_cols = pickle.load(f)

            with open(self.models_dir / "winner_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)

            return True
        except Exception as e:
            print(f"Error cargando: {e}")
            return False

    def predict(self, home_team: str, away_team: str,
                match_date, feature_engineer) -> Dict:
        """Predecir resultado de un partido (ensamble GB+RF si ambos están cargados)"""
        if 'gb' not in self.models:
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

            gb_probs = self.models['gb'].predict_proba(numeric)[0]
            if 'rf' in self.models:
                numeric_scaled = self.scaler.transform(numeric)
                rf_probs = self.models['rf'].predict_proba(numeric_scaled)[0]
                w = self.ENSEMBLE_WEIGHTS
                probs = (gb_probs * w[0] + rf_probs * w[1]) / sum(w)
            else:
                probs = gb_probs

            pred = int(np.argmax(probs))
            result_map = {0: 'VISITANTE', 1: 'EMPATE', 2: 'LOCAL'}

            return {
                'home': home_team,
                'away': away_team,
                'date': match_date,
                'predicted': result_map[pred],
                'code': pred,
                'confidence': float(max(probs)),
                'probabilities': {
                    'VISITANTE': float(probs[0]),
                    'EMPATE': float(probs[1]),
                    'LOCAL': float(probs[2]),
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def get_performance(self) -> Dict:
        """Obtener métricas"""
        return self.performance

    def get_feature_importance(self, feature_names: List[str]) -> List[Tuple]:
        """Obtener importance de features (del modelo GB)"""
        if 'gb' not in self.models:
            return []

        imp = self.models['gb'].feature_importances_
        return sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)


def get_winner_predictor(models_dir: str = "models") -> WinnerPredictor:
    """Factory function"""
    return WinnerPredictor(models_dir)
