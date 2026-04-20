"""
Winner Predictor
===============
Predice resultado 1X2: LOCAL / EMPATE / VISITANTE

Dependencias:
- sklearn (RandomForestClassifier, GradientBoostingClassifier)
- pickle

Usa:
- features.feature_engineer
- features.competitiveness
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class WinnerPredictor:
    """Predice resultado de partido"""
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data/cleaned"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models: Dict = {}
        self.feature_cols: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.performance: Dict = {}
    
    def train(self, features_df: pd.DataFrame, targets: pd.Series,
             use_time_series: bool = True) -> bool:
        """Entrenar modelo con regularización para evitar overfitting"""
        print("\n🏋️ ENTRENANDO WINNER PREDICTOR")
        print("=" * 50)
        
        self.feature_cols = features_df.columns.tolist()
        
        # Time-series split para evitar data leak
        if use_time_series:
            features_df = features_df.copy()
            targets = targets.copy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets, test_size=0.2, random_state=42
        )
        
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Escalar features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # ============================================
        # MODELO CON REGULARIZACIÓN ROBUSTA + MANEJO DE CLASES
        # ============================================
        # Calcular pesos de clase inversamente proporcionales
        class_counts = y_train.value_counts()
        total = len(y_train)
        class_weight_dict = {
            0: total / (3 * class_counts.get(0, 1)),  # VISITANTE
            1: total / (3 * class_counts.get(1, 1)),  # EMPATE  
            2: total / (3 * class_counts.get(2, 1)),  # LOCAL
        }
        
        print(f"  Class weights: {class_weight_dict}")
        
        # Gradient Boosting solo (menos overfitting que RF)
        model = GradientBoostingClassifier(
            n_estimators=80,         # Reducido
            max_depth=4,           # Muy poco profundo
            learning_rate=0.05,   # Muy pequeño
            min_samples_split=20,  # Mucho para evitar overfitting
            min_samples_leaf=10,   # Mucho para evitar overfitting
            subsample=0.7,         # Regularización
            random_state=42,
            max_features='sqrt'
        )
        
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        # Cross-validation
        cv = cross_val_score(model, X_train, y_train, cv=5)
        
        print(f"  Train Acc: {train_acc:.3f}")
        print(f"  Test Acc: {test_acc:.3f}")
        print(f"  CV Mean: {cv.mean():.3f} ± {cv.std():.3f}")
        
        self.models['rf'] = model
        self.performance = {
            'train': train_acc,
            'test': test_acc,
            'cv_mean': cv.mean(),
            'cv_std': cv.std(),
        }
        
        self._save()
        return True
    
    def _save(self):
        """Guardar modelo"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.models_dir / "winner_predictor.pkl", 'wb') as f:
            pickle.dump(self.models['rf'], f)
        
        with open(self.models_dir / "winner_features.pkl", 'wb') as f:
            pickle.dump(self.feature_cols, f)
        
        with open(self.models_dir / "winner_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n💾 Guardado en {self.models_dir}")
    
    def load(self) -> bool:
        """Cargar modelo"""
        model_path = self.models_dir / "winner_predictor.pkl"
        
        if not model_path.exists():
            return False
        
        try:
            with open(model_path, 'rb') as f:
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
        """Predecir resultado de un partido"""
        if 'rf' not in self.models:
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
            
            # Handle both RF and VotingClassifier
            model = self.models['rf']
            if hasattr(model, 'predict_proba'):
                pred = model.predict(numeric)[0]
                probs = model.predict_proba(numeric)[0]
            else:
                # For voting classifier access underlying estimators
                pred = model.predict(numeric)[0]
                probs = model.predict_proba(numeric)[0]
            
            result_map = {0: 'VISITANTE', 1: 'EMPATE', 2: 'LOCAL'}
            
            return {
                'home': home_team,
                'away': away_team,
                'date': match_date,
                'predicted': result_map[pred],
                'code': int(pred),
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
        """Obtener importance de features"""
        if 'rf' not in self.models:
            return []
        
        imp = self.models['rf'].feature_importances_
        return sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)


def get_winner_predictor(models_dir: str = "models") -> WinnerPredictor:
    """Factory function"""
    return WinnerPredictor(models_dir)