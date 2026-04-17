"""
Módulo para predecir Over/Under en múltiples umbrales.
Usa regresión para predecir goles esperados y calcula probabilidades para cada umbral.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
from typing import Dict, List


class GoalsPredictor:
    def __init__(self, models_dir: str = "models", data_dir: str = "data/cleaned"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.feature_engineer = None
        self.model = None
        self.feature_columns = None
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def create_dataset(self, feature_engineer) -> tuple:
        """Crea dataset para predecir número de goles"""
        feature_engineer.load_data()
        self.feature_engineer = feature_engineer
        
        all_matches = pd.concat([
            feature_engineer.matches_2023,
            feature_engineer.matches_2024,
            feature_engineer.matches_2025
        ]).copy()
        
        all_matches = all_matches[all_matches['status'] == 'FINISHED']
        
        print(f"  Total partidos: {len(all_matches)}")
        
        features_list = []
        targets_list = []
        
        for _, match in all_matches.iterrows():
            try:
                features = feature_engineer.create_match_features(
                    match['home_team'],
                    match['away_team'],
                    match['date']
                )
                
                total_goals = match['home_score'] + match['away_score']
                features_list.append(features)
                targets_list.append(total_goals)
                
            except Exception:
                continue
        
        if not features_list:
            return None, None
        
        features_df = pd.DataFrame(features_list)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        features_df = features_df[numeric_cols]
        self.feature_columns = numeric_cols
        
        print(f"  Features: {len(numeric_cols)}")
        print(f"  Goles promedio: {np.mean(targets_list):.2f}")
        
        return features_df, pd.Series(targets_list)
    
    def train(self, features_df: pd.DataFrame, targets: pd.Series):
        """Entrenar modelo de regresión para goles"""
        print("\n🏆 ENTRENANDO MODELO DE GOLES")
        print("="*50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets, test_size=0.2, random_state=42
        )
        
        print(f"  Entrenamiento: {len(X_train)}")
        print(f"  Prueba: {len(X_test)}")
        
        # Random Forest Regressor
        print("\n  Entrenando RandomForestRegressor...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"    Train MAE: {train_mae:.2f}, R2: {train_r2:.3f}")
        print(f"    Test MAE: {test_mae:.2f}, R2: {test_r2:.3f}")
        
        self._save_model()
        
        return True
    
    def _save_model(self):
        """Guardar modelo"""
        if self.model:
            with open(f"{self.models_dir}/goals_predictor.pkl", 'wb') as f:
                pickle.dump(self.model, f)
        if self.feature_columns:
            with open(f"{self.models_dir}/goals_columns.pkl", 'wb') as f:
                pickle.dump(self.feature_columns, f)
        print(f"\n💾 Modelo guardado")
    
    def load_model(self) -> bool:
        """Cargar modelo desde disco"""
        model_path = f"{self.models_dir}/goals_predictor.pkl"
        cols_path = f"{self.models_dir}/goals_columns.pkl"
        
        if not os.path.exists(model_path):
            return False
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            if os.path.exists(cols_path):
                with open(cols_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def predict_goals(self, home_team: str, away_team: str, match_date) -> Dict:
        """Predecir número de goles y calcular Over/Under para cada umbral"""
        if not self.model:
            return {'error': 'Modelo no cargado'}
        
        if self.feature_engineer is None:
            return {'error': 'Feature engineer no inicializado'}
        
        try:
            if hasattr(match_date, 'tzinfo') and match_date.tzinfo is not None:
                match_date = match_date.replace(tzinfo=None)
            
            features = self.feature_engineer.create_match_features(
                home_team, away_team, match_date
            )
            
            features_df = pd.DataFrame([features])
            numeric_features = features_df[self.feature_columns].fillna(0)
            
            expected_goals = self.model.predict(numeric_features)[0]
            
            # Calcular Over/Under para cada umbral usando distribución de Poisson
            thresholds = [0.5, 1.5, 2.5, 3.5]
            results = {
                'home_team': home_team,
                'away_team': away_team,
                'expected_goals': expected_goals,
                'markets': {}
            }
            
            for threshold in thresholds:
                over_prob = self._calculate_over_prob(expected_goals, threshold)
                under_prob = 1 - over_prob
                
                confidence = max(over_prob, under_prob)
                prediction = "OVER" if over_prob > under_prob else "UNDER"
                
                results['markets'][f'over_{threshold}'] = {
                    'over_prob': over_prob,
                    'under_prob': under_prob,
                    'prediction': prediction,
                    'confidence': confidence,
                    'threshold': threshold
                }
            
            return results
            
        except Exception as e:
            return {'error': f'Error: {str(e)}'}
    
    def _calculate_over_prob(self, expected_goals: float, threshold: float) -> float:
        """
        Calcula probabilidad de Over usando aproximación de Poisson.
        P(X >= threshold) donde X ~ Poisson(lambda)
        """
        from math import exp, factorial
        
        lamb = max(0.1, expected_goals)
        
        # P(Over) = 1 - P(0) - P(1) - ... - P(threshold-1)
        k = int(threshold)
        over_prob = 0.0
        
        for i in range(k):
            poi = exp(-lamb) * (lamb ** i) / factorial(i)
            over_prob += poi
        
        return 1 - over_prob


def get_goals_predictor(models_dir: str = "models", data_dir: str = "data/cleaned") -> GoalsPredictor:
    """Factory function"""
    return GoalsPredictor(models_dir, data_dir)
