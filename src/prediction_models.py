import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class MatchPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.feature_engineer = None
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.model_performance = {}
        
        # Crear directorio de modelos
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"📁 Creado directorio: {models_dir}")
    
    def train_models(self, features_df: pd.DataFrame, targets_df: pd.Series):
        """Entrenar múltiples modelos de predicción"""
        print(" Entrenando modelos de predicción...")
        
        # Guardar columnas de características
        self.feature_columns = features_df.columns.tolist()
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets_df, test_size=0.2, random_state=42, stratify=targets_df
        )
        
        print(f"  Datos de entrenamiento: {len(X_train)} partidos")
        print(f"  Datos de prueba: {len(X_test)} partidos")
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Definir modelos
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        # Entrenar cada modelo
        for model_name, model in models_config.items():
            print(f"\n📊 Entrenando {model_name}...")
            
            if model_name == 'logistic_regression':
                # Para regresión logística usar datos escalados
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
            else:
                # Para modelos basados en árboles usar datos originales
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
            
            # Calcular rendimiento
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Validación cruzada
            if model_name == 'logistic_regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Guardar modelo y rendimiento
            self.models[model_name] = model
            self.model_performance[model_name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, test_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
            }
            
            print(f"   Accuracy entrenamiento: {train_accuracy:.3f}")
            print(f"  Accuracy prueba: {test_accuracy:.3f}")
            print(f"  Validación cruzada: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Guardar modelo en disco
            model_path = f"{self.models_dir}/{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  💾 Modelo guardado: {model_path}")
        
        # Guardar scaler y columnas
        with open(f"{self.models_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scalers['standard'], f)
        
        with open(f"{self.models_dir}/feature_columns.pkl", 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        # Guardar rendimiento
        with open(f"{self.models_dir}/performance.pkl", 'wb') as f:
            pickle.dump(self.model_performance, f)
        
        print(f"\n🎉 ¡Entrenamiento completado!")
        self._print_model_comparison()
        
        return True
    
    def load_models(self):
        """Cargar modelos entrenados desde disco"""
        try:
            # Cargar modelos
            model_files = {
                'random_forest': 'random_forest.pkl',
                'gradient_boosting': 'gradient_boosting.pkl',
                'logistic_regression': 'logistic_regression.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = f"{self.models_dir}/{filename}"
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                else:
                    print(f" No se encontró modelo: {model_path}")
            
            # Cargar scaler
            scaler_path = f"{self.models_dir}/scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers['standard'] = pickle.load(f)
            
            # Cargar columnas
            columns_path = f"{self.models_dir}/feature_columns.pkl"
            if os.path.exists(columns_path):
                with open(columns_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
            
            # Cargar rendimiento
            performance_path = f"{self.models_dir}/performance.pkl"
            if os.path.exists(performance_path):
                with open(performance_path, 'rb') as f:
                    self.model_performance = pickle.load(f)
            
            print(f" Modelos cargados: {list(self.models.keys())}")
            return True
            
        except Exception as e:
            print(f" Error cargando modelos: {e}")
            return False
    
    def predict_match(self, home_team: str, away_team: str, match_date: datetime, 
                     model_name: str = 'random_forest') -> Dict:
        """Predecir resultado de un partido"""
        if model_name not in self.models:
            return {'error': f'Modelo {model_name} no disponible'}
        
        if self.feature_engineer is None:
            return {'error': 'Feature engineer no inicializado'}
        
        try:
            # Crear características
            features = self.feature_engineer.create_match_features(
                home_team, away_team, match_date
            )
            
            # Convertir a DataFrame
            features_df = pd.DataFrame([features])
            
            # Seleccionar solo columnas numéricas usadas en entrenamiento
            numeric_features = features_df[self.feature_columns].fillna(0)
            
            # Hacer predicción
            model = self.models[model_name]
            
            if model_name == 'logistic_regression':
                # Escalar características para regresión logística
                scaled_features = self.scalers['standard'].transform(numeric_features)
                prediction = model.predict(scaled_features)[0]
                probabilities = model.predict_proba(scaled_features)[0]
            else:
                # Usar características originales para modelos de árboles
                prediction = model.predict(numeric_features)[0]
                probabilities = model.predict_proba(numeric_features)[0]
            
            # Mapear resultados
            result_map = {0: 'VISITANTE', 1: 'EMPATE', 2: 'LOCAL'}
            predicted_result = result_map[prediction]
            
            # Calcular confianza
            confidence = max(probabilities)
            
            # Obtener importancia de características (solo para modelos de árboles)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
                # Ordenar por importancia
                feature_importance = sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'match_date': match_date,
                'predicted_result': predicted_result,
                'prediction_code': prediction,
                'confidence': confidence,
                'probabilities': {
                    'VISITANTE': probabilities[0],
                    'EMPATE': probabilities[1],
                    'LOCAL': probabilities[2]
                },
                'model_used': model_name,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            return {'error': f'Error en predicción: {str(e)}'}
    
    def predict_week_matches(self, matchday: int, season: int = 2025, 
                           model_name: str = 'random_forest') -> List[Dict]:
        """Predecir todos los partidos de una jornada"""
        if self.feature_engineer is None:
            return [{'error': 'Feature engineer no inicializado'}]
        
        try:
            # Cargar partidos de la jornada
            matches_path = f"{self.feature_engineer.data_dir}/matches_{season}_cleaned.csv"
            matches_df = pd.read_csv(matches_path)
            matches_df['date'] = pd.to_datetime(matches_df['date'])
            
            week_matches = matches_df[matches_df['matchday'] == matchday]
            
            if len(week_matches) == 0:
                return [{'error': f'No se encontraron partidos para la jornada {matchday}'}]
            
            predictions = []
            
            for _, match in week_matches.iterrows():
                prediction = self.predict_match(
                    match['home_team'],
                    match['away_team'],
                    match['date'],
                    model_name
                )
                
                if 'error' not in prediction:
                    # Añadir información real del partido si está disponible
                    if match['status'] == 'FINISHED':
                        if match['home_score'] > match['away_score']:
                            actual_result = 'LOCAL'
                        elif match['home_score'] < match['away_score']:
                            actual_result = 'VISITANTE'
                        else:
                            actual_result = 'EMPATE'
                        
                        prediction['actual_result'] = actual_result
                        prediction['home_score'] = match['home_score']
                        prediction['away_score'] = match['away_score']
                        prediction['correct'] = prediction['predicted_result'] == actual_result
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            return [{'error': f'Error prediciendo jornada: {str(e)}'}]
    
    def get_model_performance(self) -> Dict:
        """Obtener rendimiento de todos los modelos"""
        return self.model_performance
    
    def get_best_model(self) -> str:
        """Obtener el mejor modelo basado en accuracy de prueba"""
        if not self.model_performance:
            return 'random_forest'  # Default
        
        best_model = max(self.model_performance.items(), 
                        key=lambda x: x[1]['test_accuracy'])
        return best_model[0]
    
    def _print_model_comparison(self):
        """Imprimir comparación de modelos"""
        print("\n COMPARACIÓN DE MODELOS:")
        print("=" * 60)
        print(f"{'Modelo':<20} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12}")
        print("-" * 60)
        
        for model_name, performance in self.model_performance.items():
            print(f"{model_name:<20} "
                  f"{performance['train_accuracy']:<12.3f} "
                  f"{performance['test_accuracy']:<12.3f} "
                  f"{performance['cv_mean']:<12.3f}")
        
        # Mejor modelo
        best_model = self.get_best_model()
        best_acc = self.model_performance[best_model]['test_accuracy']
        print(f"\n🏆 Mejor modelo: {best_model} (Accuracy: {best_acc:.3f})")

if __name__ == "__main__":
    # Prueba del predictor
    from feature_engineering import FeatureEngineer
    
    predictor = MatchPredictor()
    fe = FeatureEngineer()
    
    if fe.load_data():
        print(" Creando dataset de entrenamiento...")
        features_df, targets_df = fe.create_training_dataset()
        
        print(f"📊 Dataset: {features_df.shape[0]} partidos, {features_df.shape[1]} características")
        
        # Entrenar modelos
        predictor.train_models(features_df, targets_df)
        
        # Probar predicción
        predictor.feature_engineer = fe
        test_prediction = predictor.predict_match(
            "Arsenal FC",
            "Manchester City FC",
            datetime(2025, 1, 15)
        )
        
        print("\n Predicción de prueba:")
        for key, value in test_prediction.items():
            if key != 'feature_importance':
                print(f"  {key}: {value}")