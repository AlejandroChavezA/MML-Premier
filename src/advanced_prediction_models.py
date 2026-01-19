import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class AdvancedMatchPredictor:
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
            print(f"Creado directorio: {models_dir}")
    
    def train_advanced_models(self, features_df: pd.DataFrame, targets_df: pd.Series):
        """Entrenar modelos avanzados con optimización de hiperparámetros"""
        print("ENTRENANDO MODELOS AVANZADOS...")
        print("=" * 60)
        
        # Guardar columnas de características
        self.feature_columns = features_df.columns.tolist()
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets_df, test_size=0.2, random_state=42, stratify=targets_df
        )
        
        print(f"  Datos de entrenamiento: {len(X_train)} partidos")
        print(f"  Datos de prueba: {len(X_test)} partidos")
        print(f"  Características: {len(X_train.columns)}")
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Configurar modelos avanzados con hiperparámetros optimizados
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,  # Más árboles
                max_depth=15,      # Más profundo
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1  # Usar todos los cores
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,     # Más estimadores
                max_depth=8,          # Más profundo
                learning_rate=0.05,    # Más conservador
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                subsample=0.8          # Regularización
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=2000,         # Más iteraciones
                solver='lbfgs',         # Compatible con multiclass
                C=1.0                  # Regularización
            ),
            'xgboost': None  # Placeholder para XGBoost si está disponible
        }
        
        # Intentar usar XGBoost si está disponible
        try:
            import xgboost as xgb
            models_config['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            print("  XGBoost disponible y añadido")
        except ImportError:
            print("  XGBoost no disponible (opcional)")
        
        # Entrenar cada modelo
        for model_name, model in models_config.items():
            if model is None:
                continue
                
            print(f"\nENTRENANDO {model_name.upper()} (AVANZADO)...")
            
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
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Feature importance (solo para modelos de árboles)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(X_train.columns, model.feature_importances_))
                # Ordenar por importancia
                feature_importance = sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True)[:15]
                
                # Mostrar features más importantes
                print(f"  Top 5 features más importantes:")
                for feature, importance in feature_importance[:5]:
                    print(f"    {feature}: {importance:.4f}")
            
            # Guardar modelo y rendimiento
            self.models[model_name] = model
            self.model_performance[model_name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, test_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, test_pred).tolist(),
                'feature_importance': feature_importance
            }
            
            print(f"  Accuracy entrenamiento: {train_accuracy:.3f}")
            print(f"  Accuracy prueba: {test_accuracy:.3f}")
            print(f"  Validación cruzada: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Guardar modelo en disco
            model_path = f"{self.models_dir}/{model_name}_advanced.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Modelo guardado: {model_path}")
        
        # Guardar scaler y columnas
        with open(f"{self.models_dir}/scaler_advanced.pkl", 'wb') as f:
            pickle.dump(self.scalers['standard'], f)
        
        with open(f"{self.models_dir}/feature_columns_advanced.pkl", 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        # Guardar rendimiento
        with open(f"{self.models_dir}/performance_advanced.pkl", 'wb') as f:
            pickle.dump(self.model_performance, f)
        
        print(f"\nENTRENAMIENTO AVANZADO COMPLETADO!")
        self._print_model_comparison()
        
        return True
    
    def load_advanced_models(self):
        """Cargar modelos avanzados desde disco"""
        try:
            # Cargar modelos avanzados
            model_files = {
                'random_forest': 'random_forest_advanced.pkl',
                'gradient_boosting': 'gradient_boosting_advanced.pkl',
                'logistic_regression': 'logistic_regression_advanced.pkl',
                'xgboost': 'xgboost_advanced.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = f"{self.models_dir}/{filename}"
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"Cargado: {model_name}")
            
            # Cargar scaler
            scaler_path = f"{self.models_dir}/scaler_advanced.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers['standard'] = pickle.load(f)
            
            # Cargar columnas
            columns_path = f"{self.models_dir}/feature_columns_advanced.pkl"
            if os.path.exists(columns_path):
                with open(columns_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
            
            # Cargar rendimiento
            performance_path = f"{self.models_dir}/performance_advanced.pkl"
            if os.path.exists(performance_path):
                with open(performance_path, 'rb') as f:
                    self.model_performance = pickle.load(f)
            
            print(f"Modelos avanzados cargados: {list(self.models.keys())}")
            return True
            
        except Exception as e:
            print(f"Error cargando modelos avanzados: {e}")
            return False
    
    def predict_match_advanced(self, home_team: str, away_team: str, match_date: datetime, 
                             model_name: str = 'random_forest') -> Dict:
        """Predecir resultado con características avanzadas"""
        if model_name not in self.models:
            return {'error': f'Modelo {model_name} no disponible'}
        
        if self.feature_engineer is None:
            return {'error': 'Feature engineer no inicializado'}
        
        try:
            # Crear características AVANZADAS
            features = self.feature_engineer.create_advanced_match_features(
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
            
            # Obtener importancia de características
            feature_importance = None
            if model_name in self.model_performance and 'feature_importance' in self.model_performance[model_name]:
                feature_importance = self.model_performance[model_name]['feature_importance']
            
            # Extraer características clave para mostrar
            key_features = {
                'points_diff': features.get('points_diff', 0),
                'gd_diff': features.get('gd_diff', 0),
                'position_diff': features.get('position_diff', 0),
                'form_diff_last5_points': features.get('form_diff_last5_points', 0),
                'home_advantage_rate': features.get('home_advantage_rate', 0),
                'rest_diff': features.get('rest_diff', 0)
            }
            
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
                'model_used': f"{model_name}_advanced",
                'feature_importance': feature_importance,
                'key_features': key_features
            }
            
        except Exception as e:
            return {'error': f'Error en predicción avanzada: {str(e)}'}
    
    def get_advanced_model_performance(self) -> Dict:
        """Obtener rendimiento de modelos avanzados"""
        return self.model_performance
    
    def get_best_advanced_model(self) -> str:
        """Obtener el mejor modelo avanzado basado en accuracy de prueba"""
        if not self.model_performance:
            return 'random_forest'  # Default
        
        best_model = max(self.model_performance.items(), 
                        key=lambda x: x[1]['test_accuracy'])
        return best_model[0]
    
    def _print_model_comparison(self):
        """Imprimir comparación de modelos avanzados"""
        print("\nCOMPARACIÓN DE MODELOS AVANZADOS:")
        print("=" * 70)
        print(f"{'Modelo':<25} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12}")
        print("-" * 70)
        
        for model_name, performance in self.model_performance.items():
            print(f"{model_name:<25} "
                  f"{performance['train_accuracy']:<12.3f} "
                  f"{performance['test_accuracy']:<12.3f} "
                  f"{performance['cv_mean']:<12.3f}")
        
        # Mejor modelo
        best_model = self.get_best_advanced_model()
        best_acc = self.model_performance[best_model]['test_accuracy']
        print(f"\nMEJOR MODELO AVANZADO: {best_model}")
        print(f"Accuracy: {best_acc:.3f}")
        
        # Mejora esperada
        if len(self.model_performance) > 0:
            baseline_acc = max(p['test_accuracy'] for p in self.model_performance.values())
            print(f"\nESPERADO: +{baseline_acc*100 - 53.6:.1f}% sobre baseline original")

if __name__ == "__main__":
    # Prueba del predictor avanzado
    from advanced_feature_engineering import AdvancedFeatureEngineer
    
    predictor = AdvancedMatchPredictor()
    fe = AdvancedFeatureEngineer()
    
    if fe.load_data():
        print("Creando dataset AVANZADO de entrenamiento...")
        features_df, targets_df = fe.create_advanced_training_dataset()
        
        print(f"Dataset AVANZADO: {features_df.shape[0]} partidos, {features_df.shape[1]} características")
        
        # Entrenar modelos avanzados
        predictor.train_advanced_models(features_df, targets_df)
        
        # Probar predicción
        predictor.feature_engineer = fe
        test_prediction = predictor.predict_match_advanced(
            "Arsenal FC",
            "Manchester City FC",
            datetime(2025, 1, 15),
            'random_forest'
        )
        
        print("\nPREDICCIÓN AVANZADA de prueba:")
        for key, value in test_prediction.items():
            if key != 'feature_importance':
                print(f"  {key}: {value}")