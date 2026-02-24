#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from typing import List, Dict
import pandas as pd
from pathlib import Path

class JornadaPredictor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.feature_engineer = None
        self.predictor = None
        self.teams_df = None
        self.current_model = 'random_forest'
        
    def initialize(self):
        """Initialize system components"""
        print(" Initializing jornada predictor...")
        
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from feature_engineering import FeatureEngineer
            from prediction_models import MatchPredictor
            
            data_dir = self.project_root / "data" / "cleaned"
            models_dir = self.project_root / "models"
            
            self.feature_engineer = FeatureEngineer(data_dir=str(data_dir))
            self.predictor = MatchPredictor(models_dir=str(models_dir))
            
            # Load data
            if not self.feature_engineer.load_data():
                return False
            
            self.teams_df = pd.read_csv(data_dir / "teams_cleaned.csv")
            
            # Load models
            if not self.predictor.load_models():
                print(" No trained models found. Training new models...")
                features_df, targets_df = self.feature_engineer.create_training_dataset()
                self.predictor.train_models(features_df, targets_df)
            
            self.predictor.feature_engineer = self.feature_engineer
            return True
            
        except Exception as e:
            print(f" Error initializing: {e}")
            return False
    
    def get_jornada_predictions(self, matchday: int, season: int = 2025):
        """Get predictions for a specific jornada"""
        try:
            predictions = self.predictor.predict_week_matches(matchday, season, self.current_model)
            return predictions
        except Exception as e:
            print(f" Error getting predictions: {e}")
            return []
    
    def get_feature_explanation(self, prediction: Dict, top_n: int = 4):
        """Get detailed feature explanation for a prediction"""
        try:
            # Get feature importance from prediction or model
            feature_importance = prediction.get('feature_importance', [])
            
            if not feature_importance:
                # Try to get from model directly
                model = self.predictor.models[self.current_model]
                if hasattr(model, 'feature_importances_'):
                    feature_names = self.feature_engineer.get_feature_names()
                    importances = model.feature_importances_
                    feature_importance = list(zip(feature_names, importances))
                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Separate positive and negative features
            positive_features = []
            negative_features = []
            
            for feature, importance in feature_importance[:top_n*2]:
                if importance > 0:
                    positive_features.append((feature, importance))
                else:
                    negative_features.append((feature, importance))
            
            return {
                'positive': positive_features[:top_n],
                'negative': negative_features[:top_n]
            }
            
        except Exception as e:
            print(f" Error getting feature explanation: {e}")
            return {'positive': [], 'negative': []}
    
    def format_feature_name(self, feature: str) -> str:
        """Format feature name for display"""
        # Common feature mappings
        mappings = {
            'home_form_5': 'Local forma últimos 5 partidos',
            'away_form_5': 'Visitante forma últimos 5 partidos',
            'home_win_rate_5': 'Local victorias últimos 5 partidos',
            'away_win_rate_5': 'Visitante victorias últimos 5 partidos',
            'goal_diff_avg': 'Diferencia de goles promedio',
            'home_advantage': 'Ventaja de local',
            'head_to_head_home_win_rate': 'Histórico victorias local',
            'points_per_game': 'Puntos por partido',
            'goals_scored_per_game': 'Goles anotados por partido',
            'goals_conceded_per_game': 'Goles recibidos por partido'
        }
        
        return mappings.get(feature, feature.replace('_', ' ').title())
    
    def display_jornada_detailed(self, matchday: int, season: int = 2025):
        """Display jornada predictions in detailed format"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f" JORNADA {matchday} - PREDICCIONES DETALLADAS")
        print("=" * 80)
        
        predictions = self.get_jornada_predictions(matchday, season)
        
        if not predictions or 'error' in predictions[0]:
            print(f" Error getting predictions for jornada {matchday}")
            input(" Press Enter to continue...")
            return
        
        # Display each match in detailed format
        for i, prediction in enumerate(predictions, 1):
            if 'error' in prediction:
                continue
            
            # Match header
            home_team = prediction['home_team'][:25]
            away_team = prediction['away_team'][:25]
            
            print("\\n" + "─" * 80)
            print(f"⚽ PARTIDO {i}: {home_team} vs {away_team}")
            
            # Match date
            match_date = prediction.get('match_date', 'Fecha no disponible')
            if isinstance(match_date, str):
                try:
                    date_obj = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%A %d de %B - %H:%M')
                except:
                    formatted_date = match_date
            else:
                formatted_date = match_date.strftime('%A %d de %B - %H:%M') if match_date else 'Fecha no disponible'
            
            print(f"📅 {formatted_date}")
            print("─" * 80)
            
            # Prediction result
            result = prediction['predicted_result']
            confidence = prediction['confidence']
            probabilities = prediction['probabilities']
            
            # Main prediction
            if result == 'LOCAL':
                prediction_text = f"🏠 GANA {home_team.upper()}"
            elif result == 'VISITANTE':
                prediction_text = f"✈️ GANA {away_team.upper()}"
            else:
                prediction_text = f"🤝 EMPATE"
            
            print(f"\\n🎯 EL MODELO DICE: {prediction_text}")
            print(f"   Confianza: {confidence:.1%}")
            
            # Probabilities
            home_prob = probabilities.get('LOCAL', 0)
            away_prob = probabilities.get('VISITANTE', 0)
            draw_prob = probabilities.get('EMPATE', 0)
            
            print(f"   {home_team[:20]}: {home_prob:.1%} chance | {away_team[:20]}: {away_prob:.1%} chance")
            
            # Feature explanation
            feature_explanation = self.get_feature_explanation(prediction)
            
            if feature_explanation['positive']:
                print(f"\\n✅ ¿POR QUÉ FAVORECE A {'LOCAL' if result == 'LOCAL' else 'VISITANTE' if result == 'VISITANTE' else 'EMPATE'}?")
                print("─" * 80)
                
                for j, (feature, importance) in enumerate(feature_explanation['positive'], 1):
                    formatted_feature = self.format_feature_name(feature)
                    print(f"  {j}. Señal del modelo: {formatted_feature} (+{importance:.3f}) ⭐")
            
            if feature_explanation['negative']:
                opposite_team = away_team if result == 'LOCAL' else home_team
                print(f"\\n❌ ¿QUÉ FAVORECE A {opposite_team.upper()}?")
                print("─" * 80)
                
                for j, (feature, importance) in enumerate(feature_explanation['negative'], 1):
                    formatted_feature = self.format_feature_name(feature)
                    print(f"  {j}. Señal del modelo: {formatted_feature} ({importance:.3f}) ⭐")
        
        # Summary
        print("\\n" + "─" * 80)
        print(f"📊 RESUMEN JORNADA {matchday}")
        print("─" * 80)
        
        # Count predictions
        local_wins = sum(1 for p in predictions if p['predicted_result'] == 'LOCAL')
        away_wins = sum(1 for p in predictions if p['predicted_result'] == 'VISITANTE')
        draws = sum(1 for p in predictions if p['predicted_result'] == 'EMPATE')
        
        print(f"Victorias locales: {local_wins}")
        print(f"Victorias visitantes: {away_wins}")
        print(f"Empates: {draws}")
        print(f"Total partidos: {len(predictions)}")
        
        # Model info
        avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
        print(f"Confianza promedio: {avg_confidence:.1%}")
        print(f"Modelo utilizado: {self.current_model}")
        
        print("\\n" + "─" * 80)
        
    def interactive_jornada_menu(self):
        """Interactive jornada selection menu"""
        if not self.initialize():
            print(" Failed to initialize system")
            return
        
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🏆 PREDICTOR POR JORNADA")
            print("=" * 50)
            
            # Get available jornadas
            matches_path = self.project_root / "data" / "cleaned" / "matches_2025_cleaned.csv"
            matches_df = pd.read_csv(matches_path)
            available_jornadas = sorted(matches_df['matchday'].unique())
            
            print(f"Jornadas disponibles: {min(available_jornadas)} - {max(available_jornadas)}")
            
            # Show recent jornada results if available
            finished_jornadas = []
            for jornada in available_jornadas:
                jornada_matches = matches_df[matches_df['matchday'] == jornada]
                if jornada_matches['status'].iloc[0] == 'FINISHED':
                    finished_jornadas.append(jornada)
            
            if finished_jornadas:
                print(f"Jornadas completadas: {len(finished_jornadas)}")
                print(f"Última jornada completada: {max(finished_jornadas)}")
            
            print("\\nOpciones:")
            print("1. Seleccionar jornada específica")
            print("2. Siguiente jornada no completada")
            print("3. Ver jornada actual")
            print("4. Cambiar modelo de predicción")
            print("0. Volver al menú principal")
            
            choice = input("\\nSelecciona opción: ").strip()
            
            if choice == '1':
                jornada = input("Ingresa número de jornada: ").strip()
                if jornada.isdigit() and int(jornada) in available_jornadas:
                    self.display_jornada_detailed(int(jornada))
                    input("\\nPress Enter to continue...")
                else:
                    print("Jornada no válida")
                    input("Press Enter to continue...")
            
            elif choice == '2':
                # Find next unfinished jornada
                next_jornada = None
                for jornada in available_jornadas:
                    jornada_matches = matches_df[matches_df['matchday'] == jornada]
                    if jornada_matches['status'].iloc[0] != 'FINISHED':
                        next_jornada = jornada
                        break
                
                if next_jornada:
                    self.display_jornada_detailed(next_jornada)
                    input("\\nPress Enter to continue...")
                else:
                    print("No hay jornadas pendientes")
                    input("Press Enter to continue...")
            
            elif choice == '3':
                # Current date-based jornada
                current_date = datetime.now()
                current_jornada = None
                
                for jornada in available_jornadas:
                    jornada_matches = matches_df[matches_df['matchday'] == jornada]
                    jornada_dates = pd.to_datetime(jornada_matches['date'])
                    
                    if current_date >= jornada_dates.min() and current_date <= jornada_dates.max():
                        current_jornada = jornada
                        break
                
                if current_jornada:
                    self.display_jornada_detailed(current_jornada)
                    input("\\nPress Enter to continue...")
                else:
                    print("No hay jornada activa actualmente")
                    input("Press Enter to continue...")
            
            elif choice == '4':
                self.change_model()
                input("Press Enter to continue...")
            
            elif choice == '0':
                break
            
            else:
                print("Opción no válida")
                input("Press Enter to continue...")
    
    def change_model(self):
        """Change prediction model"""
        available_models = list(self.predictor.models.keys())
        
        print("\\nModelos disponibles:")
        for i, model in enumerate(available_models, 1):
            current = " (ACTUAL)" if model == self.current_model else ""
            performance = self.predictor.get_model_performance()
            acc = performance.get(model, {}).get('test_accuracy', 0)
            print(f"{i}. {model}{current} - Accuracy: {acc:.3f}")
        
        choice = input(f"\\nSelecciona modelo (1-{len(available_models)}): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(available_models):
            self.current_model = available_models[int(choice) - 1]
            print(f"Modelo cambiado a: {self.current_model}")
        else:
            print("Opción no válida")

def main():
    predictor = JornadaPredictor()
    predictor.interactive_jornada_menu()

if __name__ == "__main__":
    main()