import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path

class PredictionMenu:
    def __init__(self):
        self.feature_engineer = None
        self.predictor = None
        self.teams_df = None
        self.current_model = 'random_forest'
        
        # Obtener ruta absoluta del proyecto
        self.project_root = Path(__file__).parent.parent
        
    def initialize(self):
        """Inicializar todos los componentes"""
        print("🚀 Inicializando sistema de predicción...")
        
        try:
            # Importar clases
            sys.path.insert(0, str(self.project_root / "src"))
            from feature_engineering import FeatureEngineer
            from prediction_models import MatchPredictor
            
            # Inicializar componentes con rutas absolutas
            data_dir = self.project_root / "data" / "cleaned"
            models_dir = self.project_root / "models"
            
            self.feature_engineer = FeatureEngineer(data_dir=str(data_dir))
            self.predictor = MatchPredictor(models_dir=str(models_dir))
            
            # Cargar datos
            if not self.feature_engineer.load_data():
                return False
            
            # Cargar equipos
            teams_path = data_dir / "teams_cleaned.csv"
            self.teams_df = pd.read_csv(teams_path)
            
            # Cargar modelos entrenados o entrenar nuevos
            if not self.predictor.load_models():
                print("📊 No se encontraron modelos entrenados. Entrenando nuevos modelos...")
                features_df, targets_df = self.feature_engineer.create_training_dataset()
                self.predictor.train_models(features_df, targets_df)
            
            # Conectar feature engineer con predictor
            self.predictor.feature_engineer = self.feature_engineer
            
            # Obtener mejor modelo
            self.current_model = self.predictor.get_best_model()
            
            print(f"✅ Sistema inicializado correctamente")
            print(f"🤖 Modelo actual: {self.current_model}")
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando sistema: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def display_main_menu(self):
        """Mostrar menú principal"""
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("⚽ PREDICTOR PREMIER LEAGUE ⚽")
            print("=" * 50)
            print("1. 📅 Predicción de jornada completa")
            print("2. 🎯 Predicción partido por partido")
            print("3. 📊 Estadísticas de equipos")
            print("4. 🏆 Ver tabla de posiciones actual")
            print("5. 🤖 Cambiar modelo de predicción")
            print("6. 📈 Rendimiento de modelos")
            print("7. ❌ Salir")
            print("=" * 50)
            
            choice = input("Selecciona una opción (1-7): ").strip()
            
            if choice == '1':
                self.weekly_predictions_mode()
            elif choice == '2':
                self.individual_match_mode()
            elif choice == '3':
                self.team_statistics_mode()
            elif choice == '4':
                self.display_current_standings()
            elif choice == '5':
                self.change_model_mode()
            elif choice == '6':
                self.display_model_performance()
            elif choice == '7':
                print("👋 ¡Gracias por usar el Predictor Premier League!")
                break
            else:
                print("❌ Opción no válida. Intenta nuevamente.")
                input("Presiona Enter para continuar...")
    
    def weekly_predictions_mode(self):
        """Modo de predicciones semanales"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("📅 PREDICCIÓN DE JORNADA COMPLETA")
        print("=" * 50)
        
        try:
            # Obtener jornadas disponibles
            matches_2025_path = self.project_root / "data" / "cleaned" / "matches_2025_cleaned.csv"
            matches_2025 = pd.read_csv(matches_2025_path)
            available_matchdays = sorted(matches_2025['matchday'].unique())
            
            print(f"Jornadas disponibles: {min(available_matchdays)} - {max(available_matchdays)}")
            
            matchday = input("Ingresa el número de jornada: ").strip()
            
            if not matchday.isdigit() or int(matchday) not in available_matchdays:
                print("❌ Jornada no válida")
                input("Presiona Enter para continuar...")
                return
            
            matchday = int(matchday)
            
            print(f"\n🔮 Prediciendo jornada {matchday}...")
            predictions = self.predictor.predict_week_matches(matchday, 2025, self.current_model)
            
            if 'error' in predictions[0]:
                print(f"❌ Error: {predictions[0]['error']}")
                input("Presiona Enter para continuar...")
                return
            
            # Mostrar predicciones
            self._display_week_predictions(predictions, matchday)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("Presiona Enter para continuar...")
    
    def individual_match_mode(self):
        """Modo de predicción individual"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🎯 PREDICCIÓN PARTIDO POR PARTIDO")
        print("=" * 50)
        
        try:
            # Seleccionar equipos
            teams = self.teams_df['name_clean'].tolist()
            
            print("\n🏠 Equipo local:")
            home_team = self._select_team(teams)
            if home_team is None:
                return
            
            print("\n✈️ Equipo visitante:")
            away_team = self._select_team(teams)
            if away_team is None:
                return
            
            if home_team == away_team:
                print("❌ Un equipo no puede jugar contra sí mismo")
                input("Presiona Enter para continuar...")
                return
            
            # Seleccionar fecha
            print("\n📅 Fecha del partido:")
            print("1. Hoy")
            print("2. Mañana")
            print("3. Fecha personalizada")
            
            date_choice = input("Selecciona (1-3): ").strip()
            
            if date_choice == '1':
                match_date = datetime.now()
            elif date_choice == '2':
                match_date = datetime.now() + timedelta(days=1)
            elif date_choice == '3':
                date_str = input("Ingresa fecha (YYYY-MM-DD): ").strip()
                try:
                    match_date = datetime.strptime(date_str, '%Y-%m-%d')
                except:
                    print("❌ Formato de fecha inválido")
                    input("Presiona Enter para continuar...")
                    return
            else:
                print("❌ Opción no válida")
                input("Presiona Enter para continuar...")
                return
            
            print(f"\n🔮 Prediciendo partido: {home_team} vs {away_team}")
            print(f"📅 Fecha: {match_date.strftime('%Y-%m-%d')}")
            print(f"🤖 Modelo: {self.current_model}")
            
            prediction = self.predictor.predict_match(
                home_team, away_team, match_date, self.current_model
            )
            
            if 'error' in prediction:
                print(f"❌ Error: {prediction['error']}")
                input("Presiona Enter para continuar...")
                return
            
            # Mostrar predicción
            self._display_match_prediction(prediction)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("Presiona Enter para continuar...")
    
    def team_statistics_mode(self):
        """Modo de estadísticas de equipos"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("📊 ESTADÍSTICAS DE EQUIPOS")
        print("=" * 50)
        
        try:
            teams = self.teams_df['name_clean'].tolist()
            team = self._select_team(teams)
            
            if team is None:
                return
            
            print(f"\n📈 Estadísticas de: {team}")
            print("=" * 50)
            
            # Obtener forma actual
            form = self.feature_engineer.calculate_team_form(team, datetime.now())
            
            print(f"📊 Forma actual (últimos {form['matches_played']} partidos):")
            print(f"  Victorias: {form['wins']}")
            print(f"  Empates: {form['draws']}")
            print(f"  Derrotas: {form['losses']}")
            print(f"  Puntos: {form['points']}")
            print(f"  Win rate: {form['win_rate']:.1%}")
            print(f"  Goles por partido: {form['goals_per_game']:.2f}")
            print(f"  Goles recibidos por partido: {form['goals_conceded_per_game']:.2f}")
            
            # Rendimiento local/visitante
            home_perf = self.feature_engineer.get_home_away_performance(team, 'home', datetime.now())
            away_perf = self.feature_engineer.get_home_away_performance(team, 'away', datetime.now())
            
            print(f"\n🏠 Rendimiento local:")
            print(f"  Win rate: {home_perf['win_rate']:.1%}")
            print(f"  Puntos por partido: {home_perf['points_per_game']:.2f}")
            print(f"  Goles por partido: {home_perf['goals_per_game']:.2f}")
            
            print(f"\n✈️ Rendimiento visitante:")
            print(f"  Win rate: {away_perf['win_rate']:.1%}")
            print(f"  Puntos por partido: {away_perf['points_per_game']:.2f}")
            print(f"  Goles por partido: {away_perf['goals_per_game']:.2f}")
            
            # Posición en tabla
            standings = self.feature_engineer.get_current_standings_position(team)
            print(f"\n🏆 Posición en tabla:")
            print(f"  Posición: {standings['position']}°")
            print(f"  Puntos: {standings['points']}")
            print(f"  Puntos por partido: {standings['points_per_game']:.2f}")
            print(f"  Diferencia de gol: {standings['goal_difference']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("Presiona Enter para continuar...")
    
    def display_current_standings(self):
        """Mostrar tabla de posiciones actual"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🏆 TABLA DE POSICIONES ACTUAL")
        print("=" * 70)
        
        try:
            standings_path = self.project_root / "data" / "cleaned" / "standings_2025_cleaned.csv"
            standings = pd.read_csv(standings_path)
            
            print(f"{'Pos':<4} {'Equipo':<25} {'PJ':<3} {'PG':<3} {'PE':<3} {'PP':<3} {'PTS':<4} {'DG':<5}")
            print("-" * 70)
            
            for _, row in standings.iterrows():
                team_name = row['team'][:24]  # Limitar longitud
                print(f"{row['position']:<4} {team_name:<25} {row['played_games']:<3} "
                      f"{row['won']:<3} {row['draw']:<3} {row['lost']:<3} "
                      f"{row['points']:<4} {row['goal_difference']:<5}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("Presiona Enter para continuar...")
    
    def change_model_mode(self):
        """Cambiar modelo de predicción"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🤖 CAMBIAR MODELO DE PREDICCIÓN")
        print("=" * 50)
        
        available_models = list(self.predictor.models.keys())
        performance = self.predictor.get_model_performance()
        
        print(f"Modelo actual: {self.current_model}")
        print("\nModelos disponibles:")
        
        for i, model in enumerate(available_models, 1):
            acc = performance.get(model, {}).get('test_accuracy', 0)
            current = " (ACTUAL)" if model == self.current_model else ""
            print(f"{i}. {model}{current} - Accuracy: {acc:.3f}")
        
        choice = input(f"\nSelecciona modelo (1-{len(available_models)}): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(available_models):
            self.current_model = available_models[int(choice) - 1]
            print(f"✅ Modelo cambiado a: {self.current_model}")
        else:
            print("❌ Opción no válida")
        
        input("Presiona Enter para continuar...")
    
    def display_model_performance(self):
        """Mostrar rendimiento de modelos"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("📈 RENDIMIENTO DE MODELOS")
        print("=" * 60)
        
        performance = self.predictor.get_model_performance()
        
        if not performance:
            print("❌ No hay datos de rendimiento disponibles")
            input("Presiona Enter para continuar...")
            return
        
        print(f"{'Modelo':<20} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12}")
        print("-" * 60)
        
        for model_name, metrics in performance.items():
            print(f"{model_name:<20} "
                  f"{metrics['train_accuracy']:<12.3f} "
                  f"{metrics['test_accuracy']:<12.3f} "
                  f"{metrics['cv_mean']:<12.3f}")
        
        # Información adicional del mejor modelo
        best_model = self.predictor.get_best_model()
        best_metrics = performance[best_model]
        
        print(f"\n🏆 Mejor modelo: {best_model}")
        print(f"   Accuracy prueba: {best_metrics['test_accuracy']:.3f}")
        print(f"   Validación cruzada: {best_metrics['cv_mean']:.3f} ± {best_metrics['cv_std']:.3f}")
        
        input("Presiona Enter para continuar...")
    
    def _select_team(self, teams: List[str]) -> Optional[str]:
        """Seleccionar equipo de una lista"""
        print("\nEquipos disponibles:")
        
        # Mostrar equipos en páginas de 10
        page_size = 10
        current_page = 0
        total_pages = (len(teams) + page_size - 1) // page_size
        
        while True:
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, len(teams))
            
            print(f"\nPágina {current_page + 1}/{total_pages}:")
            for i in range(start_idx, end_idx):
                print(f"{i + 1:2d}. {teams[i]}")
            
            if current_page < total_pages - 1:
                print(f"{end_idx + 1:2d}. Siguiente página →")
            if current_page > 0:
                print(f"{end_idx + 2:2d}. ← Página anterior")
            
            choice = input(f"\nSelecciona equipo (1-{end_idx + 2}): ").strip()
            
            if not choice.isdigit():
                print("❌ Ingresa un número válido")
                continue
            
            choice_num = int(choice)
            
            if choice_num == end_idx + 1 and current_page < total_pages - 1:
                current_page += 1
            elif choice_num == end_idx + 2 and current_page > 0:
                current_page -= 1
            elif 1 <= choice_num <= end_idx:
                return teams[choice_num - 1]
            else:
                print("❌ Opción no válida")
    
    def _display_week_predictions(self, predictions: List[Dict], matchday: int):
        """Mostrar predicciones de jornada"""
        print(f"\n📅 PREDICCIONES JORNADA {matchday}")
        print("=" * 80)
        
        correct_count = sum(1 for p in predictions if p.get('correct', False))
        total_with_result = sum(1 for p in predictions if 'actual_result' in p)
        
        if total_with_result > 0:
            accuracy = correct_count / total_with_result
            print(f"🎯 Accuracy: {correct_count}/{total_with_result} ({accuracy:.1%})")
            print()
        
        print(f"{'Local':<25} {'Visitante':<25} {'Predicción':<12} {'Confianza':<10} {'Real':<8}")
        print("-" * 80)
        
        for pred in predictions:
            if 'error' in pred:
                continue
            
            home = pred['home_team'][:24]
            away = pred['away_team'][:24]
            result = pred['predicted_result']
            confidence = f"{pred['confidence']:.1%}"
            actual = pred.get('actual_result', 'N/A')
            
            # Colorear según si es correcto
            if actual != 'N/A':
                if result == actual:
                    result += "✅"
                else:
                    result += "❌"
            
            print(f"{home:<25} {away:<25} {result:<12} {confidence:<10} {actual:<8}")
    
    def _display_match_prediction(self, prediction: Dict):
        """Mostrar predicción individual"""
        print(f"\n🎯 PREDICCIÓN DEL PARTIDO")
        print("=" * 50)
        print(f"🏠 {prediction['home_team']}")
        print(f"vs")
        print(f"✈️ {prediction['away_team']}")
        print(f"📅 {prediction['match_date'].strftime('%Y-%m-%d')}")
        print(f"🤖 Modelo: {prediction['model_used']}")
        print()
        
        # Resultado predicho
        result = prediction['predicted_result']
        confidence = prediction['confidence']
        
        result_emoji = {'LOCAL': '🏠', 'VISITANTE': '✈️', 'EMPATE': '🤝'}
        print(f"🔮 Resultado predicho: {result_emoji.get(result, '❓')} {result}")
        print(f"📊 Confianza: {confidence:.1%}")
        print()
        
        # Probabilidades
        print("📈 Probabilidades:")
        for outcome, prob in prediction['probabilities'].items():
            emoji = {'LOCAL': '🏠', 'VISITANTE': '✈️', 'EMPATE': '🤝'}
            print(f"  {emoji.get(outcome, '❓')} {outcome}: {prob:.1%}")
        print()
        
        # Importancia de características (si está disponible)
        if prediction.get('feature_importance'):
            print("🔍 Factores más importantes:")
            for feature, importance in prediction['feature_importance'][:5]:
                print(f"  • {feature}: {importance:.3f}")

def main():
    """Función principal del menú"""
    menu = PredictionMenu()
    
    if not menu.initialize():
        print("❌ No se pudo inicializar el sistema")
        print("\n💡 Intenta entrenar los modelos primero:")
        print("   python main.py --train")
        return
    
    menu.display_main_menu()

if __name__ == "__main__":
    main()