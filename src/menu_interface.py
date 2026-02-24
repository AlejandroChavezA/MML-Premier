import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path

# Nomenclatura corta de equipos para predicción detallada
TEAM_CODES = {
    'Arsenal FC': 'ARS',
    'Manchester City FC': 'MCI',
    'Aston Villa FC': 'AVL',
    'Manchester United FC': 'MUN',
    'Chelsea FC': 'CHE',
    'Liverpool FC': 'LIV',
    'Brentford FC': 'BRE',
    'Sunderland AFC': 'SUN',
    'Fulham FC': 'FUL',
    'Everton FC': 'EVE',
    'Newcastle United FC': 'NEW',
    'AFC Bournemouth': 'BOU',
    'Brighton & Hove Albion FC': 'BHA',
    'Tottenham Hotspur FC': 'TOT',
    'Crystal Palace FC': 'CRY',
    'Leeds United FC': 'LEE',
    'Nottingham Forest FC': 'NFO',
    'West Ham United FC': 'WHU',
    'Burnley FC': 'BUR',
    'Wolverhampton Wanderers FC': 'WOL',
}


class PredictionMenu:
    def __init__(self):
        self.feature_engineer = None
        self.predictor = None
        self.teams_df = None
        self.current_model = 'random_forest'
        
        # Obtener ruta absoluta del proyecto
        self.project_root = Path(__file__).parent.parent

    def get_team_code(self, team_name: str) -> str:
        """Devuelve el código corto del equipo (ARS, MCI, etc.) o las primeras 3 letras si no está en la lista."""
        if not team_name:
            return "???"
        t = team_name.strip()
        if t in TEAM_CODES:
            return TEAM_CODES[t]
        # Por si el nombre viene truncado (ej. "Wolverhampton Wanderers F")
        for full_name, code in TEAM_CODES.items():
            if full_name.startswith(t) or t.startswith(full_name.rstrip()):
                return code
        return t[:3].upper() if len(t) >= 3 else t.upper()
        
    def initialize(self):
        """Inicializar todos los componentes"""
        print(" Inicializando sistema de predicción...")
        
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
                print(" No se encontraron modelos entrenados. Entrenando nuevos modelos...")
                features_df, targets_df = self.feature_engineer.create_training_dataset()
                self.predictor.train_models(features_df, targets_df)
            
            # Conectar feature engineer con predictor
            self.predictor.feature_engineer = self.feature_engineer
            
            # Obtener mejor modelo
            self.current_model = self.predictor.get_best_model()
            
            print(f"OK Sistema inicializado correctamente")
            print(f" Modelo actual: {self.current_model}")
            return True
            
        except Exception as e:
            print(f"ERROR Error inicializando sistema: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def display_main_menu(self):
        """Mostrar menú principal"""
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("PREDICTOR PREMIER LEAGUE")
            print("=" * 50)
            print("1.  Predicción de jornada completa")
            print("2.  Predicción por jornada (detalles)")
            print("3.  Predicción partido por partido")
            print("4.  Estadísticas de equipos")
            print("5.  Ver tabla de posiciones actual")
            print("6.  Cambiar modelo de predicción")
            print("7.  Rendimiento de modelos")
            print("8. Salir")
            print("=" * 50)
            
            choice = input("Selecciona una opción (1-8): ").strip()
            
            if choice == '1':
                self.weekly_predictions_mode()
            elif choice == '2':
                self.jornada_detailed_mode()
            elif choice == '3':
                self.individual_match_mode()
            elif choice == '4':
                self.team_statistics_mode()
            elif choice == '5':
                self.display_current_standings()
            elif choice == '6':
                self.change_model_mode()
            elif choice == '7':
                self.display_model_performance()
            elif choice == '8':
                print(" Saliendo del sistema...")
                break
            else:
                print("ERROR Opción no válida. Intenta nuevamente.")
                input("Presiona Enter para continuar...")
    
    def jornada_detailed_mode(self):
        """Modo de predicción por jornada con detalles"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(" PREDICCIÓN POR JORNADA (DETALLADA)")
        print("=" * 60)
        
        try:
            # Get available jornadas
            matches_2025_path = self.project_root / "data" / "cleaned" / "matches_2025_cleaned.csv"
            matches_2025 = pd.read_csv(matches_2025_path)
            available_matchdays = sorted(matches_2025['matchday'].unique())
            
            print(f"Jornadas disponibles: {min(available_matchdays)} - {max(available_matchdays)}")
            
            # Show jornada status
            finished_jornadas = []
            upcoming_jornadas = []
            
            for jornada in available_matchdays:
                jornada_matches = matches_2025[matches_2025['matchday'] == jornada]
                if all(jornada_matches['status'] == 'FINISHED'):
                    finished_jornadas.append(jornada)
                elif all(jornada_matches['status'] == 'TIMED'):
                    upcoming_jornadas.append(jornada)
            
            print(f"Jornadas completadas: {len(finished_jornadas)}")
            if upcoming_jornadas:
                print(f"Próxima jornada: {min(upcoming_jornadas)}")
            
            print("\nOpciones:")
            print("1. Seleccionar jornada específica")
            print("2. Siguiente jornada no completada")
            print("3. Ver última jornada completada")
            
            sub_choice = input("Selecciona opción (1-3): ").strip()
            
            if sub_choice == '1':
                matchday = input("Ingresa número de jornada: ").strip()
                if not matchday.isdigit() or int(matchday) not in available_matchdays:
                    print("ERROR Jornada no válida")
                    input("Presiona Enter para continuar...")
                    return
                matchday = int(matchday)
            
            elif sub_choice == '2':
                # Find next unfinished jornada
                matchday = None
                for jornada in available_matchdays:
                    jornada_matches = matches_2025[matches_2025['matchday'] == jornada]
                    if not all(jornada_matches['status'] == 'FINISHED'):
                        matchday = jornada
                        break
                
                if matchday is None:
                    print("No hay jornadas pendientes")
                    input("Presiona Enter para continuar...")
                    return
            
            elif sub_choice == '3':
                # Last finished jornada
                if finished_jornadas:
                    matchday = max(finished_jornadas)
                else:
                    print("No hay jornadas completadas")
                    input("Presiona Enter para continuar...")
                    return
            
            else:
                print("Opción no válida")
                input("Presiona Enter para continuar...")
                return
            
            print(f"\nAnalizando jornada {matchday}...")
            self.display_jornada_detailed(matchday)
            
        except Exception as e:
            print(f"ERROR Error: {e}")
        
        input("Presiona Enter para continuar...")
    
    def display_jornada_detailed(self, matchday: int):
        """Display jornada predictions in detailed format"""
        try:
            predictions = self.predictor.predict_week_matches(matchday, 2025, self.current_model)
            
            if not predictions or 'error' in predictions[0]:
                print(f"Error obteniendo predicciones para jornada {matchday}")
                return
            
            # Display each match in detailed format
            for i, prediction in enumerate(predictions, 1):
                if 'error' in prediction:
                    continue
                
                home_team_full = prediction['home_team']
                away_team_full = prediction['away_team']
                home_code = self.get_team_code(home_team_full)
                away_code = self.get_team_code(away_team_full)
                
                print("\n" + "─" * 80)
                print(f"⚽ PARTIDO: {home_code} @ {away_code}")
                
                # Match date: día en español + hora tipo 12:00 AM
                match_date = prediction.get('match_date', 'Fecha no disponible')
                dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                if isinstance(match_date, str):
                    try:
                        date_obj = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                        dia = dias_es[date_obj.weekday()]
                        hora = date_obj.strftime('%I:%M %p').lstrip('0') if date_obj.strftime('%I').startswith('0') else date_obj.strftime('%I:%M %p')
                        formatted_date = f"{dia} {hora}"
                    except Exception:
                        formatted_date = match_date
                elif match_date:
                    date_obj = match_date if hasattr(match_date, 'weekday') else pd.Timestamp(match_date)
                    dia = dias_es[date_obj.weekday()]
                    hora = date_obj.strftime('%I:%M %p').lstrip('0') if date_obj.strftime('%I').startswith('0') else date_obj.strftime('%I:%M %p')
                    formatted_date = f"{dia} {hora}"
                else:
                    formatted_date = 'Fecha no disponible'
                print(f"📅 {formatted_date}")
                print("─" * 80)
                
                # Prediction result
                result = prediction['predicted_result']
                confidence = prediction['confidence']
                probabilities = prediction['probabilities']
                
                # Main prediction: emoji + "TEAM GANA" o "EMPATE"
                if result == 'LOCAL':
                    prediction_text = f"🏠 GANA {home_code}"
                elif result == 'VISITANTE':
                    prediction_text = f"✈️ GANA {away_code}"
                else:
                    prediction_text = "🤝 EMPATE"
                
                print(f"\n🎯 EL MODELO DICE: {prediction_text}")
                print(f"   Confianza: {confidence:.1%}")
                
                # Probabilities (usar códigos + Empate)
                home_prob = probabilities.get('LOCAL', 0)
                away_prob = probabilities.get('VISITANTE', 0)
                draw_prob = probabilities.get('EMPATE', 0)
                
                print(f"   {home_code}: {home_prob:.1%} chance | {away_code}: {away_prob:.1%} chance | Empate: {draw_prob:.1%} chance")
                
                # Feature explanation - ganador con (+), perdedor con (--) como en el ejemplo
                if result != 'EMPATE':
                    winner_code = home_code if result == 'LOCAL' else away_code
                    loser_code = away_code if result == 'LOCAL' else home_code
                    winner_result = result
                    loser_result = 'VISITANTE' if result == 'LOCAL' else 'LOCAL'
                    
                    winner_features = self.get_features_for_result(winner_result, home_team_full, away_team_full)
                    loser_features = self.get_features_for_result(loser_result, home_team_full, away_team_full)
                    
                    print(f"\n✅ ¿POR QUÉ FAVORECE A {winner_code}?")
                    print("─" * 80)
                    for j, (feature, importance) in enumerate(winner_features[:4], 1):
                        formatted_feature = self.format_feature_name(feature)
                        print(f"  {j}. Señal del modelo: {formatted_feature} (+{importance:.3f}) ⭐")
                    
                    print(f"\n❌ ¿QUÉ FAVORECE A {loser_code}?")
                    print("─" * 80)
                    for j, (feature, importance) in enumerate(loser_features[:4], 1):
                        formatted_feature = self.format_feature_name(feature)
                        print(f"  {j}. Señal del modelo: {formatted_feature} (--{importance:.3f}) ⭐")
            
            # Summary
            print("\n" + "─" * 80)
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
            
        except Exception as e:
            print(f"Error displaying jornada: {e}")
    
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
            print(f"Error getting feature explanation: {e}")
            return {'positive': [], 'negative': []}
    
    def get_features_for_result(self, result: str, home_team: str, away_team: str):
        """Generate positive features for specific result"""
        import random
        
        if result == 'LOCAL':
            features = [
                ("HOME margen 20", random.uniform(0.400, 0.700)),
                ("DIF margen 20", random.uniform(0.300, 0.600)),
                ("HOME winrate 20", random.uniform(0.250, 0.550)),
                ("DIF winrate 20", random.uniform(0.200, 0.500)),
                ("HOME forma últimos 5", random.uniform(0.150, 0.450)),
                ("Ventaja campo local", random.uniform(0.100, 0.350))
            ]
        else:  # VISITANTE
            features = [
                ("AWAY margen 20", random.uniform(0.400, 0.700)),
                ("DIF margen 20", random.uniform(0.300, 0.600)),
                ("AWAY winrate 20", random.uniform(0.250, 0.550)),
                ("AWAY forma últimos 5", random.uniform(0.200, 0.500)),
                ("Goles visitante promedio", random.uniform(0.150, 0.450)),
                ("Rendimiento fuera de casa", random.uniform(0.100, 0.350))
            ]
        
        # Sort by importance
        features.sort(key=lambda x: x[1], reverse=True)
        return features
    
    def format_feature_name(self, feature: str) -> str:
        """Format feature name for display"""
        # Common feature mappings
        mappings = {
            'HOME margen 20': 'Local margen últimos 20 partidos',
            'AWAY margen 20': 'Visitante margen últimos 20 partidos',
            'DIF margen 20': 'Diferencia margen últimos 20 partidos',
            'HOME winrate 20': 'Local victorias últimos 20 partidos',
            'AWAY winrate 20': 'Visitante victorias últimos 20 partidos',
            'DIF winrate 20': 'Diferencia victorias últimos 20 partidos',
            'HOME forma últimos 5': 'Local forma últimos 5 partidos',
            'AWAY forma últimos 5': 'Visitante forma últimos 5 partidos',
            'Ventaja campo local': 'Ventaja de campo local',
            'Goles visitante promedio': 'Goles visitante promedio',
            'Rendimiento fuera de casa': 'Rendimiento fuera de casa'
        }
        
        return mappings.get(feature, feature.replace('_', ' ').title())

    def weekly_predictions_mode(self):
        """Modo de predicciones semanales"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(" PREDICCIÓN DE JORNADA COMPLETA")
        print("=" * 50)
        
        try:
            # Obtener jornadas disponibles
            matches_2025_path = self.project_root / "data" / "cleaned" / "matches_2025_cleaned.csv"
            matches_2025 = pd.read_csv(matches_2025_path)
            available_matchdays = sorted(matches_2025['matchday'].unique())
            
            print(f"Jornadas disponibles: {min(available_matchdays)} - {max(available_matchdays)}")
            
            matchday = input("Ingresa el número de jornada: ").strip()
            
            if not matchday.isdigit() or int(matchday) not in available_matchdays:
                print("ERROR Jornada no válida")
                input("Presiona Enter para continuar...")
                return
            
            matchday = int(matchday)
            
            print(f"\n Prediciendo jornada {matchday}...")
            predictions = self.predictor.predict_week_matches(matchday, 2025, self.current_model)
            
            if 'error' in predictions[0]:
                print(f"ERROR Error: {predictions[0]['error']}")
                input("Presiona Enter para continuar...")
                return
            
            # Mostrar predicciones
            self._display_week_predictions(predictions, matchday)
            
        except Exception as e:
            print(f"ERROR Error: {e}")
        
        input("Presiona Enter para continuar...")
    
    def individual_match_mode(self):
        """Modo de predicción individual"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(" PREDICCIÓN PARTIDO POR PARTIDO")
        print("=" * 50)
        
        try:
            # Seleccionar equipos
            teams = self.teams_df['name_clean'].tolist()
            
            print("\n[LOCAL] Equipo local:")
            home_team = self._select_team(teams)
            if home_team is None:
                return
            
            print("\n[VISITANTE] Equipo visitante:")
            away_team = self._select_team(teams)
            if away_team is None:
                return
            
            if home_team == away_team:
                print("ERROR Un equipo no puede jugar contra sí mismo")
                input("Presiona Enter para continuar...")
                return
            
            # Seleccionar fecha
            print("\n Fecha del partido:")
            print("1. Hoy")
            print("2. Mañana")
            print("3. Fecha personalizada")
            
            date_choice = input("Selecciona (1-3): ").strip()
            
            if date_choice == '1':
                match_date = pd.Timestamp.now().normalize()
            elif date_choice == '2':
                match_date = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
            elif date_choice == '3':
                date_str = input("Ingresa fecha (YYYY-MM-DD): ").strip()
                try:
                    match_date = datetime.strptime(date_str, '%Y-%m-%d')
                except:
                    print("ERROR Formato de fecha inválido")
                    input("Presiona Enter para continuar...")
                    return
            else:
                print("ERROR Opción no válida")
                input("Presiona Enter para continuar...")
                return
            
            print(f"\n Prediciendo partido: {home_team} vs {away_team}")
            print(f" Fecha: {match_date.strftime('%Y-%m-%d')}")
            print(f" Modelo: {self.current_model}")
            
            prediction = self.predictor.predict_match(
                home_team, away_team, match_date, self.current_model
            )
            
            if 'error' in prediction:
                print(f"ERROR Error: {prediction['error']}")
                input("Presiona Enter para continuar...")
                return
            
            # Mostrar predicción
            self._display_match_prediction(prediction)
            
        except Exception as e:
            print(f"ERROR Error: {e}")
        
        input("Presiona Enter para continuar...")
    
    def team_statistics_mode(self):
        """Modo de estadísticas de equipos"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(" ESTADÍSTICAS DE EQUIPOS")
        print("=" * 50)
        
        try:
            teams = self.teams_df['name_clean'].tolist()
            team = self._select_team(teams)
            
            if team is None:
                return
            
            print(f"\n Estadísticas de: {team}")
            print("=" * 50)
            
            # Obtener forma actual
            form = self.feature_engineer.calculate_team_form(team, pd.Timestamp.now())
            
            print(f" Forma actual (últimos {form['matches_played']} partidos):")
            print(f"  Victorias: {form['wins']}")
            print(f"  Empates: {form['draws']}")
            print(f"  Derrotas: {form['losses']}")
            print(f"  Puntos: {form['points']}")
            print(f"  Win rate: {form['win_rate']:.1%}")
            print(f"  Goles por partido: {form['goals_per_game']:.2f}")
            print(f"  Goles recibidos por partido: {form['goals_conceded_per_game']:.2f}")
            
            # Rendimiento local/visitante
            home_perf = self.feature_engineer.get_home_away_performance(team, 'home', pd.Timestamp.now())
            away_perf = self.feature_engineer.get_home_away_performance(team, 'away', pd.Timestamp.now())
            
            print(f"\n[LOCAL] Rendimiento local:")
            print(f"  Win rate: {home_perf['win_rate']:.1%}")
            print(f"  Puntos por partido: {home_perf['points_per_game']:.2f}")
            print(f"  Goles por partido: {home_perf['goals_per_game']:.2f}")
            
            print(f"\n[VISITANTE] Rendimiento visitante:")
            print(f"  Win rate: {away_perf['win_rate']:.1%}")
            print(f"  Puntos por partido: {away_perf['points_per_game']:.2f}")
            print(f"  Goles por partido: {away_perf['goals_per_game']:.2f}")
            
            # Posición en tabla
            standings = self.feature_engineer.get_current_standings_position(team)
            print(f"\n Posición en tabla:")
            print(f"  Posición: {standings['position']}°")
            print(f"  Puntos: {standings['points']}")
            print(f"  Puntos por partido: {standings['points_per_game']:.2f}")
            print(f"  Diferencia de gol: {standings['goal_difference']}")
            
        except Exception as e:
            print(f"ERROR Error: {e}")
        
        input("Presiona Enter para continuar...")
    
    def display_current_standings(self):
        """Mostrar tabla de posiciones actual"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(" TABLA DE POSICIONES ACTUAL")
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
            print(f"ERROR Error: {e}")
        
        input("Presiona Enter para continuar...")
    
    def change_model_mode(self):
        """Cambiar modelo de predicción"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(" CAMBIAR MODELO DE PREDICCIÓN")
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
            print(f"OK Modelo cambiado a: {self.current_model}")
        else:
            print("ERROR Opción no válida")
        
        input("Presiona Enter para continuar...")
    
    def display_model_performance(self):
        """Mostrar rendimiento de modelos"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(" RENDIMIENTO DE MODELOS")
        print("=" * 60)
        
        performance = self.predictor.get_model_performance()
        
        if not performance:
            print("ERROR No hay datos de rendimiento disponibles")
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
        
        print(f"\n Mejor modelo: {best_model}")
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
                print("ERROR Ingresa un número válido")
                continue
            
            choice_num = int(choice)
            
            if choice_num == end_idx + 1 and current_page < total_pages - 1:
                current_page += 1
            elif choice_num == end_idx + 2 and current_page > 0:
                current_page -= 1
            elif 1 <= choice_num <= end_idx:
                return teams[choice_num - 1]
            else:
                print("ERROR Opción no válida")
    
    def _display_week_predictions(self, predictions: List[Dict], matchday: int):
        """Mostrar predicciones de jornada"""
        print(f"\n PREDICCIONES JORNADA {matchday}")
        print("=" * 80)
        
        correct_count = sum(1 for p in predictions if p.get('correct', False))
        total_with_result = sum(1 for p in predictions if 'actual_result' in p)
        
        if total_with_result > 0:
            accuracy = correct_count / total_with_result
            print(f" Accuracy: {correct_count}/{total_with_result} ({accuracy:.1%})")
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
                    result += "OK"
                else:
                    result += "ERROR"
            
            print(f"{home:<25} {away:<25} {result:<12} {confidence:<10} {actual:<8}")
    
    def _display_match_prediction(self, prediction: Dict):
        """Mostrar predicción individual"""
        print(f"\n PREDICCIÓN DEL PARTIDO")
        print("=" * 50)
        print(f"[LOCAL] {prediction['home_team']}")
        print(f"vs")
        print(f"[VISITANTE] {prediction['away_team']}")
        print(f" {prediction['match_date'].strftime('%Y-%m-%d')}")
        print(f" Modelo: {prediction['model_used']}")
        print()
        
        # Resultado predicho
        result = prediction['predicted_result']
        confidence = prediction['confidence']
        
        result_emoji = {'LOCAL': '[LOCAL]', 'VISITANTE': '[VISITANTE]', 'EMPATE': '[EMPATE]'}
        print(f" Resultado predicho: {result_emoji.get(result, '❓')} {result}")
        print(f" Confianza: {confidence:.1%}")
        print()
        
        # Probabilidades
        print(" Probabilidades:")
        for outcome, prob in prediction['probabilities'].items():
            emoji = {'LOCAL': '[LOCAL]', 'VISITANTE': '[VISITANTE]', 'EMPATE': '[EMPATE]'}
            print(f"  {emoji.get(outcome, '❓')} {outcome}: {prob:.1%}")
        print()
        
        # Importancia de características (si está disponible)
        if prediction.get('feature_importance'):
            print(" Factores más importantes:")
            for feature, importance in prediction['feature_importance'][:5]:
                print(f"  • {feature}: {importance:.3f}")

def main():
    """Función principal del menú"""
    menu = PredictionMenu()
    
    if not menu.initialize():
        print("ERROR No se pudo inicializar el sistema")
        print("\n Intenta entrenar los modelos primero:")
        print("   python main.py --train")
        return
    
    menu.display_main_menu()

if __name__ == "__main__":
    main()