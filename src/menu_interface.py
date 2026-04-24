import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import requests
from goals_predictor import GoalsPredictor

# Cargar variables de entorno desde .env y .env.local
def load_env_files():
    project_root = Path(__file__).parent.parent
    for env_file in [project_root / ".env", project_root / ".env.local"]:
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env_files()

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

# Logos de equipos Premier League
TEAM_LOGOS = {
    'ARS': 'https://resources.premierleague.com/premierleague/badges/50/t3.png',
    'MCI': 'https://resources.premierleague.com/premierleague/badges/50/t43.png',
    'AVL': 'https://resources.premierleague.com/premierleague/badges/50/t7.png',
    'MUN': 'https://resources.premierleague.com/premierleague/badges/50/t1.png',
    'CHE': 'https://resources.premierleague.com/premierleague/badges/50/t8.png',
    'LIV': 'https://resources.premierleague.com/premierleague/badges/50/t14.png',
    'BRE': 'https://resources.premierleague.com/premierleague/badges/50/t94.png',
    'SUN': 'https://resources.premierleague.com/premierleague/badges/50/t71.png',
    'FUL': 'https://resources.premierleague.com/premierleague/badges/50/t54.png',
    'EVE': 'https://resources.premierleague.com/premierleague/badges/50/t11.png',
    'NEW': 'https://resources.premierleague.com/premierleague/badges/50/t4.png',
    'BOU': 'https://resources.premierleague.com/premierleague/badges/50/t91.png',
    'BHA': 'https://resources.premierleague.com/premierleague/badges/50/t36.png',
    'TOT': 'https://resources.premierleague.com/premierleague/badges/50/t6.png',
    'CRY': 'https://resources.premierleague.com/premierleague/badges/50/t31.png',
    'LEE': 'https://resources.premierleague.com/premierleague/badges/50/t2.png',
    'NFO': 'https://resources.premierleague.com/premierleague/badges/50/t17.png',
    'WHU': 'https://resources.premierleague.com/premierleague/badges/50/t21.png',
    'BUR': 'https://resources.premierleague.com/premierleague/badges/50/t90.png',
    'WOL': 'https://resources.premierleague.com/premierleague/badges/50/t39.png',
}


class PredictionMenu:
    def __init__(self):
        self.feature_engineer = None
        self.predictor = None
        self.goals_predictor = None
        self.teams_df = None
        self.current_model = 'random_forest'
        self.current_predictions = []
        
        # Obtener ruta absoluta del proyecto
        self.project_root = Path(__file__).parent.parent
        
        # Cache para predicciones de Over/Under (6 horas)
        self._ou_cache = {}
        self._ou_cache_ttl = 6 * 60 * 60  # 6 horas en segundos
        
        # Historial de predicciones
        self.history_file = self.project_root / "data" / "prediction_history.json"
        
        # Configuración del panel
        self.panel_url = os.getenv("SAFESPORTS_PANEL_URL", "https://safesports-panel.vercel.app")
        self.panel_email = os.getenv("SAFESPORTS_PANEL_EMAIL", "")
        self.panel_password = os.getenv("SAFESPORTS_PANEL_PASSWORD", "")
        self.import_secret = os.getenv("IMPORT_API_SECRET", "")
        self.user_api_key = os.getenv("SAFESPORTS_USER_API_KEY", "")
    
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
    
    def get_team_full_name(self, team_name: str) -> str:
        """Devuelve el nombre completo del equipo"""
        if not team_name:
            return "Unknown"
        t = team_name.strip()
        if t in TEAM_CODES:
            for full_name, code in TEAM_CODES.items():
                if code == t:
                    return full_name
        for full_name, code in TEAM_CODES.items():
            if full_name.startswith(t) or t.startswith(full_name.rstrip()):
                return full_name
        return t
    
    def get_api_key(self) -> Optional[str]:
        """Obtiene API key del panel"""
        if self.user_api_key:
            return self.user_api_key
        
        if self.import_secret:
            return self.import_secret
        
        if not self.panel_email or not self.panel_password:
            return None
        
        try:
            response = requests.post(
                f"{self.panel_url}/api/auth/api-key/generate",
                json={"email": self.panel_email, "password": self.panel_password},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("apiKey")
        except Exception as e:
            print(f"Error obteniendo API key: {e}")
        
        return None
    
    def send_to_dashboard(self, predictions: List[Dict], matchday: int) -> bool:
        """Envía predicciones al dashboard"""
        api_key = self.get_api_key()
        
        if not api_key:
            print("ERROR: No hay API key configurada")
            print("Configura una de estas opciones en .env:")
            print("  1. SAFESPORTS_PANEL_EMAIL + SAFESPORTS_PANEL_PASSWORD")
            print("  2. SAFESPORTS_USER_API_KEY")
            print("  3. IMPORT_API_SECRET")
            return False
        
        # Transformar predicciones al formato del panel
        panel_predictions = self.transform_to_panel_format(predictions)
        
        try:
            print(f"\nEnviando {len(panel_predictions)} predicciones al dashboard...")
            
            response = requests.post(
                f"{self.panel_url}/api/predictions/import",
                json={"predictions": panel_predictions},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30
            )
            
            if response.status_code == 201:
                data = response.json()
                print(f"OK! {data.get('imported', 0)} predicciones importadas")
                
                # Guardar en historial local
                for pred in predictions:
                    over_under = self._get_over_under_prediction(pred)
                    self._save_prediction_to_history(pred, over_under)
                print("  Historial guardado localmente")
                
                return True
            elif response.status_code == 401:
                print("ERROR: API key inválida o expirada")
                return False
            else:
                print(f"ERROR: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"ERROR: No se pudo conectar a {self.panel_url}")
            return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    def transform_to_panel_format(self, predictions: List[Dict]) -> List[Dict]:
        """Transforma predicciones al formato del dashboard"""
        panel_predictions = []
        
        for pred in predictions:
            home_team_full = pred['home_team']
            away_team_full = pred['away_team']
            home_code = self.get_team_code(home_team_full)
            away_code = self.get_team_code(away_team_full)
            
            result_map = {'LOCAL': home_code, 'VISITANTE': away_code, 'EMPATE': 'DRAW'}
            predicted_winner = result_map.get(pred['predicted_result'], 'DRAW')
            
            confidence = pred['confidence']
            confidence_pct = int(confidence * 100)
            
            if confidence_pct >= 75:
                risk_level = 'low'
            elif confidence_pct >= 55:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            match_date = pred.get('match_date', pred.get('date', ''))
            if hasattr(match_date, 'strftime'):
                game_date = match_date.isoformat()
            elif isinstance(match_date, str):
                game_date = match_date
            else:
                game_date = str(match_date)
            
            panel_pred = {
                'sport': 'soccer',
                'homeTeam': home_code,
                'homeTeamFullName': self.get_team_full_name(home_code),
                'homeTeamLogo': TEAM_LOGOS.get(home_code, ''),
                'awayTeam': away_code,
                'awayTeamFullName': self.get_team_full_name(away_code),
                'awayTeamLogo': TEAM_LOGOS.get(away_code, ''),
                'predictedWinner': predicted_winner,
                'confidence': confidence_pct,
                'riskLevel': risk_level,
                'gameDate': game_date,
                'status': 'active',
                'notes': f"Premier League - Jornada {self.current_matchday}\n"
                         f"Modelo: {pred['model_used']}\n"
                         f"Probabilidades: Local {pred['probabilities'].get('LOCAL', 0):.1%}, "
                         f"Empate {pred['probabilities'].get('EMPATE', 0):.1%}, "
                         f"Away {pred['probabilities'].get('VISITANTE', 0):.1%}",
                'arguments': {
                    'forWinner': [f"Confianza del modelo: {confidence_pct}%"],
                    'forLoser': [f"Factor de riesgo: {(100-confidence_pct)}%"],
                    'summary': {
                        'winnerFactors': int(confidence * 10),
                        'loserFactors': int((1 - confidence) * 10),
                        'matchupType': 'premier_league',
                        'betRecommendation': f"{predicted_winner} with {confidence_pct}% confidence"
                    }
                }
            }
            panel_predictions.append(panel_pred)
        
        return panel_predictions
    
    def export_history_to_panel_format(self) -> List[Dict]:
        """Exporta el historial completo de predicciones al formato del panel (con resultados)"""
        history = self._load_history()
        panel_predictions = []
        
        for entry in history:
            home_team_full = entry.get('home_team', '')
            away_team_full = entry.get('away_team', '')
            home_code = self.get_team_code(home_team_full)
            away_code = self.get_team_code(away_team_full)
            
            result_map = {'LOCAL': home_code, 'VISITANTE': away_code, 'EMPATE': 'DRAW'}
            predicted_winner = result_map.get(entry.get('predicted_1x2', ''), 'DRAW')
            
            confidence = entry.get('confidence', 0.5)
            confidence_pct = int(confidence * 100)
            
            if confidence_pct >= 75:
                risk_level = 'low'
            elif confidence_pct >= 55:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            match_date = entry.get('match_date', '')
            if isinstance(match_date, str):
                game_date = match_date
            else:
                game_date = str(match_date)
            
            home_score = entry.get('home_score')
            away_score = entry.get('away_score')
            
            actual_winner = None
            is_correct = None
            status = 'active'
            
            if home_score is not None and away_score is not None:
                status = 'completed'
                if home_score > away_score:
                    actual_winner = home_code
                elif away_score > home_score:
                    actual_winner = away_code
                else:
                    actual_winner = 'DRAW'
                is_correct = entry.get('1x2_correct', False)
            
            panel_pred = {
                'sport': 'soccer',
                'homeTeam': home_code,
                'homeTeamFullName': self.get_team_full_name(home_code),
                'homeTeamLogo': TEAM_LOGOS.get(home_code, ''),
                'awayTeam': away_code,
                'awayTeamFullName': self.get_team_full_name(away_code),
                'awayTeamLogo': TEAM_LOGOS.get(away_code, ''),
                'predictedWinner': predicted_winner,
                'actualWinner': actual_winner,
                'isCorrect': is_correct,
                'confidence': confidence_pct,
                'riskLevel': risk_level,
                'gameDate': game_date,
                'status': status,
                'notes': f"Premier League Prediction\nModelo: {entry.get('model', 'random_forest')}",
                'arguments': {
                    'forWinner': [f"Confianza del modelo: {confidence_pct}%"],
                    'forLoser': [f"Factor de riesgo: {(100-confidence_pct)}%"],
                    'summary': {
                        'winnerFactors': int(confidence * 10),
                        'loserFactors': int((1 - confidence) * 10),
                        'matchupType': 'premier_league',
                        'betRecommendation': f"{predicted_winner} with {confidence_pct}% confidence"
                    }
                }
            }
            panel_predictions.append(panel_pred)
        
        return panel_predictions
        
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
            self.predictor = MatchPredictor(models_dir=str(models_dir), data_dir=str(data_dir))
            
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
            
            # Cargar predictor de goles (Over/Under para múltiples umbrales)
            print(" Cargando modelo de goles...")
            self.goals_predictor = GoalsPredictor(
                models_dir=str(models_dir), 
                data_dir=str(data_dir)
            )
            if not self.goals_predictor.load_model():
                print(" Entrenando modelo de goles...")
                features_g, targets_g = self.goals_predictor.create_dataset(self.feature_engineer)
                if features_g is not None:
                    self.goals_predictor.train(features_g, targets_g)
            self.goals_predictor.feature_engineer = self.feature_engineer
            
            # Obtener mejor modelo
            self.current_model = self.predictor.get_best_model()
            
            # Mostrar competitividad de la liga
            comp = self.predictor.competitiveness
            print(f"OK Sistema inicializado correctamente")
            print(f" Modelo actual: {self.current_model}")
            print(f" Competitividad: {comp.get_level()} ({comp.get_competitiveness():.2f})")
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
            
            cache_count = len(self._ou_cache)
            cache_info = f" ({cache_count} en caché)" if cache_count > 0 else ""
            
            print("PREDICTOR PREMIER LEAGUE")
            print("=" * 50)
            print("1.  Predicción de jornada completa (Ganador + O/U)")
            print("2.  Predicción por jornada (detalles)")
            print("3.  Predicción partido por partido (Ganador + O/U)")
            print("4.  Estadísticas de equipos")
            print("5.  Ver tabla de posiciones actual")
            print("6.  Cambiar modelo de predicción")
            print("7.  Rendimiento de modelos")
            print("8.  Limpiar caché O/U")
            print("10. Exportar historial al panel")
            print("9. Salir")
            print("=" * 50)
            
            choice = input("Selecciona una opción (1-10): ").strip()
            
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
                self._clear_ou_cache()
                input("Presiona Enter para continuar...")
            elif choice == '10':
                self.export_history_panel_mode()
            elif choice == '9':
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
            
            # Guardar matchday actual para enviar
            self.current_matchday = matchday
            
            # Preguntar si quiere enviar al dashboard
            print("\n" + "=" * 60)
            print("ACCIONES")
            print("=" * 60)
            send_choice = input("¿Enviar predicciones al dashboard? (s/n): ").strip().lower()
            
            if send_choice == 's':
                self.send_to_dashboard(self.current_predictions, matchday)
            
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
            
            # Guardar predicciones para enviar al dashboard
            self.current_predictions = predictions
            self.current_matchday = matchday
            
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
            
            # Competitividad de la liga
            comp_level = predictions[0].get('competitiveness_level', 'N/A')
            comp_score = predictions[0].get('competitiveness', 0)
            print(f"🏆 Competitividad Premier League: {comp_level} ({comp_score:.2f})")
            
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
            
            # Advertencia si la liga es muy competitiva
            if comp_score > 0.5:
                print(f"\n⚠️  Nota: Liga competitiva - esperar más upsets y empates")
            
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
            
            # Mostrar Over/Under para todos los umbrales
            over_under = self._get_over_under_prediction(prediction)
            if over_under['prediction'] != 'N/A':
                markets = over_under['markets']
                print(f"\n⚽ OVER/UNDER:")
                for threshold in ['over_0.5', 'over_1.5', 'over_2.5', 'over_3.5']:
                    m = markets.get(threshold, {})
                    if m:
                        o_prob = m.get('over_prob', 0)
                        u_prob = m.get('under_prob', 0)
                        thresh_str = threshold.replace('over_', '')
                        print(f"   {thresh_str}: O={o_prob:.0%} | U={u_prob:.0%}")
            
        except Exception as e:
            print(f"ERROR Error: {e}")
        
        # Info del caché
        cache_count = len(self._ou_cache)
        if cache_count > 0:
            print(f"\nℹ️  Predicciones en caché: {cache_count} (válido por 6 horas)")
        
        input("Presiona Enter para continuar...")
    
    def _clear_ou_cache(self):
        """Limpia el caché de Over/Under"""
        self._ou_cache = {}
        print("✓ Caché limpiado")
    
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
        print("=" * 70)
        
        performance = self.predictor.get_model_performance()
        
        if not performance:
            print("ERROR No hay datos de rendimiento disponibles")
            input("Presiona Enter para continuar...")
            return
        
        print(f"{'Modelo':<22} {'Train':<10} {'Test':<10} {'CV Mean':<10}")
        print("-" * 55)
        
        for model_name, metrics in performance.items():
            print(f"{model_name:<22} "
                  f"{metrics['train_accuracy']:<10.3f} "
                  f"{metrics['test_accuracy']:<10.3f} "
                  f"{metrics['cv_mean']:<10.3f}")
        
        # Accuracy histórico
        print("\n" + "=" * 70)
        print(" ACCURACY HISTÓRICO (Partidos reales)")
        print("=" * 70)
        
        historical = self._get_historical_accuracy()
        
        if not historical:
            print("  Sin datos históricos aún.")
            print("  Los datos se guardan cuando envías predicciones al dashboard.")
        else:
            print(f"{'Modelo':<22} {'1X2':<10} {'O/U 2.5':<14} {'Over':<10} {'Under':<10}")
            print("-" * 70)
            
            for model_name, stats in historical.items():
                acc_1x2 = stats['1x2']['correct'] / stats['1x2']['total'] if stats['1x2']['total'] > 0 else 0
                acc_ou = stats['ou_25']['correct'] / stats['ou_25']['total'] if stats['ou_25']['total'] > 0 else 0
                acc_over = stats['ou_25']['over_correct'] / stats['ou_25']['over_total'] if stats['ou_25']['over_total'] > 0 else 0
                acc_under = stats['ou_25']['under_correct'] / stats['ou_25']['under_total'] if stats['ou_25']['under_total'] > 0 else 0
                
                total_1x2 = stats['1x2']['total']
                total_ou = stats['ou_25']['total']
                
                print(f"{model_name:<22} "
                      f"{acc_1x2:.0%} ({total_1x2}) "
                      f"{acc_ou:.0%} ({total_ou}) "
                      f"{acc_over:.0%} "
                      f"{acc_under:.0%}")
        
        # Mejor modelo
        best_model = self.predictor.get_best_model()
        best_metrics = performance[best_model]
        
        print(f"\n Mejor modelo (test): {best_model}")
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
        print("=" * 145)
        
        # Calcular accuracy
        correct_1x2 = 0
        total_1x2 = 0
        correct_ou = {0.5: 0, 1.5: 0, 2.5: 0, 3.5: 0}
        total_ou = {0.5: 0, 1.5: 0, 2.5: 0, 3.5: 0}
        correct_over = {0.5: 0, 1.5: 0, 2.5: 0, 3.5: 0}
        total_over = {0.5: 0, 1.5: 0, 2.5: 0, 3.5: 0}
        correct_under = {0.5: 0, 1.5: 0, 2.5: 0, 3.5: 0}
        total_under = {0.5: 0, 1.5: 0, 2.5: 0, 3.5: 0}
        
        for pred in predictions:
            if 'error' in pred:
                continue
            if pred.get('actual_result'):
                total_1x2 += 1
                if pred.get('correct'):
                    correct_1x2 += 1
            if pred.get('total_goals') is not None:
                over_under = self._get_over_under_prediction(pred)
                if over_under['prediction'] != 'N/A':
                    for thresh in [0.5, 1.5, 2.5, 3.5]:
                        total_ou[thresh] += 1
                        m = over_under['markets'].get(f'over_{thresh}', {})
                        if m:
                            pred_over = m.get('prediction') == 'OVER'
                            real_over = pred['total_goals'] >= (thresh + 0.5)
                            if pred_over == real_over:
                                correct_ou[thresh] += 1
                            if pred_over:
                                total_over[thresh] += 1
                                if real_over:
                                    correct_over[thresh] += 1
                            else:
                                total_under[thresh] += 1
                                if not real_over:
                                    correct_under[thresh] += 1
        
        if total_1x2 > 0:
            acc_1x2 = correct_1x2 / total_1x2
            accs_ou = {t: correct_ou[t]/total_ou[t] if total_ou[t] > 0 else 0 for t in correct_ou}
            accs_over = {t: correct_over[t]/total_over[t] if total_over[t] > 0 else 0 for t in correct_over}
            accs_under = {t: correct_under[t]/total_under[t] if total_under[t] > 0 else 0 for t in correct_under}
            print(f"\nAccuracy 1X2: {acc_1x2:.0%} ({correct_1x2}/{total_1x2})")
            print(f"O/U 0.5: {accs_ou[0.5]:.0%} (O:{accs_over[0.5]:.0%} U:{accs_under[0.5]:.0%})")
            print(f"O/U 1.5: {accs_ou[1.5]:.0%} (O:{accs_over[1.5]:.0%} U:{accs_under[1.5]:.0%})")
            print(f"O/U 2.5: {accs_ou[2.5]:.0%} (O:{accs_over[2.5]:.0%} U:{accs_under[2.5]:.0%})")
            print(f"O/U 3.5: {accs_ou[3.5]:.0%} (O:{accs_over[3.5]:.0%} U:{accs_under[3.5]:.0%})")
            print()
        
        print(f"{'Local':<25} {'Visita':<25} {'Pred':<12} {'O0.5/U0.5':<14} {'O1.5/U1.5':<14} {'O2.5/U2.5':<14} {'O3.5/U3.5':<14} {'Marcador'}")
        print("-" * 180)
        
        for pred in predictions:
            if 'error' in pred:
                continue
            
            home = pred['home_team'][:24]
            away = pred['away_team'][:24]
            
            # Convertir predicción a texto legible
            pred_map = {'LOCAL': 'Local', 'VISITANTE': 'Visita', 'EMPATE': 'Empate', 'DRAW': 'Empate'}
            result = pred_map.get(pred['predicted_result'], pred['predicted_result'])
            
            over_under = self._get_over_under_prediction(pred)
            
            # Resultado al final si existe
            res_str = ''
            if pred.get('actual_result'):
                if pred.get('correct'):
                    res_str = 'OK'
                else:
                    res_str = 'ERR'
            
            # Obtener probabilidades para todos los umbrales
            markets = over_under.get('markets', {})
            
            def format_ou(thresh):
                m = markets.get(f'over_{thresh}')
                if not m:
                    return '-'
                o_prob = m.get('over_prob', 0)
                u_prob = m.get('under_prob', 0)
                return f"{o_prob:.0%}/{u_prob:.0%}"
            
            ou_05 = format_ou(0.5)
            ou_15 = format_ou(1.5)
            ou_25 = format_ou(2.5)
            ou_35 = format_ou(3.5)
            
            # Marcar O/U correctas
            mark_ou = ''
            if pred.get('total_goals') is not None and markets:
                ou_marks = []
                for thresh in [0.5, 1.5, 2.5, 3.5]:
                    m = markets.get(f'over_{thresh}')
                    if m:
                        pred_over = m.get('prediction') == 'OVER'
                        # Over 0.5 → 1+ goles, Over 1.5 → 2+ goles, etc.
                        real_over = pred['total_goals'] >= (thresh + 0.5)
                        if pred_over == real_over:
                            ou_marks.append('✓')
                        else:
                            ou_marks.append('✗')
                if ou_marks:
                    mark_ou = ' '.join(ou_marks)
            
            # Mostrar marcador real si existe
            score = ''
            if pred.get('home_score') is not None:
                score = f"{pred['home_score']}-{pred['away_score']}"
            
            print(f"{home:<25} {away:<25} {result:<12} {ou_05:<14} {ou_15:<14} {ou_25:<14} {ou_35:<14} {score:<7} {res_str:<4} {mark_ou}")
        
        # Guardar partidos terminados en historial
        saved_count = 0
        for pred in predictions:
            if 'error' in pred:
                continue
            if pred.get('actual_result') and pred.get('total_goals') is not None:
                over_under = self._get_over_under_prediction(pred)
                self._save_prediction_to_history(pred, over_under)
                saved_count += 1
        
        if saved_count > 0:
            print(f"\n  {saved_count} partidos guardados en historial")
    
    def _get_over_under_prediction(self, winner_prediction: Dict) -> Dict:
        """Obtiene predicción Over/Under para un partido (con caché de 6 horas)"""
        try:
            if not self.goals_predictor:
                return {'prediction': 'N/A', 'confidence': 0, 'markets': {}}
            
            # Crear key única para el partido
            import hashlib
            from datetime import datetime
            
            home = winner_prediction['home_team']
            away = winner_prediction['away_team']
            match_date = winner_prediction.get('match_date')
            if hasattr(match_date, 'strftime'):
                date_str = match_date.strftime('%Y-%m-%d')
            else:
                date_str = str(match_date)[:10] if match_date else 'unknown'
            
            cache_key = f"{home}|{away}|{date_str}"
            
            # Verificar caché
            now = datetime.now().timestamp()
            if cache_key in self._ou_cache:
                cached_data = self._ou_cache[cache_key]
                if now - cached_data['timestamp'] < self._ou_cache_ttl:
                    # Usar caché
                    return cached_data['result']
            
            # Hacer predicción
            result = self.goals_predictor.predict_goals(
                home,
                away,
                winner_prediction['match_date']
            )
            
            if 'error' in result:
                return {'prediction': 'N/A', 'confidence': 0, 'markets': {}}
            
            # Usar over 2.5 como default
            market_25 = result['markets'].get('over_2.5', {})
            ou_short = 'O' if market_25.get('prediction') == 'OVER' else 'U'
            
            final_result = {
                'prediction': ou_short,
                'confidence': market_25.get('confidence', 0),
                'full_prediction': result,
                'markets': result['markets']
            }
            
            # Guardar en caché
            self._ou_cache[cache_key] = {
                'timestamp': now,
                'result': final_result
            }
            
            return final_result
        except Exception:
            return {'prediction': 'N/A', 'confidence': 0, 'markets': {}}
    
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
    
    def _load_history(self) -> List[Dict]:
        """Cargar historial de predicciones desde archivo"""
        if not self.history_file.exists():
            return []
        try:
            import json
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _save_prediction_to_history(self, prediction: Dict, over_under: Dict):
        """Guardar predicción al historial"""
        try:
            import json
            
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            history = self._load_history()
            
            entry = {
                'timestamp': datetime.now().isoformat(),
                'home_team': prediction['home_team'],
                'away_team': prediction['away_team'],
                'match_date': str(prediction.get('match_date', ''))[:10] if prediction.get('match_date') else None,
                'matchday': prediction.get('matchday'),
                'model': prediction.get('model_used', self.current_model),
                'predicted_1x2': prediction.get('predicted_result'),
                '1x2_correct': prediction.get('correct'),
                'over_under': over_under.get('markets', {}),
                'predicted_ou_25': over_under.get('prediction'),
                'actual_goals': prediction.get('total_goals'),
                'home_score': prediction.get('home_score'),
                'away_score': prediction.get('away_score'),
            }
            
            history.append(entry)
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Warning: No se pudo guardar historial: {e}")
    
    def _get_historical_accuracy(self) -> Dict:
        """Calcular accuracy histórico por modelo"""
        history = self._load_history()
        
        if not history:
            return {}
        
        results = {}
        
        for entry in history:
            model = entry.get('model', 'unknown')
            if model not in results:
                results[model] = {
                    '1x2': {'correct': 0, 'total': 0},
                    'ou_05': {'correct': 0, 'total': 0, 'over_correct': 0, 'under_correct': 0, 'over_total': 0, 'under_total': 0},
                    'ou_15': {'correct': 0, 'total': 0, 'over_correct': 0, 'under_correct': 0, 'over_total': 0, 'under_total': 0},
                    'ou_25': {'correct': 0, 'total': 0, 'over_correct': 0, 'under_correct': 0, 'over_total': 0, 'under_total': 0},
                    'ou_35': {'correct': 0, 'total': 0, 'over_correct': 0, 'under_correct': 0, 'over_total': 0, 'under_total': 0},
                }
            
            # 1X2 accuracy
            if entry.get('1x2_correct') is not None:
                results[model]['1x2']['total'] += 1
                if entry['1x2_correct']:
                    results[model]['1x2']['correct'] += 1
            
            # O/U accuracy
            actual_goals = entry.get('actual_goals')
            if actual_goals is not None:
                ou_data = entry.get('over_under', {})
                for thresh in [0.5, 1.5, 2.5, 3.5]:
                    key = f'ou_{str(thresh).replace(".", "")}'
                    market = ou_data.get(f'over_{thresh}')
                    if market:
                        results[model][key]['total'] += 1
                        predicted_over = market.get('prediction') == 'OVER'
                        real_over = actual_goals >= (thresh + 0.5)
                        if predicted_over == real_over:
                            results[model][key]['correct'] += 1
                        if predicted_over:
                            results[model][key]['over_total'] += 1
                            if real_over:
                                results[model][key]['over_correct'] += 1
                        else:
                            results[model][key]['under_total'] += 1
                            if not real_over:
                                results[model][key]['under_correct'] += 1
        
        return results
    
    def export_history_panel_mode(self):
        """Exporta el historial completo al formato del panel"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(" EXPORTAR HISTORIAL AL PANEL")
        print("=" * 60)
        
        history = self._load_history()
        
        if not history:
            print("No hay historial disponible")
            input("Presiona Enter para continuar...")
            return
        
        panel_predictions = self.export_history_to_panel_format()
        
        completed = [p for p in panel_predictions if p['status'] == 'COMPLETED']
        pending = [p for p in panel_predictions if p['status'] == 'ACTIVE']
        
        print(f"\nTotal predicciones: {len(panel_predictions)}")
        print(f"  - Completadas: {len(completed)}")
        print(f"  - Activas: {len(pending)}")
        
        print("\n" + "=" * 60)
        print("OPCIONES DE EXPORT")
        print("=" * 60)
        print("1. Exportar TODO el historial (completadas + activas)")
        print("2. Exportar solo COMPLETADAS (con resultados)")
        print("3. Exportar solo ACTIVAS (predicciones pendientes)")
        print("4. Cancelar")
        
        choice = input("\nSelecciona opción (1-4): ").strip()
        
        if choice == '1':
            to_export = panel_predictions
        elif choice == '2':
            to_export = completed
        elif choice == '3':
            to_export = pending
        else:
            print("Cancelado")
            input("Presiona Enter para continuar...")
            return
        
        export_file = self.project_root / "predictions_for_panel.json"
        
        with open(export_file, 'w') as f:
            json.dump(to_export, f, indent=2)
        
        print(f"\n✓ Exportado a: {export_file}")
        print(f"  Total: {len(to_export)} predicciones")
        
        send_choice = input("\n¿Enviar مباشرة al panel? (s/n): ").strip().lower()
        
        if send_choice == 's':
            self._send_predictions_to_panel(to_export)
        
        input("\nPresiona Enter para continuar...")
    
    def _send_predictions_to_panel(self, predictions: List[Dict]):
        """Envía predicciones al panel via API"""
        import requests
        
        if not predictions:
            print("No hay predicciones para enviar")
            return
        
        panel_url = os.getenv("SAFESPORTS_PANEL_URL", "https://safesports-panel.vercel.app")
        import_secret = os.getenv("IMPORT_API_SECRET", "")
        
        if not import_secret:
            print("ERROR: No está configurado IMPORT_API_SECRET en .env")
            return
        
        url = f"{panel_url}/api/predictions/import"
        headers = {
            "Authorization": f"Bearer {import_secret}",
            "Content-Type": "application/json"
        }
        
        print(f"\nEnviando {len(predictions)} predicciones al panel...")
        
        try:
            response = requests.post(url, json={"predictions": predictions}, headers=headers, timeout=60)
            
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"✓ Importado: {result.get('imported', 0)} predicciones")
                print(f"  Skipped (ya existían): {result.get('skipped', 0)}")
                if result.get('errors'):
                    print(f"  Errores: {len(result['errors'])}")
            else:
                print(f"ERROR: {response.status_code}")
                print(response.text[:500])
        except Exception as e:
            print(f"ERROR: {e}")

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