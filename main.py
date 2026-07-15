#!/usr/bin/env python3
"""
MML-Premier: Premier League Match Prediction System

Sistema completo de predicción de partidos de la Premier League
utilizando Machine Learning y estadísticas avanzadas.

Author: OpenCode Assistant
Date: 2025-01-15
"""

import sys
import os
from pathlib import Path

# Añadir directorio src al path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def update_data_if_needed(force: bool = False):
    """Actualizar datos automáticamente desde la API.

    Por defecto respeta el caché de 6 horas, pero se puede forzar
    la actualización al iniciar el programa.
    """
    import subprocess
    import json
    import pandas as pd
    from datetime import datetime, timedelta
    
    cache_file = project_root / ".update_cache.json"
    update_interval = timedelta(hours=6)
    
    # Snapshot de los datos actuales para validar que la actualización
    # realmente incorporó datos nuevos (no se quede atorada).
    def _snapshot():
        try:
            df = pd.read_csv(project_root / "data" / "cleaned" / "matches_2025_cleaned.csv")
            df['status'] = df['status'].astype(str).str.upper()
            finished = df[df['status'] == 'FINISHED']
            return {
                'finished': int(len(finished)),
                'max_matchday': int(finished['matchday'].max()) if len(finished) else 0,
            }
        except Exception:
            return {'finished': 0, 'max_matchday': 0}
    
    # Verificar si necesita actualización
    if not force and cache_file.exists():
        try:
            with open(cache_file) as f:
                cache_data = json.load(f)
            last_update = datetime.fromisoformat(cache_data['last_update'])
            if datetime.now() - last_update < update_interval:
                print(f" Datos actualizados hace {datetime.now() - last_update}")
                print(f" Siguiente actualización en {update_interval - (datetime.now() - last_update)}")
                return True
        except:
            pass
    
    print(" Actualizando datos desde API...")
    update_script = project_root / "update_premier_data.py"
    
    if update_script.exists():
        before = _snapshot()
        result = subprocess.run([sys.executable, str(update_script)], 
                              capture_output=True, text=True, cwd=project_root)
        
        if result.returncode != 0:
            print(f" ERROR: La actualización de datos falló.")
            print(f" Salida: {result.stdout[-500:]}")
            print(f" Error:  {result.stderr[-500:]}")
            print(f" Se usarán los datos existentes (posiblemente desactualizados).")
            return False
        
        # Limpiar datos después de actualizar
        print(" Limpiando datos actualizados...")
        cleaning_result = subprocess.run(
            [sys.executable, "-c", """
import sys
sys.path.insert(0, 'src')
from data_cleaning import PremierLeagueDataCleaner
cleaner = PremierLeagueDataCleaner(data_dir='data')
cleaner.run_full_cleaning()
"""], capture_output=True, text=True, cwd=project_root)
        
        if cleaning_result.returncode != 0:
            print(f" ERROR: La limpieza de datos falló: {cleaning_result.stderr[:300]}")
            return False
        
        after = _snapshot()
        if after['finished'] < before['finished']:
            print(f" ERROR: La actualización REDUJO los partidos finalizados "
                  f"({before['finished']} -> {after['finished']}). "
                  f"Revisa la API o el caché.")
            return False
        if after['max_matchday'] < before['max_matchday']:
            print(f" ADVERTENCIA: La jornada finalizada máxima bajó "
                  f"({before['max_matchday']} -> {after['max_matchday']}).")
        
        # Guardar timestamp de actualización junto con stats para futuras validaciones
        with open(cache_file, 'w') as f:
            json.dump({'last_update': datetime.now().isoformat(),
                       'finished': after['finished'],
                       'max_matchday': after['max_matchday']}, f)
        print(f" Datos actualizados y limpios exitosamente "
              f"({after['finished']} finalizados, jornada {after['max_matchday']})")
        return True
    else:
        print(" Script de actualización no encontrado - usando datos existentes")
        return True

def check_environment():
    """Verificar que el entorno esté configurado correctamente"""
    print(" Verificando entorno...")
    
    # Verificar directorios necesarios
    required_dirs = [
        "data",
        "data/cleaned", 
        "src",
        "models"
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            if dir_name in ["models"]:
                print(f" Creando directorio: {dir_name}")
                dir_path.mkdir(exist_ok=True)
            else:
                print(f" Falta directorio requerido: {dir_name}")
                return False
    
    # Verificar archivos de datos limpios
    required_files = [
        "data/cleaned/teams_cleaned.csv",
        "data/cleaned/matches_2023_cleaned.csv",
        "data/cleaned/matches_2024_cleaned.csv", 
        "data/cleaned/matches_2025_cleaned.csv",
        "data/cleaned/standings_2025_cleaned.csv"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = project_root / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(" Faltan archivos de datos:")
        for file_name in missing_files:
            print(f"   • {file_name}")
        print("\n Ejecuta primero el proceso de limpieza de datos:")
        print("   python src/data_cleaning.py")
        return False
    
    print(" Entorno verificado correctamente")
    return True

def display_welcome():
    """Mostrar mensaje de bienvenida"""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("""
 P R E D I C T O R   P R E M I E R   L E A G U E 
=====================================================

Sistema de predicción de partidos de la Premier League
Utilizando Machine Learning y estadísticas avanzadas
Basado en datos históricos de 3 temporadas (2023-2025)

Características:
•  Predicciones de jornada completa
•  Predicciones partido por partido  
•  Estadísticas detalladas de equipos
•  Tabla de posiciones actual
•  Múltiples modelos de ML (Random Forest, XGBoost, LR)

=====================================================
    """)

def display_help():
    """Mostrar ayuda del sistema"""
    print("""
 PREDICTOR PREMIER LEAGUE - AYUDA
  ===================================
 
 Uso:
   python main.py [opciones]
 
  Opciones:
    --help, -h     Mostrar esta ayuda
    --train        Entrenar modelos con datos actuales
    --jornada      Iniciar predictor por jornada detallado
    --v2           Usar sistema v2 reorganizado (más limpio)
 
 Ejemplos:
   python main.py           Iniciar menú interactivo
   python main.py --train   Entrenar modelos
   python main.py --jornada Predictor por jornada detallado

 Menú interactivo:
   1. Predicción de jornada completa
   2. Predicción por jornada (detalles) ← incluye envío al dashboard
   3. Predicción partido por partido
   4. Estadísticas de equipos
   5. Ver tabla de posiciones actual
   6. Cambiar modelo de predicción
   7. Rendimiento de modelos
   8. Salir
 """)

def run_v2():
    """Usar el sistema v2 reorganizado"""
    import sys
    from pathlib import Path
    from datetime import datetime
    
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        from src_v2.features.feature_engineer import get_feature_engineer
        from src_v2.models.winner_predictor import get_winner_predictor
        from src_v2.models.goals_predictor import get_goals_predictor
    except ImportError as e:
        print(f"Error: No se encontró src_v2. {e}")
        return False
    
    print("\n⚽ SISTEMA v2 (REORGANIZADO)")
    print("=" * 50)
    
    # Load
    print("\n[1] Cargando datos...")
    fe = get_feature_engineer("data/cleaned")
    fe.load_data()
    
    print("[2] Cargando modelos...")
    winner = get_winner_predictor("models")
    if not winner.load():
        print("❌ Modelos no encontrados. Ejecuta: python src_v2/train.py")
        return False
    
    goals = get_goals_predictor("models")
    goals.load()
    
    # Mostrar rendimiento
    perf = winner.get_performance()
    if perf:
        print(f"\n  Rendimiento: Train={perf['train']:.0%} Test={perf['test']:.0%} CV={perf['cv_mean']:.0%}")
    else:
        print("\n  Rendimiento: (modelo cargado sin métricas guardadas)")
    
    # Predicción interactiva
    print("\n[3] Predicción interactiva")
    print("  Formato: HOME vs AWAY (ej: Arsenal FC vs Liverpool FC)")
    print("  Escribe 's' para salir, 'j' para jornada, 't' para test")
    
    while True:
        try:
            choice = input("\nEquipo local: ").strip()
            if choice.lower() == 's':
                break
            if choice.lower() == 'j':
                j = input("Jornada: ")
                if j.isdigit():
                    from datetime import datetime
                    matches = fe.matches_2025[fe.matches_2025['matchday'] == int(j)]
                    for _, m in matches[matches['status'] == 'TIMED'].iterrows():
                        pred = winner.predict(m['home_team'], m['away_team'], m['date'], fe)
                        if 'error' not in pred:
                            c = pred['confidence']
                            mkr = "🔥" if c > 0.5 else "  "
                            print(f"  {mkr} {pred['predicted']:10} ({c:.0%}) | {m['home_team']} vs {m['away_team']}")
                continue
            if choice.lower() == 't':
                import json
                h = project_root / "data" / "prediction_history.json"
                if h.exists():
                    with open(h) as f:
                        history = json.load(f)
                    recent = sorted(history, key=lambda x: x['match_date'])[-10:]
                    correct = sum(1 for p in recent if p.get('1x2_correct'))
                    print(f"  Últimas 10: {correct}/10 ({correct*10}%)")
                continue
            
            home_team = choice
            away_team = input("Equipo visitante: ").strip()
            if not away_team:
                continue
            
            pred = winner.predict(home_team, away_team, datetime.now(), fe)
            if 'error' in pred:
                print(f"  {pred['error']}")
                continue
            
            print(f"  → {pred['predicted']} ({pred['confidence']:.0%})")
            for k, v in pred['probabilities'].items():
                print(f"    {k}: {v:.0%}")
            
            # Goals
            g = goals.predict(home_team, away_team, datetime.now(), fe)
            if 'error' not in g:
                print(f"  Goles: {g['expected_goals']:.1f}")
                ou = g['markets']['over_2.5']
                print(f"  O/U 2.5: {ou['prediction']} ({ou['over_prob']:.0%})")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ Listo")
    return True


def run_jornada_detailed():
    """Ejecutar predictor de jornada detallado (usa menu interactivo)"""
    from menu_interface import PredictionMenu
    
    try:
        menu = PredictionMenu()
        if not menu.initialize():
            print("Error inicializando sistema")
            return False
        
        menu.jornada_detailed_mode()
        return True
    except KeyboardInterrupt:
        print("\n Predictor interrumpido")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def train_models():
    """Entrenar modelos desde cero"""
    print(" ENTRENANDO MODELOS DESDE CERO")
    print("=" * 50)
    
    try:
        # Importar clases necesarias
        from feature_engineering import FeatureEngineer
        from prediction_models import MatchPredictor
        
        # Inicializar componentes con rutas absolutas
        print(" Inicializando feature engineering...")
        fe = FeatureEngineer(data_dir="data/cleaned")
        
        if not fe.load_data():
            print("Error cargando datos")
            return False
        
        print(" Creando dataset de entrenamiento...")
        features_df, targets_df = fe.create_training_dataset()
        
        print(f" Dataset creado: {features_df.shape[0]} partidos, {features_df.shape[1]} características")
        
        # Entrenar modelos
        print(" Entrenando modelos de ML...")
        predictor = MatchPredictor()
        predictor.feature_engineer = fe
        
        if predictor.train_models(features_df, targets_df):
            print(" Entrenamiento completado exitosamente")
            print("\n Modelos guardados en directorio 'models/'")
            print("Ahora puedes iniciar el menú interactivo")
            return True
        else:
            print(" Error en el entrenamiento")
            return False
            
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_interactive_menu():
    """Iniciar menú interactivo"""
    try:
        from menu_interface import PredictionMenu
        
        menu = PredictionMenu()
        
        if not menu.initialize():
            print(" No se pudo inicializar el sistema")
            print("\nIntenta entrenar los modelos primero:")
            print("   python main.py --train")
            return False
        
        menu.display_main_menu()
        return True
        
    except Exception as e:
        print(f" Error iniciando menú: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    # Parsear argumentos de línea de comandos
    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        display_help()
        return
    
    if "--train" in args:
        if not check_environment():
            return
        train_models()
        return
    
    if "--jornada" in args:
        run_jornada_detailed()
        return
    
    if "--v2" in args:
        run_v2()
        return
    
    # Ejecución normal
# display_welcome()  # Omitido para limpiar salida
    
    # Actualizar datos automáticamente
    update_data_if_needed(force=True)
    
    # Verificar entorno
    if not check_environment():
        print("\n El entorno no está configurado correctamente")
        print("Por favor, ejecuta los scripts de preparación de datos primero")
        return
    
    # Verificar si existen modelos entrenados
    models_dir = project_root / "models"
    has_models = any(models_dir.glob("*.pkl"))
    
    if not has_models:
        print(" No se encontraron modelos entrenados")
        print("Iniciando entrenamiento automático...")
        
        if not train_models():
            print(" No se pudo completar el entrenamiento")
            return
    
    # Iniciar menú interactivo
    start_interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n ¡Gracias por usar el Predictor Premier League!")
    except Exception as e:
        print(f"\n Error inesperado: {e}")
        import traceback
        traceback.print_exc()
