#!/usr/bin/env python3

import sys
from pathlib import Path

# Añadir directorio src al path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def train_advanced_models():
    """Entrenar modelos avanzados con features CLAVE"""
    print("ENTRENANDO MODELOS AVANZADOS CON FEATURES CLAVE")
    print("=" * 60)
    
    try:
        # Importar clases avanzadas
        from advanced_feature_engineering import AdvancedFeatureEngineer
        from advanced_prediction_models import AdvancedMatchPredictor
        
        # Inicializar feature engineering avanzado
        print("Inicializando feature engineering AVANZADO...")
        fe = AdvancedFeatureEngineer(data_dir="data/cleaned")
        
        if not fe.load_data():
            print("Error cargando datos")
            return False
        
        print("Creando dataset AVANZADO de entrenamiento...")
        features_df, targets_df = fe.create_advanced_training_dataset()
        
        print(f"Dataset AVANZADO creado:")
        print(f"  Partidos: {features_df.shape[0]}")
        print(f"  Características totales: {features_df.shape[1]}")
        
        # Mostrar features clave
        key_features = [col for col in features_df.columns if any(x in col for x in [
            'points_diff', 'gd_diff', 'position_diff',
            'last5_points', 'last5_gd', 'form_diff',
            'home_advantage', 'rest_days', 'fatigue'
        ])]
        
        print(f"  Features CLAVE: {len(key_features)}")
        print("  Features CLAVE implementadas:")
        for feature in sorted(key_features):
            print(f"    - {feature}")
        
        # Entrenar modelos avanzados
        print("\nEntrenando modelos AVANZADOS...")
        predictor = AdvancedMatchPredictor(models_dir="models")
        predictor.feature_engineer = fe
        
        if predictor.train_advanced_models(features_df, targets_df):
            print("\nENTRENAMIENTO AVANZADO COMPLETADO!")
            print("Models guardados en 'models/' con sufijo '_advanced'")
            
            # Mostrar rendimiento esperado
            best_model = predictor.get_best_advanced_model()
            performance = predictor.get_advanced_model_performance()
            best_acc = performance[best_model]['test_accuracy']
            
            print(f"\nRESULTADOS ESPERADOS:")
            print(f"  Modelo base (anterior): ~53.6% accuracy")
            print(f"  Modelo avanzado ({best_model}): {best_acc:.1%} accuracy")
            print(f"  Mejora esperada: +{(best_acc - 0.536)*100:.1f}%")
            
            print(f"\nPara usar el sistema avanzado:")
            print("  1. python3 main_advanced.py")
            print("  2. Seleccionar modelos '_advanced'")
            
            return True
        else:
            print("Error en entrenamiento avanzado")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_advanced_models()
    
    if success:
        print("\nENTRENAMIENTO AVANZADO COMPLETADO CON EXITO!")
        print("Features CLAVE implementadas:")
        print("  🔥 Diferencias entre equipos (points_diff, gd_diff, position_diff)")
        print("  📈 Forma reciente (last5_points, last5_gd, form_diff)")
        print("  🏠 Home advantage real (home_win_rate_home vs away)")
        print("  💤 Días de descanso (rest_days, fatigue)")
        print("  🏆 Racha de invicto y estadísticas secundarias")
    else:
        print("\nERROR EN ENTRENAMIENTO AVANZADO")