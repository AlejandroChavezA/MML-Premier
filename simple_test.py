#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Añadir directorio src al path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def simple_test():
    """Prueba simple del sistema"""
    print("🧪 PRUEBA SIMPLE DEL SISTEMA")
    print("=" * 40)
    
    try:
        # Importar clases
        from feature_engineering import FeatureEngineer
        from prediction_models import MatchPredictor
        
        print("✅ Clases importadas correctamente")
        
        # Inicializar feature engineering
        print("\n📊 Inicializando feature engineering...")
        fe = FeatureEngineer(data_dir="data/cleaned")
        
        if fe.load_data():
            print("✅ Datos cargados correctamente")
            
            # Probar características para un partido
            print("\n🔮 Creando características para partido de prueba...")
            features = fe.create_match_features(
                "Arsenal FC",
                "Manchester City FC",
                pd.Timestamp.now().normalize()
            )
            
            print(f"✅ Características creadas: {len(features)} features")
            
            # Mostrar algunas características
            key_features = ['home_form_win_rate', 'away_form_win_rate', 'h2h_home_win_rate']
            for feature in key_features:
                if feature in features:
                    print(f"  {feature}: {features[feature]:.3f}")
            
            # Probar dataset de entrenamiento
            print("\n📈 Creando dataset de entrenamiento...")
            features_df, targets_df = fe.create_training_dataset()
            
            print(f"✅ Dataset creado:")
            print(f"  Partidos: {len(features_df)}")
            print(f"  Características: {len(features_df.columns)}")
            print(f"  Distribución de resultados: {targets_df.value_counts().to_dict()}")
            
            print("\n🎉 ¡Sistema funcionando correctamente!")
            return True
            
        else:
            print("❌ Error cargando datos")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import pandas as pd
    simple_test()