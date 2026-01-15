#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Añadir directorio src al path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def test_data_access():
    """Probar acceso a los datos"""
    print("🔍 Probando acceso a datos...")
    
    # Verificar archivos
    data_dir = project_root / "data" / "cleaned"
    
    files_to_check = [
        "matches_2023_cleaned.csv",
        "matches_2024_cleaned.csv", 
        "matches_2025_cleaned.csv",
        "teams_cleaned.csv",
        "standings_2025_cleaned.csv"
    ]
    
    print(f"📁 Buscando archivos en: {data_dir}")
    
    for filename in files_to_check:
        file_path = data_dir / filename
        if file_path.exists():
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename}")
    
    # Probar cargar un archivo con pandas
    try:
        import pandas as pd
        
        matches_file = data_dir / "matches_2023_cleaned.csv"
        if matches_file.exists():
            df = pd.read_csv(matches_file)
            print(f"\n📊Archivo cargado: {len(df)} partidos")
            print(f"Columnas: {list(df.columns)}")
            return True
        else:
            print("❌ No existe el archivo de prueba")
            return False
            
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return False

if __name__ == "__main__":
    test_data_access()