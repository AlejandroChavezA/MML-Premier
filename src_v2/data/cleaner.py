"""
Data Cleaner
===========
Limpieza de datos crudos -> datos limpios

Dependencias:
- pandas, numpy

Salida:
- data/cleaned/matches_*_cleaned.csv
- data/cleaned/standings_*_cleaned.csv
- data/cleaned/teams_cleaned.csv
"""

import pandas as pd
import numpy as np
import re
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys_path = str(PROJECT_ROOT / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)
from season_utils import get_current_season


class DataCleaner:
    """Limpia datos crudos de Premier League"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.cleaned_dir = self.data_dir / "cleaned"
        self._ensure_cleaned_dir()
    
    def _ensure_cleaned_dir(self):
        """Crear directorio cleaned si no existe"""
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_matches(self, year: int) -> pd.DataFrame:
        """Limpiar datos de partidos para una temporada"""
        input_path = self.data_dir / f"matches_{year}.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró: {input_path}")
        
        df = pd.read_csv(input_path)
        original_count = len(df)
        
        # 1. Convertir tipos
        df['date'] = pd.to_datetime(df['date'])
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        
        # 2. Eliminar invalid rows
        df = df.dropna(subset=['home_team', 'away_team', 'matchday'])
        
        # 3. Para la temporada actual: manejar partidos programados
        if year == get_current_season():
            scheduled = df[df['status'] == 'TIMED'].copy()
            scheduled['home_score'] = np.nan
            scheduled['away_score'] = np.nan
            finished = df[df['status'] == 'FINISHED'].copy()
            df = pd.concat([finished, scheduled], ignore_index=True)
        
        # 4. Validar rangos
        df = df[(df['matchday'] >= 1) & (df['matchday'] <= 38)]
        
        # 5. Columns derivadas
        df['total_goals'] = df['home_score'].fillna(0) + df['away_score'].fillna(0)
        df['goal_difference'] = df['home_score'].fillna(0) - df['away_score'].fillna(0)
        
        # 6. Resultado (solo finished)
        def get_result(row):
            if pd.isna(row['home_score']):
                return 'SCHEDULED'
            elif row['home_score'] > row['away_score']:
                return 'LOCAL'
            elif row['home_score'] < row['away_score']:
                return 'VISITANTE'
            return 'EMPATE'
        
        df['result'] = df.apply(get_result, axis=1)
        
        # Guardar
        output_path = self.cleaned_dir / f"matches_{year}_cleaned.csv"
        df.to_csv(output_path, index=False)
        
        print(f"  {year}: {original_count} → {len(df)} ({len(df[df['status']=='FINISHED'])} finished)")
        return df
    
    def clean_standings(self, year: int) -> pd.DataFrame:
        """Limpiar tabla de posiciones"""
        input_path = self.data_dir / f"standings_{year}.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró: {input_path}")
        
        df = pd.read_csv(input_path)
        
        # Calcular puntos si no cuadran
        df['points'] = df['won'] * 3 + df['draw'] * 1
        
        # Calcular GD
        df['goal_difference'] = df['goals_for'] - df['goals_against']
        
        # Metrics derivados
        df['win_rate'] = (df['won'] / df['played_games']).round(3)
        df['points_per_game'] = (df['points'] / df['played_games']).round(2)
        df['goals_for_per_game'] = (df['goals_for'] / df['played_games']).round(2)
        df['goals_against_per_game'] = (df['goals_against'] / df['played_games']).round(2)
        
        # Ordenar por posición
        df = df.sort_values('position').reset_index(drop=True)
        
        # Guardar
        output_path = self.cleaned_dir / f"standings_{year}_cleaned.csv"
        df.to_csv(output_path, index=False)
        
        print(f"  {year} standings: {len(df)} equipos")
        return df
    
    def clean_teams(self) -> pd.DataFrame:
        """Limpiar datos de equipos"""
        input_path = self.data_dir / "teams.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró: {input_path}")
        
        df = pd.read_csv(input_path)
        
        # Solo columns relevantes
        cols = ['id', 'name', 'shortName', 'tla', 'founded', 'venue', 'clubColors']
        df = df[[c for c in cols if c in df.columns]].copy()
        
        # Fill NaN
        df['founded'] = df['founded'].fillna('Unknown')
        df['clubColors'] = df['clubColors'].fillna('N/A')
        
        # Guardar
        output_path = self.cleaned_dir / "teams_cleaned.csv"
        df.to_csv(output_path, index=False)
        
        print(f"  teams: {len(df)} equipos")
        return df
    
    def run_cleaning(self, years: Optional[List[int]] = None):
        """Ejecutar limpieza completa"""
        if years is None:
            years = []
            for f in glob.glob(f"{self.data_dir}/matches_*_cleaned.csv"):
                m = re.search(r"matches_(\d{4})_cleaned\.csv", os.path.basename(f))
                if m:
                    years.append(int(m.group(1)))
            years = sorted(set(years)) or [get_current_season()]
        
        print("🧹 LIMPIEZA DE DATOS")
        print("=" * 40)
        
        for year in years:
            print(f"\n[{year}]")
            self.clean_matches(year)
            self.clean_standings(year)
        
        self.clean_teams()
        
        print("\n✅ Limpieza completa")
        print(f"  Output: {self.cleaned_dir}")


def get_cleaner(data_dir: str = "data") -> DataCleaner:
    """Factory function"""
    return DataCleaner(data_dir)


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run_cleaning()