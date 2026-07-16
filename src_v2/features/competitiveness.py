"""
League Competitiveness
====================
Mide qué tan competitiva es la liga y ajusta predicciones.

Dependencias:
- pandas, numpy

Usa:
- data.cleaned/standings_*_cleaned.csv
"""

import pandas as pd
import numpy as np
import glob
import os
import re
from pathlib import Path
from typing import Dict


class LeagueCompetitiveness:
    """Mide competitividad de la liga"""
    
    def __init__(self, data_dir: str = "data/cleaned"):
        self.data_dir = Path(data_dir)
        self.score: float = 0.5
        self.level: str = "MEDIUM"
        self.metrics: Dict = {}
        self._calculate()
    
    def _calculate(self):
        """Calcular competitividad desde standings"""
        all_standings = []
        
        for f in glob.glob(str(self.data_dir / "standings_*_cleaned.csv")):
            m = re.search(r"standings_(\d{4})_cleaned\.csv", os.path.basename(f))
            if m:
                year = int(m.group(1))
                df = pd.read_csv(f)
                df['season'] = year
                all_standings.append(df)
        
        if not all_standings:
            self.score = 0.5
            return
        
        combined = pd.concat(all_standings, ignore_index=True)
        points = combined['points'].values
        
        mean_pts = np.mean(points)
        std_pts = np.std(points)
        
        # Coeficiente de variación
        cv = std_pts / mean_pts if mean_pts > 0 else 0
        
        # Normalizar: CV típico 0.25-0.45
        # Invertir: bajo CV = alta competitividad
        self.score = max(0, min(1, 1 - (cv - 0.25) / 0.2))
        
        self.metrics = {
            'mean': mean_pts,
            'std': std_pts,
            'cv': cv,
        }
        
        if self.score > 0.6:
            self.level = "HIGH"
        elif self.score > 0.4:
            self.level = "MEDIUM"
        else:
            self.level = "LOW"
    
    def get_score(self) -> float:
        """Retorna score 0-1"""
        return self.score
    
    def get_level(self) -> str:
        """Retorna nivel como string"""
        return self.level
    
    def get_adjustment_factors(self) -> Dict:
        """Factores para ajustar predicciones"""
        return {
            'confidence': 0.7 + self.score * 0.3,  # 0.7-1.0
            'draw_weight': 1.0 + self.score * 0.4,    # 1.0-1.4
            'upset_risk': 1.0 + self.score * 0.5,    # 1.0-1.5
        }
    
    def adjust_probabilities(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Ajustar probabilidades según competitividad"""
        factors = self.get_adjustment_factors()
        base = probs.copy()
        
        if self.score > 0.5:  # Liga competitiva
            # Reducir favorito, aumentar draw
            for key in ['LOCAL', 'VISITANTE']:
                diff = abs(base[key] - base.get('EMPATE', 0.25))
                base[key] = max(0.1, base[key] - diff * (1 - self.score) * 0.2)
            
            base['EMPATE'] = min(0.6, base.get('EMPATE', 0.25) + 0.05)
            
            # Renormalize
            total = sum(base.values())
            base = {k: v/total for k, v in base.items()}
        
        return base
    
    def print_summary(self):
        """Imprimir resumen"""
        print(f"\n📊 Competitiveness: {self.level} ({self.score:.2f})")
        print(f"  Mean: {self.metrics.get('mean', 0):.1f} pts, Std: {self.metrics.get('std', 0):.1f}")
        print(f"  CV: {self.metrics.get('cv', 0):.3f}")


def get_competitiveness(data_dir: str = "data/cleaned") -> LeagueCompetitiveness:
    """Factory function"""
    return LeagueCompetitiveness(data_dir)