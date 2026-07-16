"""
Módulo para calcular la competitividad de la liga.
Usa desviación estándar normalizada para medir qué tan "competitiva" es la liga.

Fórmula:
- Coeficiente de variación (CV) = std(puntos) / mean(puntos)
- Competitividad = 1 - min(CV, 1)
- 0 = liga perfectamente competitiva (sin favoritos)
- 1 = liga nada competitiva (dominada por pocos equipos)
"""

import pandas as pd
import numpy as np
import glob
import os
import re
from pathlib import Path
from typing import Dict, Optional


class LeagueCompetitiveness:
    def __init__(self, data_dir: str = "data/cleaned"):
        self.data_dir = data_dir
        self.competitiveness = None
        self.metrics = {}
        self._calculate()
    
    def _calculate(self):
        """Calcula la competitividad de la liga"""
        
        # Cargar standings de todas las temporadas disponibles
        all_standings = []
        for f in glob.glob(f"{self.data_dir}/standings_*_cleaned.csv"):
            m = re.search(r"standings_(\d{4})_cleaned\.csv", os.path.basename(f))
            if m:
                year = int(m.group(1))
                df = pd.read_csv(f)
                df['season'] = year
                all_standings.append(df)
        
        if not all_standings:
            #默认值 si no hay datos
            self.competitiveness = 0.5
            self.metrics = {'error': 'No se encontraron datos'}
            return
        
        combined_df = pd.concat(all_standings, ignore_index=True)
        
        # Calcular por temporada
        by_season = {}
        for season in combined_df['season'].unique():
            season_df = combined_df[combined_df['season'] == season]
            points = season_df['points'].values
            
            mean_pts = np.mean(points)
            std_pts = np.std(points)
            cv = std_pts / mean_pts if mean_pts > 0 else 0
            
            # Más bajo CV = más igualada = más competitiva
            # CV típico en ligas: 0.25-0.45
            # Normalizar: invertimos para que mayor = más competitivo
            # CV bajo (0.25) -> competitividad alta (0.8)
            # CV alto (0.45) -> competitividad baja (0.2)
            competitive_score = max(0, 1 - (cv - 0.25) / 0.2)
            
            by_season[season] = {
                'mean': mean_pts,
                'std': std_pts,
                'cv': cv,
                'competitive_score': competitive_score
            }
        
        # Competitividad global (promedio)
        all_points = combined_df['points'].values
        global_mean = np.mean(all_points)
        global_std = np.std(all_points)
        global_cv = global_std / global_mean if global_mean > 0 else 0
        
        # Competitividad global: invertimos CV
        # Mayor CV relativo = menor competitividad
        self.competitiveness = max(0, min(1, 1 - (global_cv - 0.25) / 0.2))
        
        self.metrics = {
            'global_cv': global_cv,
            'global_std': global_std,
            'global_mean': global_mean,
            'by_season': by_season,
            'is_highly_competitive': self.competitiveness > 0.5
        }
    
    def get_competitiveness(self) -> float:
        """Retorna la competitividad (0-1)"""
        return self.competitiveness if self.competitiveness is not None else 0.5
    
    def get_level(self) -> str:
        """Retorna el nivel de competitividad como texto"""
        comp = self.get_competitiveness()
        
        if comp > 0.6:
            return "MUY COMPETITIVA"
        elif comp > 0.4:
            return "COMPETITIVA"
        elif comp > 0.2:
            return "MODERADAMENTE COMPETITIVA"
        else:
            return "POCO COMPETITIVA"
    
    def get_adjustment_factors(self) -> Dict[str, float]:
        """
        Retorna factores de ajuste para las predicciones.
        
        En ligas muy competitivas:
        - Reducir confianza en favoritos
        - Aumentar peso del empate
        - Considerar más upsets
        """
        comp = self.get_competitiveness()
        
        # Factor de reducción de confianza
        # En liga poco competitiva (comp bajo): confianza alta (factor alto)
        # En liga muy competitiva (comp alto): confianza baja (factor bajo)
        confidence_factor = 0.7 + (comp * 0.3)  # 0.7 a 1.0
        
        # Factor de ajuste para empate (más competitivo = más peso al empate)
        draw_factor = 1.0 + (comp * 0.4)  # 1.0 a 1.4
        
        # Factor de upset (más competitivo = más probable el upset)
        upset_factor = 1.0 + (comp * 0.5)  # 1.0 a 1.5
        
        return {
            'confidence_factor': confidence_factor,
            'draw_factor': draw_factor,
            'upset_factor': upset_factor,
            'description': self.get_level()
        }
    
    def adjust_prediction(self, probabilities: Dict[str, float], confidence: float) -> Dict:
        """
        Ajusta una predicción basada en la competitividad de la liga.
        
        Args:
            probabilities: Dict con 'LOCAL', 'VISITANTE', 'EMPATE'
            confidence: Confianza original del modelo (0-1)
        
        Returns:
            Predicción ajustada con factores de competitividad
        """
        factors = self.get_adjustment_factors()
        comp = self.get_competitiveness()
        
        # Copiar probabilidades
        adjusted_probs = probabilities.copy()
        
        if comp > 0.5:  # Liga competitiva
            # Reducir diferencia entre favorito y no favorito
            # Aumentar probabilidad de empate
            
            # Guardar probabilidades originales
            original_probs = adjusted_probs.copy()
            
            # Promedio hacia el empate
            draw_weight = factors['draw_factor']
            for key in ['LOCAL', 'VISITANTE']:
                # Reducir favorito, aumentar no favorito hacia empate
                diff_from_draw = abs(original_probs[key] - original_probs['EMPATE'])
                adjustment = diff_from_draw * (1 - comp) * 0.2
                adjusted_probs[key] = max(0.1, adjusted_probs[key] - adjustment)
                adjusted_probs['EMPATE'] = min(0.6, adjusted_probs['EMPATE'] + adjustment * 0.5)
            
            # Renormalizar
            total = sum(adjusted_probs.values())
            adjusted_probs = {k: v/total for k, v in adjusted_probs.values()}
            
            # Recalcular confianza
            # En liga competitiva, la confianza debe ser menor
            adjusted_confidence = confidence * factors['confidence_factor']
            adjusted_confidence = max(0.3, min(0.95, adjusted_confidence))
            
        else:  # Liga poco competitiva
            # Mantener predicciones originales
            adjusted_confidence = confidence
        
        # Determinar nuevo favorito
        new_favorite = max(adjusted_probs, key=adjusted_probs.get)
        
        return {
            'adjusted_probabilities': adjusted_probs,
            'adjusted_confidence': adjusted_confidence,
            'original_confidence': confidence,
            'competitiveness': comp,
            'competitiveness_level': self.get_level(),
            'adjustment_factors': factors,
            'predicted_result': new_favorite,
            'upset_risk': 'HIGH' if comp > 0.5 else 'LOW'
        }
    
    def print_summary(self):
        """Imprime resumen de competitividad"""
        print(f"\n📊 COMPETITIVIDAD DE LA LIGA")
        print(f"{'='*40}")
        print(f"  Nivel: {self.get_level()}")
        print(f"  Score: {self.get_competitiveness():.3f}")
        
        if 'by_season' in self.metrics:
            print(f"\n  Por temporada:")
            for season, data in sorted(self.metrics['by_season'].items()):
                print(f"    {season}: CV={data['cv']:.3f}, Score={data['competitive_score']:.3f}")
        
        factors = self.get_adjustment_factors()
        print(f"\n  Factores de ajuste:")
        print(f"    Confianza: {factors['confidence_factor']:.2f}")
        print(f"    Empate: {factors['draw_factor']:.2f}")
        print(f"    Upset: {factors['upset_factor']:.2f}")


# Función helper para usar directamente
def get_competitiveness(data_dir: str = "data/cleaned") -> LeagueCompetitiveness:
    """Obtiene el objeto de competitividad"""
    return LeagueCompetitiveness(data_dir)
