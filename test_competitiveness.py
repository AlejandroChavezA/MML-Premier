#!/usr/bin/env python3
"""
Script de prueba para medir competitividad de la Premier League.
Usa desviación estándar normalizada para calcular qué tan "competitiva" es la liga.

Fórmula:
- Desviación estándar de puntos normalizada
- 0 = liga perfectamente competitiva (sin favoritos)
- 1 = liga nada competitiva (dominada por pocos equipos)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent

def load_standings_data() -> pd.DataFrame:
    """Carga datos de clasificaciones de Premier League"""
    data_dir = project_root / "data" / "cleaned"
    
    # Cargar todos los años disponibles
    all_data = []
    for year in [2023, 2024, 2025]:
        file_path = data_dir / f"standings_{year}_cleaned.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['season'] = year
            all_data.append(df)
    
    if not all_data:
        print("Error: No se encontraron datos de standings")
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def calculate_competitiveness(df: pd.DataFrame) -> Dict:
    """
    Calcula la competitividad de la liga usando desviación estándar normalizada.
    
    Fórmula:
    1. Calcular desviación estándar de puntos por equipo
    2. Normalizar dividiendo entre el máximo teórico posible
    
    Returns:
        Dict con métricas de competitividad
    """
    
    # Obtener puntos por equipo por temporada
    # Si no tenemos 'team' como columna, mostrar disponibles
    print(f"Columnas disponibles: {df.columns.tolist()}")
    
    # Intentar encontrar columna de equipo
    team_col = None
    points_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['team', 'name', 'equipo', 'club']:
            team_col = col
        if col_lower in ['points', 'pts', 'puntos']:
            points_col = col
    
    if not team_col or not points_col:
        print("No se encontraron columnas de equipo/puntos")
        return {}
    
    print(f"Usando: {team_col} y {points_col}")
    
    # Calcular por temporada
    competitiveness_by_season = {}
    
    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        points = season_df[points_col].values
        
        # Desviación estándar de puntos
        std_points = np.std(points)
        
        # Máximo teórico: 38 partidos * 3 puntos = 114
        # Mínimo teórico: 38 partidos * 0 puntos = 0
        # Rango = 114
        
        # Normalizar: std / (std_max_teorico)
        # Std máximo teórico sería si todos los puntos fueran o 0 o 114
        # En la práctica, normalizamos respecto a un valor realista
        
        # Otra approach: normalizar con la media
        mean_points = np.mean(points)
        
        # Coeficiente de variación (CV) - más bajo = más igualada
        cv = std_points / mean_points if mean_points > 0 else 0
        
        # Desviación estándar normalizada (0-1)
        # Más bajo CV = más competitiva
        normalized_std = cv / 2  # Dividir por 2 para escalar a ~0-1
        
        competitiveness_by_season[season] = {
            'std_points': std_points,
            'mean_points': mean_points,
            'cv': cv,
            'normalized': normalized_std,
            'points_range': (min(points), max(points)),
            'teams_count': len(points)
        }
    
    return competitiveness_by_season


def calculate_league_competitiveness(standings_df: pd.DataFrame, matches_df: pd.DataFrame) -> Dict:
    """
    Calcula competitividad usando múltiples métricas.
    """
    
    metrics = {}
    
    # 1. Desviación estándar de puntos
    points_col = None
    for col in standings_df.columns:
        if col.lower() in ['points', 'pts', 'puntos']:
            points_col = col
    
    if points_col:
        all_points = standings_df[points_col].values
        std_points = np.std(all_points)
        mean_points = np.mean(all_points)
        
        # Competitividad basada en distribución de puntos
        # Más igualada = menor std relativo
        cv = std_points / mean_points if mean_points > 0 else 0
        
        metrics['points_std'] = std_points
        metrics['points_cv'] = cv
        metrics['competitiveness'] = 1 - min(cv, 1)  # Invertir: bajo CV = alta competitividad
    
    # 2. Upsets en partidos (visitante gana o empate contra favorito)
    if 'home_goals' in matches_df.columns and 'away_goals' in matches_df.columns:
        total_matches = len(matches_df)
        
        # Calcular quién era favorito basado en posición en tabla
        # (simplificado: visitante gana o empata = potencial upset)
        upsets = len(matches_df[
            (matches_df['away_goals'] > matches_df['home_goals']) |
            (matches_df['home_goals'] == matches_df['away_goals'])
        ])
        
        upset_rate = upsets / total_matches if total_matches > 0 else 0
        
        metrics['upset_rate'] = upset_rate
        metrics['upset_percentage'] = upset_rate * 100
    
    # 3. Diferencia de goles
    if 'home_goals' in matches_df.columns and 'away_goals' in matches_df.columns:
        goal_diff = abs(matches_df['home_goals'] - matches_df['away_goals'])
        avg_goal_diff = np.mean(goal_diff)
        
        metrics['avg_goal_difference'] = avg_goal_diff
        metrics['goal_diff_competitiveness'] = 1 - min(avg_goal_diff / 3, 1)  # Normalizar
    
    return metrics


def print_results(metrics: Dict, competitiveness_by_season: Dict):
    """Muestra los resultados del análisis"""
    
    print("\n" + "="*60)
    print("ANÁLISIS DE COMPETITIVIDAD - PREMIER LEAGUE")
    print("="*60)
    
    # Por temporada
    print("\n📊 POR TEMPORADA:")
    print("-"*60)
    for season, data in sorted(competitiveness_by_season.items()):
        print(f"\nTemporada {season}:")
        print(f"  Equipos: {data['teams_count']}")
        print(f"  Puntos: {data['points_range'][0]} - {data['points_range'][1]}")
        print(f"  Media: {data['mean_points']:.1f}")
        print(f"  Std: {data['std_points']:.2f}")
        print(f"  CV: {data['cv']:.3f}")
        print(f"  Competitividad: {data['normalized']:.3f}")
    
    # Métricas combinadas
    print("\n" + "="*60)
    print("📈 MÉTRICAS GLOBALES:")
    print("-"*60)
    
    if 'points_cv' in metrics:
        print(f"  Coeficiente de variación: {metrics['points_cv']:.3f}")
        print(f"  Competitividad (0-1): {metrics['competitiveness']:.3f}")
    
    if 'upset_rate' in metrics:
        print(f"  Tasa de upsets: {metrics['upset_percentage']:.1f}%")
    
    if 'avg_goal_difference' in metrics:
        print(f"  Diferencia de goles promedio: {metrics['avg_goal_difference']:.2f}")
        print(f"  Competitividad por goles: {metrics['goal_diff_competitiveness']:.3f}")
    
    # Interpretación
    print("\n" + "="*60)
    print("💡 INTERPRETACIÓN:")
    print("-"*60)
    
    comp = metrics.get('competitiveness', 0)
    
    if comp > 0.7:
        nivel = "MUY COMPETITIVA"
        desc = "Liga muy igualada, upsets frecuentes"
    elif comp > 0.5:
        nivel = "COMPETITIVA"
        desc = "Buen balance, variedad de resultados"
    elif comp > 0.3:
        nivel = "MODERADAMENTE COMPETITIVA"
        desc = "Algunos favoritos claros"
    else:
        nivel = "POCO COMPETITIVA"
        desc = "Dominada por pocos equipos"
    
    print(f"  Nivel: {nivel}")
    print(f"  Descripción: {desc}")
    
    print("\n" + "="*60)
    print("⚠️  IMPLICACIONES PARA PREDICCIONES:")
    print("-"*60)
    
    if comp > 0.5:
        print("  - Reducir confianza en favoritos claros")
        print("  - Considerar más empates")
        print("  - Esperar más 'sorpresas'")
        print("  - Usar umbrales más conservadores")
    else:
        print("  - Favoritos tienen mayor probabilidad")
        print("  - Menos upsets esperados")
        print("  - Modelo puede ser más 'confiado'")


def main():
    print("🏆 ANÁLISIS DE COMPETITIVIDAD - PREMIER LEAGUE")
    print("="*60)
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    
    standings_file = project_root / "data" / "cleaned" / "standings_2024_cleaned.csv"
    matches_file = project_root / "data" / "cleaned" / "matches_2024_cleaned.csv"
    
    if not standings_file.exists():
        print(f"Error: No existe {standings_file}")
        return
    
    if not matches_file.exists():
        print(f"Error: No existe {matches_file}")
        return
    
    standings_df = pd.read_csv(standings_file)
    matches_df = pd.read_csv(matches_file)
    
    print(f"  Standings: {len(standings_df)} registros")
    print(f"  Matches: {len(matches_df)} registros")
    
    # Cargar todas las temporadas
    all_standings = []
    for year in [2023, 2024, 2025]:
        file_path = project_root / "data" / "cleaned" / f"standings_{year}_cleaned.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['season'] = year
            all_standings.append(df)
    
    all_standings_df = pd.concat(all_standings, ignore_index=True) if all_standings else standings_df
    all_standings_df['season'] = standings_df['season'] if 'season' not in all_standings_df.columns else all_standings_df['season']
    
    # Calcular competitividad
    competitiveness_by_season = calculate_competitiveness(all_standings_df)
    
    metrics = calculate_league_competitiveness(standings_df, matches_df)
    
    # Mostrar resultados
    print_results(metrics, competitiveness_by_season)


if __name__ == "__main__":
    main()
