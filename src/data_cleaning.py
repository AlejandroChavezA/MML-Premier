import pandas as pd
import numpy as np
from datetime import datetime
import os

class PremierLeagueDataCleaner:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.cleaned_dir = f"{data_dir}/cleaned"
        self.create_cleaned_directory()
    
    def create_cleaned_directory(self):
        """Crear directorio para datos limpios"""
        if not os.path.exists(self.cleaned_dir):
            os.makedirs(self.cleaned_dir)
            print(f"📁 Creado directorio: {self.cleaned_dir}")
    
    def analyze_data_quality(self):
        """Analizar la calidad de los datos"""
        print("🔍 ANÁLISIS DE CALIDAD DE DATOS")
        print("="*60)
        
        issues = []
        
        # Analizar cada temporada
        for year in [2023, 2024, 2025]:
            matches = pd.read_csv(f"{self.data_dir}/matches_{year}.csv")
            standings = pd.read_csv(f"{self.data_dir}/standings_{year}.csv")
            
            print(f"\n📊 TEMPORADA {year}:")
            
            # 1. Verificar valores nulos
            null_matches = matches.isnull().sum().sum()
            null_standings = standings.isnull().sum().sum()
            
            print(f"  Valores nulos - Partidos: {null_matches}, Tabla: {null_standings}")
            
            # 2. Verificar consistencia de datos
            if year == 2025:
                null_scores = matches[(matches['home_score'].isnull()) | (matches['away_score'].isnull())]
                finished_games = matches[matches['status'] == 'FINISHED']
                timed_games = matches[matches['status'] == 'TIMED']
                
                print(f"  Partidos finalizados: {len(finished_games)}")
                print(f"  Partidos programados: {len(timed_games)}")
                print(f"  Partidos sin resultado: {len(null_scores)}")
                
                if len(null_scores) > 0:
                    issues.append(f"Temporada {year}: {len(null_scores)} partidos sin resultados")
            
            # 3. Verificar rangos lógicos
            invalid_goals = matches[
                (matches['home_score'] < 0) | 
                (matches['away_score'] < 0) |
                (matches['matchday'] < 1) |
                (matches['matchday'] > 38)
            ]
            
            if len(invalid_goals) > 0:
                issues.append(f"Temporada {year}: {len(invalid_goals)} partidos con datos inválidos")
                print(f"  ⚠️  Datos inválidos: {len(invalid_goals)} partidos")
            else:
                print(f"  ✅ Todos los datos válidos")
        
        return issues
    
    def clean_matches_data(self, year):
        """Limpiar datos de partidos para una temporada específica"""
        print(f"\n🧹 Limpiando datos de partidos {year}...")
        
        matches = pd.read_csv(f"{self.data_dir}/matches_{year}.csv")
        original_count = len(matches)
        
        # 1. Convertir tipos de datos
        matches['date'] = pd.to_datetime(matches['date'])
        matches['home_score'] = pd.to_numeric(matches['home_score'], errors='coerce')
        matches['away_score'] = pd.to_numeric(matches['away_score'], errors='coerce')
        
        # 2. Eliminar partidos completamente inválidos
        invalid_matches = matches[
            matches['home_team'].isnull() | 
            matches['away_team'].isnull() |
            matches['matchday'].isnull()
        ]
        
        if len(invalid_matches) > 0:
            matches = matches[~matches.index.isin(invalid_matches.index)]
            print(f"  Eliminados {len(invalid_matches)} partidos inválidos")
        
        # 3. Para 2025: manejar partidos futuros
        if year == 2025:
            # Separar partidos jugados vs programados
            finished = matches[matches['status'] == 'FINISHED'].copy()
            scheduled = matches[matches['status'] == 'TIMED'].copy()
            
            # Para partidos programados, establecer scores como NaN
            scheduled.loc[:, 'home_score'] = np.nan
            scheduled.loc[:, 'away_score'] = np.nan
            
            # Volver a unir
            matches = pd.concat([finished, scheduled], ignore_index=True)
            print(f"  Procesados {len(finished)} partidos finalizados y {len(scheduled)} programados")
        
        # 4. Validar rangos lógicos
        matches = matches[
            (matches['matchday'] >= 1) & 
            (matches['matchday'] <= 38)
        ]
        
        # 5. Añadir columnas adicionales
        matches['total_goals'] = matches['home_score'].fillna(0) + matches['away_score'].fillna(0)
        matches['goal_difference'] = matches['home_score'].fillna(0) - matches['away_score'].fillna(0)
        
        # 6. Determinar resultado (solo para partidos finalizados)
        def get_result(row):
            if pd.isna(row['home_score']) or pd.isna(row['away_score']):
                return 'SIN JUGAR'
            elif row['home_score'] > row['away_score']:
                return 'LOCAL'
            elif row['home_score'] < row['away_score']:
                return 'VISITANTE'
            else:
                return 'EMPATE'
        
        matches['result'] = matches.apply(get_result, axis=1)
        
        # Guardar datos limpios
        output_path = f"{self.cleaned_dir}/matches_{year}_cleaned.csv"
        matches.to_csv(output_path, index=False)
        
        print(f"  ✅ Guardado: {output_path}")
        print(f"  Registros: {original_count} → {len(matches)}")
        
        return matches
    
    def clean_standings_data(self, year):
        """Limpiar datos de tabla de posiciones"""
        print(f"\n🏆 Limpiando tabla de posiciones {year}...")
        
        standings = pd.read_csv(f"{self.data_dir}/standings_{year}.csv")
        original_count = len(standings)
        
        # 1. Validar que tengamos 20 equipos
        if len(standings) != 20:
            print(f"  ⚠️  Expected 20 teams, found {len(standings)}")
        
        # 2. Validar consistencia de puntos
        calculated_points = standings['won'] * 3 + standings['draw'] * 1
        points_mismatch = standings[standings['points'] != calculated_points]
        
        if len(points_mismatch) > 0:
            print(f"  ⚠️  {len(points_mismatch)} equipos con inconsistencia en puntos")
            # Corregir puntos automáticamente
            standings['points'] = calculated_points
            print(f"  ✅ Puntos corregidos automáticamente")
        
        # 3. Validar diferencia de gol
        calculated_gd = standings['goals_for'] - standings['goals_against']
        gd_mismatch = standings[standings['goal_difference'] != calculated_gd]
        
        if len(gd_mismatch) > 0:
            print(f"  ⚠️  {len(gd_mismatch)} equipos con inconsistencia en diferencia de gol")
            standings['goal_difference'] = calculated_gd
            print(f"  ✅ Diferencia de gol corregida")
        
        # 4. Añadir columnas útiles
        standings['win_rate'] = (standings['won'] / standings['played_games']).round(3)
        standings['points_per_game'] = (standings['points'] / standings['played_games']).round(2)
        standings['goals_for_per_game'] = (standings['goals_for'] / standings['played_games']).round(2)
        standings['goals_against_per_game'] = (standings['goals_against'] / standings['played_games']).round(2)
        
        # 5. Ordenar por posición
        standings = standings.sort_values('position').reset_index(drop=True)
        
        # Guardar datos limpios
        output_path = f"{self.cleaned_dir}/standings_{year}_cleaned.csv"
        standings.to_csv(output_path, index=False)
        
        print(f"  ✅ Guardado: {output_path}")
        print(f"  Equipos: {len(standings)}")
        
        return standings
    
    def clean_teams_data(self):
        """Limpiar datos de equipos"""
        print(f"\n📋 Limpiando datos de equipos...")
        
        teams = pd.read_csv(f"{self.data_dir}/teams.csv")
        original_count = len(teams)
        
        # 1. Seleccionar columnas relevantes
        relevant_columns = ['id', 'name', 'shortName', 'tla', 'founded', 'venue', 'clubColors']
        teams_clean = teams[relevant_columns].copy()
        
        # 2. Limpiar valores nulos
        teams_clean['founded'] = teams_clean['founded'].fillna('Desconocido')
        teams_clean['clubColors'] = teams_clean['clubColors'].fillna('N/A')
        
        # 3. Estandarizar nombres
        teams_clean['name_clean'] = teams_clean['name'].str.strip()
        
        # Guardar datos limpios
        output_path = f"{self.cleaned_dir}/teams_cleaned.csv"
        teams_clean.to_csv(output_path, index=False)
        
        print(f"  ✅ Guardado: {output_path}")
        print(f"  Equipos: {len(teams_clean)}")
        
        return teams_clean
    
    def generate_cleaning_report(self):
        """Generar reporte de limpieza"""
        print(f"\n📄 GENERANDO REPORTE DE LIMPIEZA...")
        
        report = []
        report.append("REPORTE DE LIMPIEZA DE DATOS - PREMIER LEAGUE")
        report.append("=" * 50)
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Estadísticas generales
        for year in [2023, 2024, 2025]:
            matches_original = pd.read_csv(f"{self.data_dir}/matches_{year}.csv")
            matches_clean = pd.read_csv(f"{self.cleaned_dir}/matches_{year}_cleaned.csv")
            
            standings_original = pd.read_csv(f"{self.data_dir}/standings_{year}.csv")
            standings_clean = pd.read_csv(f"{self.cleaned_dir}/standings_{year}_cleaned.csv")
            
            report.append(f"TEMPORADA {year}:")
            report.append(f"  Partidos: {len(matches_original)} → {len(matches_clean)}")
            report.append(f"  Tabla posiciones: {len(standings_original)} → {len(standings_clean)}")
            
            if year == 2025:
                finished = matches_clean[matches_clean['status'] == 'FINISHED']
                scheduled = matches_clean[matches_clean['status'] == 'TIMED']
                report.append(f"  Partidos finalizados: {len(finished)}")
                report.append(f"  Partidos programados: {len(scheduled)}")
            
            report.append("")
        
        # Guardar reporte
        report_path = f"{self.cleaned_dir}/limpieza_reporte.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Reporte guardado: {report_path}")
        return report_path
    
    def run_full_cleaning(self):
        """Ejecutar proceso completo de limpieza"""
        print("🚀 INICIANDO PROCESO COMPLETO DE LIMPIEZA")
        print("=" * 60)
        
        # 1. Analizar calidad de datos
        issues = self.analyze_data_quality()
        
        # 2. Limpiar cada dataset
        for year in [2023, 2024, 2025]:
            self.clean_matches_data(year)
            self.clean_standings_data(year)
        
        self.clean_teams_data()
        
        # 3. Generar reporte
        self.generate_cleaning_report()
        
        print(f"\n🎉 ¡LIMPIEZA COMPLETADA!")
        print(f"📁 Datos limpios guardados en: {self.cleaned_dir}")
        
        return True

if __name__ == "__main__":
    cleaner = PremierLeagueDataCleaner()
    cleaner.run_full_cleaning()