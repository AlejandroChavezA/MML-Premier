#!/usr/bin/env python3

import requests
import pandas as pd
import json
from datetime import datetime
import os
import time
from pathlib import Path

class PremierLeagueDataUpdater:
    def __init__(self):
        self.api_url = "https://api.football-data.org/v4"
        self.headers = {
            'X-Auth-Token': 'fd9ecc768e3644dfa9b30e9536031700'
        }
        self.data_dir = Path("data")
        
    def _get_json(self, url, max_retries=5):
        """Obtener JSON de la API con reintentos ante límites de tasa (429) o errores 5xx."""
        last_err = None
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = int(response.headers.get('Retry-After', 2 ** (attempt + 1)))
                    print(f"   ⏳ Esperando {wait}s (reintento {attempt + 1}/{max_retries}) "
                          f"por código {response.status_code}...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"   ⏳ Reintento {attempt + 1}/{max_retries} tras error: {e}")
                    time.sleep(wait)
                    continue
                raise
        raise last_err

    def update_premier_league_data(self, season=2025):
        """Actualizar todos los datos de Premier League desde la API"""
        print("🔄 ACTUALIZANDO DATOS PREMIER LEAGUE")
        print("=" * 60)
        print(f"📅 Temporada: {season}")
        print(f"📅 Fecha actual: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            # 1. Actualizar partidos
            print("⚽ Actualizando partidos...")
            url = f"{self.api_url}/competitions/PL/matches?season={season}"
            matches_data = self._get_json(url)

            if 'matches' not in matches_data or not matches_data['matches']:
                raise ValueError("La API no devolvió partidos (respuesta inesperada)")
            
            matches_list = []
            for match in matches_data['matches']:
                status = str(match['status']).upper()
                match_info = {
                    'id': match['id'],
                    'date': match['utcDate'],
                    'matchday': match['matchday'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': match['score']['fullTime']['home'] if status == 'FINISHED' else None,
                    'away_score': match['score']['fullTime']['away'] if status == 'FINISHED' else None,
                    'status': status
                }
                matches_list.append(match_info)
            
            matches_df = pd.DataFrame(matches_list)
            
            # Guardar datos crudos
            raw_dir = self.data_dir / "raw"
            raw_dir.mkdir(exist_ok=True)
            matches_df.to_csv(raw_dir / f"matches_{season}_raw.csv", index=False)
            print(f"✅ Partidos guardados (raw): {len(matches_df)}")
            
            # Guardar datos en archivo principal para procesamiento
            matches_df.to_csv(self.data_dir / f"matches_{season}.csv", index=False)
            print(f"✅ Partidos guardados (main): {len(matches_df)}")
            
            # 2. Actualizar tabla de posiciones
            print("🏆 Actualizando tabla de posiciones...")
            url = f"{self.api_url}/competitions/PL/standings?season={season}"
            standings_data = self._get_json(url)

            if 'standings' not in standings_data or not standings_data['standings']:
                raise ValueError("La API no devolvió la tabla de posiciones")
            
            standings_list = []
            for standing in standings_data['standings'][0]['table']:
                team_info = {
                    'position': standing['position'],
                    'team': standing['team']['name'],
                    'played_games': standing['playedGames'],
                    'won': standing['won'],
                    'draw': standing['draw'],
                    'lost': standing['lost'],
                    'points': standing['points'],
                    'goals_for': standing['goalsFor'],
                    'goals_against': standing['goalsAgainst'],
                    'goal_difference': standing.get('goalDifference', 0),
                    'form': standing.get('form', '')
                }
                standings_list.append(team_info)
            
            standings_df = pd.DataFrame(standings_list)
            
            # Guardar datos crudos
            standings_df.to_csv(raw_dir / f"standings_{season}_raw.csv", index=False)
            print(f"✅ Tabla guardada (raw): {len(standings_df)} equipos")
            
            # Guardar tabla en archivo principal
            standings_df.to_csv(self.data_dir / f"standings_{season}.csv", index=False)
            print(f"✅ Tabla guardada (main): {len(standings_df)} equipos")
            
            # 3. Verificar si hay datos nuevos
            finished_matches = len(matches_df[matches_df['status'] == 'FINISHED'])
            total_matches = len(matches_df)
            new_matches = total_matches - 220  # Asumiendo que 220 era el número anterior
            
            print(f"📊 Análisis de actualización:")
            print(f"   Total partidos: {total_matches}")
            print(f"   Finalizados: {finished_matches}")
            print(f"   Programados: {total_matches - finished_matches}")
            print(f"   Nuevos partidos: {new_matches}")
            
            # 4. Crear registro de actualización
            update_log = {
                'timestamp': datetime.now().isoformat(),
                'season': season,
                'matches_count': total_matches,
                'finished_matches': finished_matches,
                'scheduled_matches': total_matches - finished_matches,
                'teams_count': len(pd.read_csv(self.data_dir / "cleaned" / "teams_cleaned.csv")),
                'success': True,
                'last_updated': datetime.now().isoformat()
            }
            
            # Guardar registro
            update_log_file = self.data_dir / "update_log.json"
            logs = [update_log]
            
            if update_log_file.exists():
                # Cargar logs anteriores
                try:
                    with open(update_log_file, 'r') as f:
                        logs = json.load(f)
                    logs.append(update_log)
                except:
                    logs = [update_log]
            
            with open(update_log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            print(f"💾 Registro guardado: {len(logs)} actualizaciones")
            
            print("\n🎉 ¡ACTUALIZACIÓN COMPLETADA!")
            print(f"📊 Resultado:")
            print(f"   • Partidos: {total_matches} ({finished_matches} finalizados)")
            print(f"   • Equipos: {len(pd.read_csv(self.data_dir / 'cleaned' / 'teams_cleaned.csv'))}")
            print(f"   ✅ Datos actualizados y guardados")
            print(f"🚀 Sistema listo para reentrenar modelos")
            print(f"\n🚀 Para procesar los datos actualizados:")
            print(f"   python3 data_cleaning.py  # Procesar datos")
            print(f"   python3 main.py --train  # Reentrenar modelos")
            print(f"   python3 main.py           # Iniciar menú")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en actualización: {e}")
            return False

def main():
    """Función principal del actualizador"""
    print("🔄 ACTUALIZADOR DE DATOS PREMIER LEAGUE")
    print("=" * 60)
    print("Este script descarga los datos más recientes de Premier League")
    print("=" * 60)
    
    # Siempre usar 2025 (la temporada actual)
    season = 2025
    
    print(f"\n🚀 Iniciando actualización de la temporada {season}...")
    
    updater = PremierLeagueDataUpdater()
    success = updater.update_premier_league_data(season)
    
    if success:
        print("\n🎉 ¡ACTUALIZACIÓN COMPLETADA!")
        print("📊 Los datos están listos para usar con el sistema de predicciones")
        print("📊 Datos actualizados exitosamente")
        print(f"✅ Datos actualizados y guardados")
        print("🚀 Sistema listo para reentrenar modelos")
        print("\n🚀 Para procesar los datos actualizados:")
        print("   python3 data_cleaning.py # Procesar datos")
        print("   python3 main.py --train # Reentrenar modelos")
        print("   python3 main.py           # Iniciar menú")
    
    else:
        print("\n❌ Hubo un error en la actualización")


if __name__ == "__main__":
    main()