import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os

class PremierLeagueDataCollector:
    def __init__(self):
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {
            'X-Auth-Token': 'fd9ecc768e3644dfa9b30e9536031700'
        }
        self.data_dir = "../data"
        
    def create_data_directory(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def get_premier_league_teams(self):
        """Obtener información de equipos de la Premier League"""
        url = f"{self.base_url}/competitions/PL/teams"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            teams_data = response.json()
            
            teams_df = pd.DataFrame(teams_data['teams'])
            teams_df.to_csv(f"{self.data_dir}/teams.csv", index=False)
            print(f"Guardados {len(teams_df)} equipos")
            return teams_df
        except Exception as e:
            print(f"Error obteniendo equipos: {e}")
            return None
    
    def get_premier_league_matches(self, season=2023):
        """Obtener partidos de la Premier League por temporada"""
        url = f"{self.base_url}/competitions/PL/matches?season={season}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            matches_data = response.json()
            
            matches_list = []
            for match in matches_data['matches']:
                match_info = {
                    'id': match['id'],
                    'date': match['utcDate'],
                    'matchday': match['matchday'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': match['score']['fullTime']['home'],
                    'away_score': match['score']['fullTime']['away'],
                    'status': match['status']
                }
                matches_list.append(match_info)
            
            matches_df = pd.DataFrame(matches_list)
            matches_df.to_csv(f"{self.data_dir}/matches_{season}.csv", index=False)
            print(f"Guardados {len(matches_df)} partidos de la temporada {season}")
            return matches_df
        except Exception as e:
            print(f"Error obteniendo partidos: {e}")
            return None
    
    def get_standings(self, season=2023):
        """Obtener tabla de posiciones"""
        url = f"{self.base_url}/competitions/PL/standings?season={season}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            standings_data = response.json()
            
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
                    'goal_difference': standing['goalDifference']
                }
                standings_list.append(team_info)
            
            standings_df = pd.DataFrame(standings_list)
            standings_df.to_csv(f"{self.data_dir}/standings_{season}.csv", index=False)
            print(f"Guardada tabla de posiciones de la temporada {season}")
            return standings_df
        except Exception as e:
            print(f"Error obteniendo tabla de posiciones: {e}")
            return None

if __name__ == "__main__":
    collector = PremierLeagueDataCollector()
    collector.create_data_directory()
    
    print("🔄 Recolectando datos de la Premier League...")
    
    # Obtener equipos
    print("\n Obteniendo equipos...")
    teams = collector.get_premier_league_teams()
    
    # Obtener datos de múltiples temporadas
    seasons = [2023, 2024, 2025]
    
    for season in seasons:
        print(f"\n Obteniendo partidos de la temporada {season}...")
        matches = collector.get_premier_league_matches(season)
        
        print(f"\n Obteniendo tabla de posiciones de la temporada {season}...")
        standings = collector.get_standings(season)
        
        if matches is not None:
            print(f"OK Temporada {season}: {len(matches)} partidos")
        else:
            print(f"ERROR Temporada {season}: No hay datos disponibles")
    
    print("\nOK ¡Datos recolectados exitosamente!")
    print(" Archivos guardados en la carpeta 'data/'")