"""Construir cache local de datos de Championship (ELC) desde football-data.org.

Solo se ejecuta una vez para poblar data/championship_seeds.json. Las predicciones
no deben golpear la API (free ~10 req/min); se lee el cache.

Temporadas ELC disponibles con la API key free (verificado): 2023, 2024, 2025.
season=2022 esta restringido y se omite.
"""
import os
import json
import time
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_PATH = os.path.join(DATA_DIR, "championship_seeds.json")
API_TOKEN = "fd9ecc768e3644dfa9b30e9536031700"
ELC_ID = 2016
SEASONS = [2023, 2024, 2025]


def _fetch_standings(season: int) -> dict:
    url = f"https://api.football-data.org/v4/competitions/{ELC_ID}/standings?season={season}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": API_TOKEN})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_total_table(data: dict):
    for tbl in data.get("standings", []):
        if tbl.get("type") == "TOTAL":
            return tbl.get("table", [])
    # fallback: primera tabla
    if data.get("standings"):
        return data["standings"][0].get("table", [])
    return []


def build(force: bool = False) -> dict:
    if os.path.exists(CACHE_PATH) and not force:
        print(f"Cache ya existe: {CACHE_PATH} (usar force=True para reconstruir)")
        with open(CACHE_PATH) as f:
            return json.load(f)

    seeds = {"competition": "ELC", "seasons": {}}
    for season in SEASONS:
        for attempt in range(3):
            try:
                data = _fetch_standings(season)
                if "standings" not in data:
                    print(f"  ELC {season}: sin standings ({data.get('message')})")
                    break
                rows = _extract_total_table(data)
                season_data = {}
                for r in rows:
                    season_data[r["team"]["name"]] = {
                        "position": r.get("position"),
                        "points": r.get("points"),
                        "played_games": r.get("playedGames"),
                        "won": r.get("won"),
                        "draw": r.get("draw"),
                        "lost": r.get("lost"),
                        "goals_for": r.get("goalsFor"),
                        "goals_against": r.get("goalsAgainst"),
                    }
                seeds["seasons"][str(season)] = season_data
                print(f"  ELC {season}: {len(season_data)} equipos cacheados")
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 503):
                    wait = 15 * (attempt + 1)
                    print(f"  ELC {season}: rate-limit ({e.code}), reintento en {wait}s")
                    time.sleep(wait)
                else:
                    print(f"  ELC {season}: HTTP {e.code} ({e.reason})")
                    break
            except Exception as e:
                print(f"  ELC {season}: error {e}")
                break
        time.sleep(2)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(seeds, f, indent=2)
    print(f"Cache guardado: {CACHE_PATH}")
    return seeds


if __name__ == "__main__":
    build(force=True)
