"""Descargar y cachear Liga MX (fase liga solo) desde API-Football.

Salida:
- data/ligamx/ligamx_{season}.json  (cache completo de fase liga)
- data/ligamx/ligamx_{season}.csv   (tabla plana util para features)

Temporadas libres verificadas:
- 2022
- 2023
- 2024

Se excluye explicitamente la liguilla/playoffs.
"""

from __future__ import annotations

import csv
import json
import os
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ligamx" / "raw" / "matches" / "api_football"
API_BASE = "https://v3.football.api-sports.io"
LEAGUE_ID = 262
SEASONS = [2022, 2023, 2024]
PLAYOFF_MARKERS = ("Play-offs", "Quarter-finals", "Semi-finals", "Final")


def _load_api_key() -> str:
    env_files = [PROJECT_ROOT / ".env.local", PROJECT_ROOT / ".env"]
    for env in env_files:
        if not env.exists():
            continue
        for line in env.read_text().splitlines():
            if line.startswith("API_FOOTBALL_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("API_FOOTBALL_API_KEY no encontrada en .env.local/.env")


def _get_json(url: str, api_key: str) -> dict:
    req = urllib.request.Request(url, headers={"x-apisports-key": api_key})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_regular_round(round_name: str) -> bool:
    if not round_name:
        return False
    if not (round_name.startswith("Apertura") or round_name.startswith("Clausura")):
        return False
    return not any(marker in round_name for marker in PLAYOFF_MARKERS)


def _flatten_fixture(item: dict) -> dict:
    fixture = item.get("fixture", {})
    league = item.get("league", {})
    teams = item.get("teams", {})
    goals = item.get("goals", {})
    return {
        "fixture_id": fixture.get("id"),
        "date": fixture.get("date"),
        "status": fixture.get("status", {}).get("long"),
        "round": league.get("round"),
        "season": league.get("season"),
        "home_team": teams.get("home", {}).get("name"),
        "away_team": teams.get("away", {}).get("name"),
        "home_score": goals.get("home"),
        "away_score": goals.get("away"),
    }


def build(force: bool = False) -> None:
    api_key = _load_api_key()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        json_path = DATA_DIR / f"ligamx_{season}.json"
        csv_path = DATA_DIR / f"ligamx_{season}.csv"
        if json_path.exists() and csv_path.exists() and not force:
            print(f"OK cache existe: {season}")
            continue

        rounds = _get_json(f"{API_BASE}/fixtures/rounds?league={LEAGUE_ID}&season={season}", api_key).get("response", [])
        standings = _get_json(f"{API_BASE}/standings?league={LEAGUE_ID}&season={season}", api_key).get("response", [])
        fixtures = _get_json(f"{API_BASE}/fixtures?league={LEAGUE_ID}&season={season}", api_key).get("response", [])

        regular_rounds = [r for r in rounds if _is_regular_round(r)]
        regular_fixtures = [f for f in fixtures if f.get("league", {}).get("round") in regular_rounds]

        payload = {
            "league_id": LEAGUE_ID,
            "season": season,
            "rounds": rounds,
            "regular_rounds": regular_rounds,
            "standings": standings,
            "fixtures": regular_fixtures,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["fixture_id", "date", "status", "round", "season", "home_team", "away_team", "home_score", "away_score"],
            )
            writer.writeheader()
            for item in regular_fixtures:
                writer.writerow(_flatten_fixture(item))

        print(
            f"Liga MX {season}: rounds={len(rounds)} regular_rounds={len(regular_rounds)} "
            f"fixtures_regular={len(regular_fixtures)} -> {json_path.name}, {csv_path.name}"
        )


if __name__ == "__main__":
    build(force=True)
