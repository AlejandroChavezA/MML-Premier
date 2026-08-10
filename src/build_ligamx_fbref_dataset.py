"""Descarga y cachea el dataset publico de Liga MX del repo GitHub:

  yaacob117/Liga_MX_prediction_matches/matches.csv

Este dataset aporta:
- temporadas 2021-2024
- xG / xGA
- posesion, arbitro, alineacion, tiros, etc.

Se filtra solo fase regular (Apertura/Clausura Regular Season) y se excluye la liguilla.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ligamx" / "raw" / "matches" / "fbref"
RAW_URL = "https://raw.githubusercontent.com/yaacob117/Liga_MX_prediction_matches/main/matches.csv"


def download(force: bool = False) -> tuple[Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "fbref_matches_raw.csv"
    regular_path = DATA_DIR / "fbref_matches_regular.csv"

    if raw_path.exists() and regular_path.exists() and not force:
        print(f"Cache existente: {raw_path.name}, {regular_path.name}")
        return raw_path, regular_path

    resp = requests.get(RAW_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    raw_path.write_text(resp.text, encoding="utf-8")

    df = pd.read_csv(raw_path)
    # Solo temporada regular
    regular = df[df["Round"].astype(str).str.contains("Regular Season", case=False, na=False)].copy()
    # Normalizar columnas básicas para que sea más fácil de consumir
    regular.rename(
        columns={
            "Date": "date",
            "Time": "time",
            "Comp": "competition",
            "Round": "round",
            "Day": "day",
            "Venue": "venue",
            "Result": "result",
            "GF": "home_goals",
            "GA": "away_goals",
            "Opponent": "opponent",
            "xG": "xg",
            "xGA": "xga",
            "Poss": "possession",
            "Attendance": "attendance",
            "Captain": "captain",
            "Formation": "formation",
            "Referee": "referee",
            "Match Report": "match_report",
            "Notes": "notes",
            "Sh": "shots",
            "SoT": "shots_on_target",
            "Dist": "distance",
            "FK": "free_kicks",
            "PK": "penalties",
            "PKatt": "penalty_attempts",
            "Season": "season",
            "Team": "team",
        },
        inplace=True,
    )
    # Limpiar columnas sobrantes de índice
    if "Unnamed: 0" in regular.columns:
        regular.drop(columns=["Unnamed: 0"], inplace=True)

    regular_path.write_text(regular.to_csv(index=False), encoding="utf-8")

    print(
        f"Liga MX FBref-like dataset: raw={len(df)} rows | regular={len(regular)} rows | "
        f"seasons={sorted(regular['season'].dropna().unique().tolist())}"
    )
    print(f"Guardado: {raw_path}")
    print(f"Guardado: {regular_path}")
    return raw_path, regular_path


if __name__ == "__main__":
    download(force=True)
