"""Consolida las 4 fuentes de Liga MX (API-Football, FBref, Transfermarkt,
openfootball) en un dataset unico por temporada, espejando la estructura de
data/cleaned/ que ya usa Premier League (ver src/data_cleaning.py).

Una "temporada" agrupa Apertura + Clausura del mismo ciclo: mes >= 6
pertenece al Apertura de ese año calendario, mes < 6 al Clausura del año
anterior. Ej: temporada 2024 = Apertura 2024 (jun/jul-dic 2024) + Clausura
2025 (ene-may 2025). El corte es junio (no julio, a diferencia de
season_utils.get_current_season() de Premier) porque el Apertura de Liga MX
a veces arranca la ultima semana de junio (ej. Apertura 2023-24 empezo el
30-jun-2023) y el Clausura nunca pasa de mayo.

Las fuentes no cubren las mismas temporadas:
- API-Football (ligamx_{2022,2023,2024}.json): temporadas 2022, 2023, 2024.
- FBref (fbref_matches_regular.csv): temporadas 2020-2023 (corta a mitad del
  Clausura 2024).
- Transfermarkt (tm_matches_*.csv): temporadas 2025 (completa) y 2026 (en
  curso).
- openfootball (data/ligamx/raw/matches/openfootball/*_mx1.txt, via
  build_ligamx_openfootball_dataset.load_primera): temporadas 2010-2024.
  Unica fuente para 2010-2019; en 2020-2023 solo cruza/rellena (menor
  prioridad, ver SOURCE_PRIORITY) porque no trae arbitro/venue/xg.
Donde varias fuentes cubren la misma temporada se combinan y se deduplican
por partido; donde solo hay una fuente el resultado es esa fuente sola, mas
enriquecida con los campos extra que traiga.

Salida:
- data/ligamx/cleaned/matches_{season}_cleaned.csv
- data/ligamx/cleaned/standings_{season}_cleaned.csv

Uso:
    python -m src.build_ligamx_cleaned_dataset
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ligamx_team_aliases import normalize_team_name
from src.build_ligamx_openfootball_dataset import load_primera as load_openfootball

DATA_DIR = PROJECT_ROOT / "data" / "ligamx"
RAW_MATCHES_DIR = DATA_DIR / "raw" / "matches"
API_FOOTBALL_DIR = RAW_MATCHES_DIR / "api_football"
FBREF_DIR = RAW_MATCHES_DIR / "fbref"
TRANSFERMARKT_DIR = RAW_MATCHES_DIR / "transfermarkt"
CLEANED_DIR = DATA_DIR / "cleaned"

# Columnas base: mismo esquema que data/cleaned/matches_{year}_cleaned.csv de
# Premier (id,date,matchday,home_team,away_team,home_score,away_score,status,
# total_goals,goal_difference,result), mas season/phase que Liga MX si
# necesita al tener dos torneos por temporada.
BASE_COLUMNS = [
    "id", "date", "season", "phase", "matchday",
    "home_team", "away_team", "home_score", "away_score",
    "status", "total_goals", "goal_difference", "result",
]

# Campos opcionales que no todas las fuentes traen. Se agregan al final
# (Paso 1 del plan: "mantener versionado agregando campos al final") y
# quedan NaN cuando la fuente que gano el partido no los tenia.
EXTRA_COLUMNS = [
    "source", "referee", "venue", "venue_city", "attendance",
    "home_xg", "away_xg", "home_manager", "away_manager",
    "home_system_of_play", "away_system_of_play", "source_matchday",
]

ALL_COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS

# Prioridad de fuente para rellenar campos extra cuando el mismo partido
# aparece en mas de una fuente (gana quien tenga el dato, en este orden).
# openfootball va ultimo: es texto plano sin arbitro/venue/xg, solo aporta
# marcador + jornada real, y ademas es la unica fuente que cubre 2010-2019.
SOURCE_PRIORITY = ["transfermarkt", "fbref", "api_football", "openfootball"]


def _season_of(date: pd.Timestamp) -> int:
    # Corte en junio, no julio: el Apertura a veces arranca la ultima semana
    # de junio (ej. Apertura 2023-24 empezo el 30-jun-2023) y el Clausura
    # nunca pasa de mayo (fase liga ni liguilla), asi que un corte en julio
    # clasificaba esos partidos de apertura como clausura de la temporada
    # anterior -- confirmado con fbref en matches_2022_cleaned.csv (3
    # partidos del 30-jun-2023 que en realidad son Apertura 2023 jornada 1
    # quedaban como Clausura 2022 jornada 1).
    return int(date.year if date.month >= 6 else date.year - 1)


def _phase_of(date: pd.Timestamp) -> str:
    return "apertura" if date.month >= 6 else "clausura"


def _empty_extra_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({c: [None] * n for c in EXTRA_COLUMNS})


def load_api_football() -> pd.DataFrame:
    """Lee los JSON crudos de API-Football (mas completos que sus .csv)."""
    rows = []
    for path in sorted(API_FOOTBALL_DIR.glob("ligamx_20*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fixtures = payload.get("fixtures", payload if isinstance(payload, list) else [])
        for fx in fixtures:
            fixture = fx.get("fixture", {})
            teams = fx.get("teams", {})
            goals = fx.get("goals", {})
            league = fx.get("league", {})
            venue = fixture.get("venue") or {}
            status_short = (fixture.get("status") or {}).get("short", "")

            round_name = league.get("round") or ""
            round_match = re.match(r"(Apertura|Clausura)\s*-\s*(\d+)$", round_name.strip())
            if not round_match:
                # descarta Reclasificacion/liguilla: son partidos condicionales
                # (solo se juegan si hay empate en la tabla) que quedan
                # "programados" en la API para siempre si nunca se activaron,
                # y no son fase liga.
                continue

            home_raw = (teams.get("home") or {}).get("name")
            away_raw = (teams.get("away") or {}).get("name")
            if not home_raw or not away_raw:
                continue

            # API-Football guarda la fecha en UTC; los partidos de Liga MX
            # son de noche hora de Mexico (UTC-6), asi que la fecha UTC cae
            # un dia despues de la fecha local que usan FBref/Transfermarkt.
            # Se resta el offset para que la clave de dedup (dia calendario)
            # coincida entre fuentes.
            raw_date = fixture.get("date")
            local_date = None
            if raw_date:
                parsed = pd.Timestamp(raw_date)
                local_date = (parsed - pd.Timedelta(hours=6)).isoformat()

            rows.append({
                "date": local_date,
                "home_team": normalize_team_name(home_raw, source="api_football"),
                "away_team": normalize_team_name(away_raw, source="api_football"),
                "home_score": goals.get("home"),
                "away_score": goals.get("away"),
                "status": "FINISHED" if status_short == "FT" else "SCHEDULED",
                "source": "api_football",
                "referee": fixture.get("referee"),
                "venue": venue.get("name"),
                "venue_city": venue.get("city"),
                "attendance": None,
                "home_xg": None,
                "away_xg": None,
                "home_manager": None,
                "away_manager": None,
                "home_system_of_play": None,
                "away_system_of_play": None,
                "round_text": round_name,
                "source_matchday": int(round_match.group(2)),
            })
    return pd.DataFrame(rows)


def load_fbref() -> pd.DataFrame:
    """Formato por-equipo: nos quedamos solo con la fila 'Home' de cada
    partido (la fila 'Away' del rival es la misma info espejada)."""
    path = FBREF_DIR / "fbref_matches_regular.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["venue"].astype(str).str.strip().str.lower() == "home"].copy()

    rows = []
    for _, r in df.iterrows():
        home_raw = r.get("team")
        away_raw = r.get("opponent")
        if pd.isna(home_raw) or pd.isna(away_raw):
            continue
        rows.append({
            "date": r.get("date"),
            "home_team": normalize_team_name(home_raw, source="fbref"),
            "away_team": normalize_team_name(away_raw, source="fbref"),
            "home_score": r.get("home_goals"),
            "away_score": r.get("away_goals"),
            "status": "FINISHED",
            "source": "fbref",
            "referee": r.get("referee"),
            "venue": None,
            "venue_city": None,
            "attendance": r.get("attendance"),
            "home_xg": r.get("xg"),
            "away_xg": r.get("xga"),
            "home_manager": None,
            "away_manager": None,
            "home_system_of_play": r.get("formation"),
            "away_system_of_play": None,
            "round_text": r.get("round"),
            "source_matchday": None,  # FBref no trae numero de jornada explicito
        })
    return pd.DataFrame(rows)


def load_transfermarkt() -> pd.DataFrame:
    """Combina los archivos por-fase (match-row, solo FINISHED, ricos en
    detalle) con los archivos por-temporada (team-row, calendario completo
    incl. SCHEDULED) para no perder los partidos aun no jugados."""
    rows = []

    # 1) archivos por-fase: ya vienen match-row con detalle completo.
    for path in sorted(TRANSFERMARKT_DIR.glob("tm_matches_*ura_20*.csv")):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            home_raw, away_raw = r.get("home_team"), r.get("away_team")
            if pd.isna(home_raw) or pd.isna(away_raw):
                continue
            rows.append({
                "date": r.get("date"),
                "home_team": normalize_team_name(home_raw, source="transfermarkt"),
                "away_team": normalize_team_name(away_raw, source="transfermarkt"),
                "home_score": r.get("home_score"),
                "away_score": r.get("away_score"),
                "status": r.get("status", "FINISHED"),
                "source": "transfermarkt",
                "referee": r.get("referee"),
                "venue": r.get("stadium"),
                "venue_city": None,
                "attendance": r.get("attendance"),
                "home_xg": None,
                "away_xg": None,
                "home_manager": r.get("home_manager"),
                "away_manager": r.get("away_manager"),
                "home_system_of_play": r.get("home_system_of_play"),
                "away_system_of_play": r.get("away_system_of_play"),
                "round_text": f"matchday-{r.get('matchday')}",
                "source_matchday": r.get("matchday"),
            })

    detailed = pd.DataFrame(rows)
    detailed_keys = set()
    if not detailed.empty:
        detailed_keys = set(zip(
            pd.to_datetime(detailed["date"], errors="coerce").dt.date,
            detailed["home_team"], detailed["away_team"],
        ))

    # 2) archivos team-row (calendario completo): solo agregamos lo que NO
    # este ya cubierto por los archivos por-fase (para sumar los partidos
    # SCHEDULED que los por-fase no incluyen).
    extra_rows = []
    for path in sorted(TRANSFERMARKT_DIR.glob("tm_matches_20*.csv")):
        if "ura_" in path.name:
            continue  # ya procesado arriba
        df = pd.read_csv(path)
        df = df[df["venue"].astype(str).str.strip().str.upper() == "H"].copy()
        for _, r in df.iterrows():
            home_raw, away_raw = r.get("home_team"), r.get("away_team")
            if pd.isna(home_raw) or pd.isna(away_raw):
                continue
            home = normalize_team_name(home_raw, source="transfermarkt")
            away = normalize_team_name(away_raw, source="transfermarkt")
            d = pd.to_datetime(r.get("date"), errors="coerce")
            key = (d.date() if pd.notna(d) else None, home, away)
            if key in detailed_keys:
                continue
            extra_rows.append({
                "date": r.get("date"),
                "home_team": home,
                "away_team": away,
                "home_score": r.get("home_score"),
                "away_score": r.get("away_score"),
                "status": r.get("status", "SCHEDULED"),
                "source": "transfermarkt",
                "referee": None,
                "venue": None,
                "venue_city": None,
                "attendance": r.get("attendance"),
                "home_xg": None,
                "away_xg": None,
                "home_manager": None,
                "away_manager": None,
                "home_system_of_play": r.get("system_of_play"),
                "away_system_of_play": None,
                "round_text": f"matchday-{r.get('matchday')}",
                "source_matchday": r.get("matchday"),
            })

    return pd.concat([detailed, pd.DataFrame(extra_rows)], ignore_index=True)


def _combine_sources() -> pd.DataFrame:
    frames = [load_api_football(), load_fbref(), load_transfermarkt(), load_openfootball()]
    frames = [f for f in frames if not f.empty]
    combined = pd.concat(frames, ignore_index=True)

    # format="mixed": las 3 fuentes traen formatos de fecha distintos
    # (API-Football con hora+tz ISO, FBref/Transfermarkt solo fecha). Sin
    # esto, pandas infiere el formato de la primera fila y vuelve NaT todo
    # lo que no calce con ese formato al concatenar fuentes.
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    combined = combined.dropna(subset=["date", "home_team", "away_team"])
    combined["season"] = combined["date"].map(_season_of)
    combined["phase"] = combined["date"].map(_phase_of)
    combined["match_date_key"] = combined["date"].dt.date
    return combined


def _dedup_across_sources(combined: pd.DataFrame) -> pd.DataFrame:
    """Un mismo partido puede venir de 2+ fuentes (mismo dia, mismos
    equipos). Nos quedamos con una fila por partido, rellenando los campos
    extra vacios con lo que aporte otra fuente, en orden de SOURCE_PRIORITY.
    """
    priority = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    combined["source_rank"] = combined["source"].map(priority).fillna(len(priority))
    combined = combined.sort_values(["match_date_key", "home_team", "away_team", "source_rank"])

    merged_rows = []
    group_cols = ["match_date_key", "home_team", "away_team"]
    for _, group in combined.groupby(group_cols, sort=False):
        base = group.iloc[0].to_dict()
        for col in EXTRA_COLUMNS + ["home_score", "away_score", "status"]:
            if col == "source":
                continue
            if pd.isna(base.get(col)) or base.get(col) in (None, ""):
                for _, cand in group.iloc[1:].iterrows():
                    val = cand.get(col)
                    if pd.notna(val) and val not in (None, ""):
                        base[col] = val
                        break
        base["sources_agree"] = group["home_score"].nunique(dropna=True) <= 1 and group["away_score"].nunique(dropna=True) <= 1
        base["n_sources"] = group["source"].nunique()
        merged_rows.append(base)

    return pd.DataFrame(merged_rows)


def _assign_matchday(df: pd.DataFrame) -> pd.DataFrame:
    """Usa el numero de jornada real que ya trae la fuente (API-Football o
    Transfermarkt) siempre que este disponible. Solo cuando NINGUNA fuente
    lo trae (temporadas donde solo hay FBref: 2020-2021) se recurre a un
    fallback: como Liga MX siempre juega 9 partidos por jornada (18 equipos,
    sin fecha libre), se ordena por fecha dentro de (season, phase) y se
    arman bloques fijos de 9. Antes se agrupaba por huecos de fecha <=3
    dias, lo cual fusionaba varias jornadas reales en una sola cuando el
    calendario venia comprimido (partidos entre semana) y cortaba una
    jornada real en dos cuando habia un partido reprogramado.
    """
    df = df.sort_values(["season", "phase", "date"]).reset_index(drop=True)
    matchday = pd.to_numeric(df["source_matchday"], errors="coerce")

    fallback = pd.Series(index=df.index, dtype=float)
    for (season, phase), idx in df.groupby(["season", "phase"]).groups.items():
        idx = list(idx)
        fallback.loc[idx] = (np.arange(len(idx)) // 9) + 1

    df["matchday"] = matchday.fillna(fallback).astype(int)
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["total_goals"] = df["home_score"].fillna(0) + df["away_score"].fillna(0)
    df["goal_difference"] = df["home_score"].fillna(0) - df["away_score"].fillna(0)

    def _result(row):
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            return "SIN JUGAR"
        if row["home_score"] > row["away_score"]:
            return "LOCAL"
        if row["home_score"] < row["away_score"]:
            return "VISITANTE"
        return "EMPATE"

    df["result"] = df.apply(_result, axis=1)
    df = df.sort_values(["season", "date"]).reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)
    return df


def _build_standings(matches: pd.DataFrame) -> pd.DataFrame:
    finished = matches[matches["status"] == "FINISHED"].copy()
    if finished.empty:
        return pd.DataFrame(columns=[
            "position", "team", "played_games", "won", "draw", "lost", "points",
            "goals_for", "goals_against", "goal_difference", "win_rate",
            "points_per_game", "goals_for_per_game", "goals_against_per_game",
        ])

    teams = sorted(set(finished["home_team"]) | set(finished["away_team"]))
    out = []
    for team in teams:
        home = finished[finished["home_team"] == team]
        away = finished[finished["away_team"] == team]
        played = len(home) + len(away)
        wins = int((home["home_score"] > home["away_score"]).sum() + (away["away_score"] > away["home_score"]).sum())
        draws = int((home["home_score"] == home["away_score"]).sum() + (away["away_score"] == away["home_score"]).sum())
        losses = played - wins - draws
        gf = int(home["home_score"].fillna(0).sum() + away["away_score"].fillna(0).sum())
        ga = int(home["away_score"].fillna(0).sum() + away["home_score"].fillna(0).sum())
        pts = wins * 3 + draws
        out.append({
            "team": team, "played_games": played, "won": wins, "draw": draws,
            "lost": losses, "points": pts, "goals_for": gf, "goals_against": ga,
            "goal_difference": gf - ga,
            "win_rate": round(wins / played, 3) if played else 0.0,
            "points_per_game": round(pts / played, 2) if played else 0.0,
            "goals_for_per_game": round(gf / played, 2) if played else 0.0,
            "goals_against_per_game": round(ga / played, 2) if played else 0.0,
        })

    standings = pd.DataFrame(out)
    standings = standings.sort_values(
        ["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    standings.insert(0, "position", range(1, len(standings) + 1))
    return standings


def build() -> dict[int, pd.DataFrame]:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    combined = _combine_sources()
    deduped = _dedup_across_sources(combined)
    deduped = _assign_matchday(deduped)
    deduped = _finalize(deduped)

    results = {}
    for season, season_df in deduped.groupby("season"):
        season_df = season_df[ALL_COLUMNS + ["sources_agree", "n_sources"]].copy()
        matches_path = CLEANED_DIR / f"matches_{season}_cleaned.csv"
        season_df.to_csv(matches_path, index=False)

        standings = _build_standings(season_df)
        standings_path = CLEANED_DIR / f"standings_{season}_cleaned.csv"
        standings.to_csv(standings_path, index=False)

        n_finished = (season_df["status"] == "FINISHED").sum()
        n_disagree = (~season_df["sources_agree"]).sum()
        sources_used = sorted(set(season_df["source"].dropna()))
        print(
            f"Temporada {season}: {len(season_df)} partidos "
            f"({n_finished} jugados) fuentes={sources_used} "
            f"desacuerdos_entre_fuentes={n_disagree} -> {matches_path.name}"
        )
        results[season] = season_df

    return results


if __name__ == "__main__":
    build()
