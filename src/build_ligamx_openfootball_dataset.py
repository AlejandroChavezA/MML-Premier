"""Parsea los archivos de texto de openfootball/world (historial Liga MX +
Liga de Expansion MX / Ascenso MX, 2010-11 a 2024-25) descargados en
`data/ligamx/raw/matches/openfootball/`.

Formato de origen (ver NOTES.md en esa carpeta):
    ▪ Apertura, Matchday 1
      Fri Jul 23 2010
        20:10  Estudiantes Tecos       v Cruz Azul                0-3 (0-3)
      Sat Jul 24
        17:00  Jaguares Chiapas        v Club Necaxa              1-1 (0-0)
               CF Pachuca              v CF América               3-0 (2-0)

El anio solo aparece en la fecha cuando cambia respecto al anterior (se
reusa el ultimo visto). Se descartan por completo las fases que no sean
"Apertura, Matchday N" / "Clausura, Matchday N": liguilla (Playoffs,
Reclassification, Play-in, Qual. Round) y finales (Campeon, Final de
Ascenso) -- ver plan_5_ligas_ligamx.md, Liga MX entra "solo fase liga".

Dos ligas mezcladas en la misma carpeta, separadas por sufijo de archivo:
- `*_mx1.txt`              -> Liga MX (primera division).
- `*_mx2ascenso.txt`       -> Ascenso MX (segunda division, hasta 2019-20).
- `*_mx2expansion.txt`     -> Liga de Expansion MX (segunda division, desde 2020-21).

Primera y segunda division son ligas *distintas* (pool de equipos distinto):
no se mezclan resultados entre ellas. Primera se expone como fuente extra
(mas historia + cruce) para `build_ligamx_cleaned_dataset.py`; segunda se
limpia por separado en este mismo script (`build_segunda_cleaned`), pensada
para features de continuidad de equipos recien ascendidos (ej. Atlante),
no para el pool de partidos de Liga MX.
"""

from __future__ import annotations

import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ligamx_team_aliases import normalize_team_name

OPENFOOTBALL_DIR = PROJECT_ROOT / "data" / "ligamx" / "raw" / "matches" / "openfootball"
CLEANED_DIR = PROJECT_ROOT / "data" / "ligamx" / "cleaned"

STAGE_RE = re.compile(r"^▪\s+(Apertura|Clausura),\s+Matchday\s+(\d+)\s*$")
DATE_RE = re.compile(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Za-z]{3})\s+(\d{1,2})(?:\s+(\d{4}))?$")
MATCH_RE = re.compile(r"^(?:(\d{1,2}:\d{2})\s+)?(.+?)\s+v\s+(.+?)\s{2,}(\S.*)$")
SCORE_RE = re.compile(r"^(\d+)-(\d+)")


def _parse_file(path: Path) -> list[dict]:
    """Devuelve filas crudas (nombres de equipo SIN normalizar) de un solo
    archivo, solo fase liga (Apertura/Clausura, Matchday N)."""
    seed_year = int(path.name.split("-")[0])
    current_year = seed_year
    current_phase: str | None = None
    current_matchday: int | None = None
    current_date: date | None = None
    in_regular_stage = False

    rows: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("▪"):
            m = STAGE_RE.match(line)
            in_regular_stage = bool(m)
            if m:
                current_phase = m.group(1).lower()
                current_matchday = int(m.group(2))
            else:
                current_phase = None
                current_matchday = None
            continue

        dm = DATE_RE.match(line)
        if dm:
            # El anio se actualiza siempre, incluso en fases excluidas
            # (playoffs/finales), porque el marcador de cambio de anio puede
            # caer ahi (ej. "Campeon, Final" en mayo) y las jornadas
            # regulares siguientes (Clausura) dependen de ese estado.
            mon, day, year = dm.groups()
            if year:
                current_year = int(year)
            current_date = datetime.strptime(f"{mon} {day} {current_year}", "%b %d %Y").date()
            continue

        if not in_regular_stage:
            continue

        mm = MATCH_RE.match(line)
        if not mm:
            continue
        _time, home_raw, away_raw, rest = mm.groups()
        if current_date is None:
            raise ValueError(f"Partido sin fecha en {path.name}: {line!r}")

        sm = SCORE_RE.match(rest)
        if sm:
            home_score, away_score = int(sm.group(1)), int(sm.group(2))
            status = "FINISHED"
        else:
            home_score, away_score = None, None
            status = "CANCELLED" if "cancelled" in rest.lower() else "SCHEDULED"

        rows.append({
            "date": current_date.isoformat(),
            "phase": current_phase,
            "source_matchday": current_matchday,
            "home_raw": home_raw.strip(),
            "away_raw": away_raw.strip(),
            "home_score": home_score,
            "away_score": away_score,
            "status": status,
        })
    return rows


def _load(glob_pattern: str, *, normalize) -> pd.DataFrame:
    all_rows: list[dict] = []
    for path in sorted(OPENFOOTBALL_DIR.glob(glob_pattern)):
        for r in _parse_file(path):
            all_rows.append({
                "date": r["date"],
                "phase": r["phase"],
                "home_team": normalize(r["home_raw"]),
                "away_team": normalize(r["away_raw"]),
                "home_score": r["home_score"],
                "away_score": r["away_score"],
                "status": r["status"],
                "source": "openfootball",
                "referee": None,
                "venue": None,
                "venue_city": None,
                "attendance": None,
                "home_xg": None,
                "away_xg": None,
                "home_manager": None,
                "away_manager": None,
                "home_system_of_play": None,
                "away_system_of_play": None,
                "round_text": f"{r['phase']}-{r['source_matchday']}",
                "source_matchday": r["source_matchday"],
            })
    return pd.DataFrame(all_rows)


def _normalize_primera(name: str) -> str:
    return normalize_team_name(name, source="openfootball_primera")


def _normalize_segunda(name: str) -> str:
    # Un equipo de segunda puede ser uno que tambien jugo primera alguna vez
    # (ej. Atlante en 2019-20): probar el mapa completo, no solo el de
    # segunda division.
    return normalize_team_name(name, source="openfootball_segunda")


def load_primera() -> pd.DataFrame:
    """Liga MX (primera division), formato compatible con las otras fuentes
    de `build_ligamx_cleaned_dataset.py` (mismas columnas EXTRA_COLUMNS)."""
    return _load("*_mx1.txt", normalize=_normalize_primera)


def load_segunda() -> pd.DataFrame:
    """Ascenso MX / Liga de Expansion MX (segunda division). Liga distinta,
    NO se mezcla con los resultados de Liga MX."""
    return _load("*_mx2ascenso.txt", normalize=_normalize_segunda)


def load_segunda_expansion() -> pd.DataFrame:
    return _load("*_mx2expansion.txt", normalize=_normalize_segunda)


def _finalize_segunda(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # La fase ya viene de la fuente (encabezado "Apertura/Clausura, Matchday
    # N"), asi que la temporada se deriva de ahi en vez de inferirla de la
    # fecha: mas robusto que cualquier corte de mes fijo.
    df["season"] = np.where(df["phase"] == "apertura", df["date"].dt.year, df["date"].dt.year - 1)
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
    df["matchday"] = df["source_matchday"]
    df = df.sort_values(["season", "date"]).reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)

    columns = [
        "id", "date", "season", "phase", "matchday",
        "home_team", "away_team", "home_score", "away_score",
        "status", "total_goals", "goal_difference", "result",
    ]
    return df[columns]


def build_segunda_cleaned() -> dict[int, pd.DataFrame]:
    """Escribe data/ligamx/cleaned/segunda_matches_{season}_cleaned.csv, uno
    por temporada, separado de los archivos matches_{season}_cleaned.csv de
    Liga MX (primera)."""
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([load_segunda(), load_segunda_expansion()], ignore_index=True)
    finalized = _finalize_segunda(combined)

    results = {}
    for season, season_df in finalized.groupby("season"):
        path = CLEANED_DIR / f"segunda_matches_{season}_cleaned.csv"
        season_df.to_csv(path, index=False)
        n_finished = (season_df["status"] == "FINISHED").sum()
        print(f"Segunda division {season}: {len(season_df)} partidos ({n_finished} jugados) -> {path.name}")
        results[season] = season_df
    return results


if __name__ == "__main__":
    build_segunda_cleaned()
