"""Trae partidos SCHEDULED (todavia no jugados) de Liga MX desde ESPN.

Tapa un hueco real: Transfermarkt (la fuente principal de src/build_ligamx_transfermarkt_dataset.py)
es editada por voluntarios y publica el calendario de la proxima jornada con
dias/semanas de retraso -- ESPN (y los sitios de noticias deportivas mexicanas)
lo tienen apenas la propia Liga MX lo anuncia. Este script NO reemplaza a
Transfermarkt para historial/resultados -- solo llena el hueco de "la jornada
que sigue todavia no esta en la fuente principal", con SOLO partidos SCHEDULED.

Bug real de ESPN encontrado y verificado (17-sep-2026): la clase CSS
"Table__Team away" del primer equipo de cada fila esta invertida -- verificado
cruzando el campo `venue` contra el estadio real de cada uno de los 9 partidos
de la jornada 10 del Apertura 2026 (fuentes: Infobae/TUDN/Fox Sports Mexico,
confirmando quien es local esa semana): en los 9/9 casos, el equipo marcado
"away" por ESPN es el LOCAL real. Por eso este script invierte esa etiqueta
--"home_team" = el marcado "away" por ESPN, "away_team" = el otro. Si ESPN
corrige su template en el futuro, esto habria que revisarlo (ver _parse_fixtures).
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from src.ligamx_team_aliases import normalize_team_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = PROJECT_ROOT / "data" / "ligamx" / "cleaned"

BASE_URL = "https://www.espn.com/soccer/schedule/_/league/mex.1/date/{date}"
_SOURCE = "espn"

_TITLE_RE = re.compile(r'<div class="Table__Title">([^<]*)</div>')
_ROW_RE = re.compile(
    r'<span class="Table__Team away"><a[^>]*>.*?<a[^>]*>(?P<team_a>[^<]+)</a></span>.*?'
    r'gameId/(?P<game_id>\d+)/[^"]*">\s*v\s*</a><span class="Table__Team"><a[^>]*>.*?<a[^>]*>(?P<team_b>[^<]+)</a></span>.*?'
    r'<td class="date__col Table__TD"><a[^>]*>(?P<time>[^<]+)</a></td>.*?'
    r'<td class="venue__col Table__TD"><div>(?P<venue>[^<]*)</div>',
    re.S,
)


def _fetch(date: str) -> str:
    url = BASE_URL.format(date=date)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_fixtures(html: str) -> pd.DataFrame:
    """`team_a` (clase "away" en el HTML de ESPN) es el LOCAL real -- ver
    docstring del modulo. `team_b` es el visitante real."""
    rows = []
    for date_label, block in zip(_TITLE_RE.findall(html), re.split(r'<div class="Table__Title">[^<]*</div>', html)[1:]):
        date = datetime.strptime(date_label.strip(), "%A, %B %d, %Y").strftime("%Y-%m-%d")
        for m in _ROW_RE.finditer(block):
            rows.append({
                "date": date,
                "home_team": normalize_team_name(m.group("team_a"), source=_SOURCE),
                "away_team": normalize_team_name(m.group("team_b"), source=_SOURCE),
                "time": m.group("time"),
                "venue": m.group("venue"),
                "espn_game_id": m.group("game_id"),
            })
    return pd.DataFrame(rows)


def fetch_upcoming_fixtures(start_date: str) -> pd.DataFrame:
    """`start_date` en formato YYYYMMDD. ESPN devuelve la semana completa
    (todos los dias con partidos) a partir de esa fecha en una sola pagina."""
    html = _fetch(start_date)
    return _parse_fixtures(html)


def merge_into_cleaned(fixtures: pd.DataFrame, season: int, phase: str, matchday: int) -> int:
    """Mete `fixtures` (ya con home_team/away_team/date normalizados) como
    SCHEDULED en matches_{season}_cleaned.csv, reemplazando cualquier fila
    previa de ese mismo matchday+phase (ej. un parche manual anterior) para
    no dejar duplicados. Devuelve cuantas filas quedaron insertadas."""
    path = CLEANED_DIR / f"matches_{season}_cleaned.csv"
    df = pd.read_csv(path)

    keep_mask = ~((df["phase"] == phase) & (df["matchday"] == matchday) & (df["status"] == "SCHEDULED"))
    df = df[keep_mask]

    next_id = int(df["id"].max()) + 1 if not df.empty else 1
    new_rows = []
    for i, row in fixtures.iterrows():
        new_rows.append({
            "id": next_id + i,
            "date": row["date"],
            "season": season,
            "phase": phase,
            "matchday": matchday,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": None,
            "away_score": None,
            "status": "SCHEDULED",
            "total_goals": None,
            "goal_difference": None,
            "result": None,
            "source": _SOURCE,
            "referee": None,
            "venue": row.get("venue"),
            "venue_city": None,
            "attendance": None,
            "home_xg": None,
            "away_xg": None,
            "home_manager": None,
            "away_manager": None,
            "home_system_of_play": None,
            "away_system_of_play": None,
            "source_matchday": matchday,
            "sources_agree": True,
            "n_sources": 1,
        })

    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(path, index=False)
    return len(new_rows)


def main():
    parser = argparse.ArgumentParser(description="Trae fixtures SCHEDULED de Liga MX desde ESPN")
    parser.add_argument("--date", required=True, help="Fecha de inicio YYYYMMDD (ESPN devuelve toda la semana desde ahi)")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--phase", choices=["apertura", "clausura"], required=True)
    parser.add_argument("--matchday", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar lo que se encontro, no escribir el CSV")
    args = parser.parse_args()

    fixtures = fetch_upcoming_fixtures(args.date)
    if fixtures.empty:
        print("No se encontraron partidos en ESPN para esa fecha.")
        return

    print(f"Encontrados {len(fixtures)} partidos:")
    for _, r in fixtures.iterrows():
        print(f"  {r['date']} {r['time']}: {r['home_team']} vs {r['away_team']} ({r['venue']})")

    if args.dry_run:
        return

    n = merge_into_cleaned(fixtures, args.season, args.phase, args.matchday)
    print(f"\n{n} partidos insertados como SCHEDULED en matches_{args.season}_cleaned.csv "
          f"(jornada {args.matchday}, {args.phase}, source=espn)")


if __name__ == "__main__":
    main()
