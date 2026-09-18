"""Scraper de BetExplorer para cuotas historicas 1X2 de Liga MX (fase regular).

BetExplorer sirve las cuotas incrustadas en HTML plano (sin JS), a diferencia
de OddsPortal (que arma la tabla via API/React y no expone filas en el HTML
crudo). Cada temporada tiene "stages" (pestañas) por fase -- Apertura,
Apertura Play In/Play Offs, Clausura, Clausura Play In/Play Offs. Solo se
scrapea Apertura y Clausura (fase regular, mismo alcance que el resto de
Liga MX en este repo -- ver PLAYOFF_STAGE_MARKERS abajo).

Cobertura real confirmada: BetExplorer solo tiene cuotas de Liga MX desde la
temporada 2019-2020 en adelante (2018-19 y anteriores redirigen al home, no
existen). Por eso SEASONS empieza en 2019, no en 2010 como el resto del
dataset.

Salida por temporada+fase:
- data/ligamx/raw/odds/betexplorer/be_odds_{phase}_{season}.csv
Y un combinado:
- data/ligamx/cleaned/odds_ligamx_betexplorer.csv
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.ligamx_team_aliases import normalize_team_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ODDS_RAW_DIR = PROJECT_ROOT / "data" / "ligamx" / "raw" / "odds" / "betexplorer"
CLEANED_PATH = PROJECT_ROOT / "data" / "ligamx" / "cleaned" / "odds_ligamx_betexplorer.csv"

BASE_URL = "https://www.betexplorer.com"
# primera temporada con cuotas disponibles en BetExplorer para Liga MX
FIRST_SEASON = 2019
LAST_SEASON = 2026  # temporada actual (apertura en curso a la fecha de este scraper)

# pestañas de fase regular -- se excluyen explicitamente "Play In"/"Play Offs"
# (liguilla), igual que el resto del pipeline de Liga MX en este repo.
REGULAR_STAGE_TITLES = {"Apertura", "Clausura"}

_SOURCE = "betexplorer"


def _get(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    return resp.text


def _season_url(season: int) -> str:
    return f"{BASE_URL}/football/mexico/liga-mx-{season}-{season + 1}/results/"


def _find_regular_stage_tabs(html: str) -> dict[str, str]:
    """Devuelve {titulo_pestaña: stage_id} solo para Apertura/Clausura (sin liguilla)."""
    tabs = re.findall(r'href="\?stage=([^"]+)"[^>]*>([^<]+)</a>', html)
    result: dict[str, str] = {}
    for stage_id, title in tabs:
        title = title.strip()
        if title in REGULAR_STAGE_TITLES:
            result[title] = stage_id
    return result


_ROW_RE = re.compile(
    r'<tr><td class="h-text-left"><a[^>]*href="[^"]*"[^>]*>(?P<teams>.*?)</a></td>'
    r'<td class="h-text-center"><a[^>]*>(?P<score>\d+:\d+)</a></td>'
    r'(?P<odds_block>.*?)'
    r'<td class="h-text-right h-text-no-wrap">(?P<date>[^<]+)</td></tr>',
    re.S,
)
_TEAM_SPAN_RE = re.compile(r'<span>(?:<strong>)?([^<]+)(?:</strong>)?</span>')
_ODD_RE = re.compile(r'data-odd="([\d.]+)"')


def _parse_match_date(raw: str, phase: str, season: int) -> pd.Timestamp:
    """BetExplorer usa 3 formatos segun que tan reciente es el partido:
    'dd.mm.yyyy' (temporadas cerradas), 'dd.mm.' sin anio (temporada en
    curso -- el anio se infiere de phase/season, igual que el resto del
    dataset: apertura = season, clausura = season+1) y 'Today'/'Yesterday'
    para los ultimos 1-2 partidos jugados.
    """
    raw = raw.strip()
    if raw == "Today":
        return pd.Timestamp.now().normalize()
    if raw == "Yesterday":
        return pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", raw):
        return pd.to_datetime(raw, format="%d.%m.%Y")
    if re.fullmatch(r"\d{2}\.\d{2}\.", raw):
        year = season if phase == "apertura" else season + 1
        return pd.to_datetime(f"{raw}{year}", format="%d.%m.%Y")
    raise ValueError(f"Formato de fecha no reconocido en BetExplorer: {raw!r}")


def _parse_stage_rows(html: str, phase: str, season: int) -> pd.DataFrame:
    rows = []
    for m in _ROW_RE.finditer(html):
        team_spans = _TEAM_SPAN_RE.findall(m.group("teams"))
        if len(team_spans) != 2:
            continue
        home_raw, away_raw = (t.strip() for t in team_spans)
        home_score, away_score = (int(x) for x in m.group("score").split(":"))
        odds = _ODD_RE.findall(m.group("odds_block"))
        if len(odds) != 3:
            continue
        odd_home, odd_draw, odd_away = (float(x) for x in odds)
        date = _parse_match_date(m.group("date"), phase=phase, season=season)

        rows.append({
            "date": date,
            "season": season,
            "phase": phase,
            "home_team_raw": home_raw,
            "away_team_raw": away_raw,
            "home_team": normalize_team_name(home_raw, source=_SOURCE),
            "away_team": normalize_team_name(away_raw, source=_SOURCE),
            "home_score": home_score,
            "away_score": away_score,
            "odd_home": odd_home,
            "odd_draw": odd_draw,
            "odd_away": odd_away,
            "source": _SOURCE,
        })
    return pd.DataFrame(rows)


def build_season(season: int) -> pd.DataFrame:
    url = _season_url(season)
    html = _get(url)
    stage_tabs = _find_regular_stage_tabs(html)

    frames = []
    for title, stage_id in stage_tabs.items():
        phase = "apertura" if title == "Apertura" else "clausura"
        stage_html = _get(f"{url}?stage={stage_id}")
        time.sleep(0.5)
        df = _parse_stage_rows(stage_html, phase=phase, season=season)
        if df.empty:
            continue
        ODDS_RAW_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ODDS_RAW_DIR / f"be_odds_{phase}_{season}.csv"
        df.to_csv(out_path, index=False)
        print(f"  {season} {phase}: {len(df)} partidos -> {out_path.relative_to(PROJECT_ROOT)}")
        frames.append(df)

    time.sleep(0.5)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    all_frames = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        print(f"Scrapeando temporada {season}-{season + 1}...")
        df = build_season(season)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("No se obtuvo ningun partido.")
        return

    combined = pd.concat(all_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CLEANED_PATH, index=False)
    print(f"\nTotal: {len(combined)} partidos con cuotas -> {CLEANED_PATH.relative_to(PROJECT_ROOT)}")
    print(combined.groupby(['season', 'phase']).size())


if __name__ == "__main__":
    main()
