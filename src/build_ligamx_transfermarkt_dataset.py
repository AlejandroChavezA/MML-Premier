"""Scraper de Transfermarkt para Liga MX (fase liga).

Usa la pagina general `gesamtspielplan`, que expone una tabla por jornada.
Eso evita duplicados por club y permite construir un dataset de partidos unico.

Salida por temporada:
- data/ligamx/raw/matches/transfermarkt/tm_matches_{phase}_{season_id}.csv
- data/ligamx/raw/standings/transfermarkt/tm_standings_{phase}_{season_id}.csv
- data/ligamx/raw/matches/transfermarkt/tm_raw_{phase}_{season_id}.json

Se trabaja solo la fase regular.
"""

from __future__ import annotations

import json
import re
import time
from io import StringIO
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCHES_DIR = PROJECT_ROOT / "data" / "ligamx" / "raw" / "matches" / "transfermarkt"
HTML_DIR = MATCHES_DIR / "html"
STANDINGS_DIR = PROJECT_ROOT / "data" / "ligamx" / "raw" / "standings" / "transfermarkt"
BASE_URL = "https://www.transfermarkt.com"

PAGES = [
    {
        "season_id": 2025,
        "phase": "apertura",
        "url": "https://www.transfermarkt.com/liga-mx-apertura/gesamtspielplan/wettbewerb/MEXA/saison_id/2025",
    },
    {
        "season_id": 2025,
        "phase": "clausura",
        "url": "https://www.transfermarkt.com/liga-mx-clausura/gesamtspielplan/wettbewerb/MEX1/saison_id/2025",
    },
    {
        "season_id": 2026,
        "phase": "apertura",
        "url": "https://www.transfermarkt.com/liga-mx-apertura/gesamtspielplan/wettbewerb/MEXA/saison_id/2026",
    },
]


def _get(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()
    return resp.text


def _clean_team(text: object) -> str:
    s = str(text).strip()
    s = re.sub(r"\s*\(\d+\.\)\s*$", "", s)
    s = re.sub(r"^\(\d+\.\)\s*", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_score(text: object) -> tuple[int | None, int | None, str]:
    s = str(text).strip()
    if not s or s in {"-:-", "nan", "None"}:
        return None, None, "SCHEDULED"
    m = re.search(r"(\d+)\s*:\s*(\d+)", s)
    if not m:
        return None, None, "SCHEDULED"
    return int(m.group(1)), int(m.group(2)), "FINISHED"


def _first_match(pattern: str, text: str, default: str | None = None, flags: int = re.S) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else default


def _count_matches(pattern: str, text: str, flags: int = re.S) -> int:
    return len(re.findall(pattern, text, flags))


def _parse_rank(text: object) -> int | None:
    m = re.search(r"\((\d+)\.\)", str(text or ""))
    return int(m.group(1)) if m else None


def _first_non_null(series: pd.Series):
    for value in series:
        if pd.notna(value) and str(value).strip().lower() != "nan":
            return value
    return None


def _normalize_team_name(text: object) -> str:
    s = _clean_team(text)
    aliases = {
        "Atlas Guadalajara": "Atlas",
        "Atlético de San Luis": "San Luis",
        "Club Necaxa": "Necaxa",
        "Club León FC": "León",
        "Deportivo Guadalajara": "Chivas",
        "Deportivo Toluca": "Toluca",
        "CF América": "América",
        "CF Pachuca": "Pachuca",
        "CF Monterrey": "Monterrey",
        "CD Cruz Azul": "CD Cruz Azul",
        "Club Puebla": "Puebla FC",
        "Puebla FC": "Puebla FC",
        "FC Juárez": "FC Juárez",
        "Mazatlán FC": "Mazatlán FC",
        "Querétaro FC": "Querétaro FC",
        "Santos Laguna": "Santos Laguna",
        "Tigres UANL": "Tigres UANL",
        "Club Tijuana": "Club Tijuana",
        "UNAM Pumas": "UNAM Pumas",
    }
    return aliases.get(s, s)


def _extract_club_name(html: str) -> str | None:
    raw = _first_match(r'<meta property="og:title" content="([^"]+)" />', html)
    if raw:
        return _normalize_team_name(raw.split(" - ", 1)[0])
    raw = _first_match(r'<meta name="keywords" content="([^"]+)" />', html)
    if raw:
        return _normalize_team_name(raw.split(",", 1)[0])
    return None


def _parse_detail_page(html: str) -> dict:
    stadium = _first_match(r'<p class="sb-zusatzinfos">.*?<a href="/stadion/stadion/verein/\d+/saison_id/\d+">([^<]+)</a>', html)
    attendance_raw = _first_match(r'Attendance:\s*([\d\.]+)', html)
    referee = _first_match(r'<strong>Referee:</strong>.*?<a title="([^"]+)"', html)
    managers = re.findall(r'<tr class="bench-table__tr">.*?<div>Manager:</div>.*?<a title="([^"]+)"', html, re.S)
    home_manager = managers[0] if len(managers) > 0 else None
    away_manager = managers[1] if len(managers) > 1 else None

    goals_section = _first_match(r'<div class="sb-ereignisse" id="sb-tore">(.*?)</div>\s*</div>\s*</div>\s*<div class="row">', html)
    cards_section = _first_match(r'<div class="sb-ereignisse" id="sb-karten">(.*?)</div>\s*</div>\s*</div>\s*<div class="ad-placement-note"', html)
    subs_section = _first_match(r'<div class="sb-ereignisse" id="sb-wechsel">(.*?)</div>\s*</div>\s*</div>\s*<div class="row">\s*<div class="large-12 columns">\s*<div class="box">\s*<h2 class="content-box-headline">\s*Cards', html)

    home_goal_scorers = []
    away_goal_scorers = []
    if goals_section:
        for side, scorer in re.findall(r'<li class="sb-aktion-(heim|gast)".*?<div class="sb-aktion-aktion">\s*<a title="([^"]+)"', goals_section, re.S):
            (home_goal_scorers if side == "heim" else away_goal_scorers).append(scorer)

    home_card_players = []
    away_card_players = []
    if cards_section:
        for side, player in re.findall(r'<li class="sb-aktion-(heim|gast)".*?<div class="sb-aktion-aktion"><a title="([^"]+)"', cards_section, re.S):
            (home_card_players if side == "heim" else away_card_players).append(player)

    home_sub_count = _count_matches(r'<li class="sb-aktion-heim">', subs_section or "")
    away_sub_count = _count_matches(r'<li class="sb-aktion-gast">', subs_section or "")
    home_yellow = _count_matches(r'<li class="sb-aktion-heim">.*?sb-gelb', cards_section or "")
    away_yellow = _count_matches(r'<li class="sb-aktion-gast">.*?sb-gelb', cards_section or "")
    home_red = _count_matches(r'<li class="sb-aktion-heim">.*?sb-rot', cards_section or "")
    away_red = _count_matches(r'<li class="sb-aktion-gast">.*?sb-rot', cards_section or "")

    return {
        "stadium": stadium,
        "attendance": int(attendance_raw.replace(".", "")) if attendance_raw else None,
        "referee": referee,
        "home_manager": home_manager,
        "away_manager": away_manager,
        "home_goal_scorers": "; ".join(home_goal_scorers) if home_goal_scorers else None,
        "away_goal_scorers": "; ".join(away_goal_scorers) if away_goal_scorers else None,
        "home_yellow_cards": home_yellow,
        "away_yellow_cards": away_yellow,
        "home_red_cards": home_red,
        "away_red_cards": away_red,
        "home_substitutions": home_sub_count,
        "away_substitutions": away_sub_count,
    }


def _club_rows_from_page(html: str, season_id: int, phase: str, source_url: str) -> pd.DataFrame:
    club_name = _extract_club_name(html)
    if not club_name:
        return pd.DataFrame()

    anchor = "MEX1" if phase == "clausura" else "MEXA"
    m = re.search(
        rf'<a name="{anchor}".*?</a>(.*?)(?=<a name="[A-Z0-9]+"|<div class="large-4 columns">)',
        html,
        re.S,
    )
    if not m:
        return pd.DataFrame()

    section = m.group(1)
    tables = pd.read_html(StringIO(section))
    if not tables:
        return pd.DataFrame()

    df = tables[0].copy()
    df.columns = [str(c) for c in df.columns]
    report_urls = [BASE_URL + href for href in re.findall(r'<a title="Match report" class="ergebnis-link"[^>]*href="([^"]+)"', section)]
    report_idx = 0
    rows = []

    for _, r in df.iterrows():
        matchday = str(r.get("Matchday") or "").strip()
        if not re.fullmatch(r"\d+", matchday):
            continue

        result_text = str(r.get("Result") or "").strip()
        home_score, away_score, status = _parse_score(result_text)
        if status != "FINISHED":
            continue

        opponent = _normalize_team_name(r.get("Opponent.1") or r.get("Opponent"))
        venue = str(r.get("Venue") or "").strip().upper()
        ranking = _parse_rank(r.get("Ranking"))
        date_raw = str(r.get("Date") or "").strip()
        date = pd.to_datetime(date_raw, errors="coerce", dayfirst=True)
        report_url = report_urls[report_idx] if report_idx < len(report_urls) else None
        report_idx += 1

        attendance_raw = r.get("Attendance")
        attendance = None
        if pd.notna(attendance_raw):
            attendance_text = str(attendance_raw).strip().replace(".", "")
            # Transfermarkt a veces pone "x" o "-" en vez de un número cuando
            # no hay asistencia reportada -- no es un dato faltante del scraper.
            if attendance_text.isdigit():
                attendance = int(attendance_text)

        row = {
            "season_id": season_id,
            "phase": phase,
            "matchday": int(matchday),
            "date_raw": date_raw,
            "date": date.isoformat() if pd.notna(date) else None,
            "time": str(r.get("Time") or "").strip(),
            "report_url": report_url,
            "attendance": attendance,
            "home_team": club_name if venue == "H" else opponent,
            "away_team": opponent if venue == "H" else club_name,
            "home_score": home_score,
            "away_score": away_score,
            "result": result_text,
            "home_system_of_play": str(r.get("System of play") or "").strip() if venue == "H" else None,
            "away_system_of_play": str(r.get("System of play") or "").strip() if venue == "A" else None,
            "home_club_rank": ranking if venue == "H" else None,
            "away_club_rank": ranking if venue == "A" else None,
            "club_team": club_name,
            "club_venue": venue,
            "source_url": source_url,
        }
        rows.append(row)

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    raw.drop_duplicates(subset=["matchday", "date_raw", "home_team", "away_team", "home_score", "away_score"], inplace=True)
    raw.reset_index(drop=True, inplace=True)
    return raw


def _load_club_enrichment(season_id: int, phase: str) -> pd.DataFrame:
    rows = []
    for path in sorted(HTML_DIR.glob(f"tm_raw_{season_id}_*.html")):
        html = path.read_text(encoding="utf-8")
        df = _club_rows_from_page(html, season_id=season_id, phase=phase, source_url=str(path))
        if not df.empty:
            rows.append(df)
    if not rows:
        return pd.DataFrame()

    raw = pd.concat(rows, ignore_index=True)
    agg = {
        "date_raw": _first_non_null,
        "date": _first_non_null,
        "time": _first_non_null,
        "report_url": _first_non_null,
        "attendance": _first_non_null,
        "home_system_of_play": _first_non_null,
        "away_system_of_play": _first_non_null,
        "home_club_rank": _first_non_null,
        "away_club_rank": _first_non_null,
        "club_team": _first_non_null,
        "club_venue": _first_non_null,
        "source_url": _first_non_null,
    }
    grouped = raw.groupby(["matchday", "home_team", "away_team", "home_score", "away_score"], as_index=False).agg(agg)
    return grouped


def _match_rows_from_page(html: str, season_id: int, phase: str, source_url: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    rows = []
    report_urls = [BASE_URL + href for href in re.findall(r'<a title="Match report" class="ergebnis-link"[^>]*href="([^"]+)"', html)]
    report_idx = 0

    # table 0 = filtro; el resto son tablas de jornadas
    for matchday_idx, table in enumerate(tables[1:], start=1):
        df = table.copy()
        df.columns = [str(c) for c in df.columns]

        # forward-fill fecha/hora dentro de la jornada. Ojo: dentro del mismo
        # dia, Transfermarkt mete una fila "header" de solo-hora (ej. "3:10
        # AM") cada vez que cambia el horario de kickoff -- esa fila NO trae
        # fecha real, solo hora. Si se hace ffill ingenuo de "Date" con esas
        # filas de por medio, la fecha real del dia se pisa con el string de
        # hora y pandas la parsea como *hoy* (bug real: partidos de jornadas
        # futuras terminaban con date=hoy en vez de su fecha real).
        if "Date" in df.columns:
            date_col = df["Date"].replace({"": pd.NA})
            looks_like_date = date_col.astype(str).str.contains(r"\d{1,2}/\d{1,2}/\d{2,4}", na=False)
            df["Date"] = date_col.where(looks_like_date).ffill()
        if "Time" in df.columns:
            df["Time"] = df["Time"].replace({"": pd.NA}).ffill()

        # filas reales: Home team = local, Home = score, Result = visitante
        if "Home team" not in df.columns or "Home" not in df.columns or "Result" not in df.columns:
            continue

        home_clean = df["Home team"].astype(str).map(_clean_team)
        away_clean = df["Result"].astype(str).map(_clean_team)
        score_raw = df["Home"].astype(str)
        # incluye terminados (N:N) Y programados (-:-) -- antes solo se
        # quedaba con terminados, lo que tiraba toda jornada futura y
        # obligaba al pipeline de limpieza a caer en el archivo legacy
        # per-club (incompleto para ciertos equipos/jornadas).
        score_mask = score_raw.str.match(r"^\d+\s*:\s*\d+$", na=False) | score_raw.str.match(r"^-\s*:\s*-$", na=False)
        mask = score_mask
        mask &= home_clean.str.contains(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", na=False)
        mask &= away_clean.str.contains(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", na=False)
        mask &= ~home_clean.str.contains(r"\d", na=False)
        mask &= ~away_clean.str.contains(r"\d", na=False)
        matches = df[mask].copy()

        for _, r in matches.iterrows():
            home_team = _clean_team(r.get("Home team"))
            away_team = _clean_team(r.get("Result"))
            home_score, away_score, status = _parse_score(r.get("Home"))
            date_raw = str(r.get("Date") or "").strip()
            date = pd.to_datetime(date_raw, errors="coerce", dayfirst=True)
            result_text = str(r.get("Home") or "").strip()
            # los partidos programados no tienen "Match report" todavia, asi
            # que el indice de report_urls solo debe avanzar en terminados
            # (si no, se desalinea y le pega reportes de otro partido).
            if status == "FINISHED":
                report_url = report_urls[report_idx] if report_idx < len(report_urls) else None
                report_idx += 1
            else:
                report_url = None

            rows.append({
                "season_id": season_id,
                "season_label": f"{season_id}/{season_id + 1}",
                "phase": phase,
                "matchday": matchday_idx,
                "date_raw": date_raw,
                "date": date.isoformat() if pd.notna(date) else None,
                "time": str(r.get("Time") or "").strip(),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "result": result_text,
                "status": status,
                "report_url": report_url,
                "source_url": source_url,
            })

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    # limpiar duplicados si aparecieran por el HTML
    raw.drop_duplicates(subset=["season_id", "phase", "matchday", "date_raw", "home_team", "away_team", "result"], inplace=True)
    raw.reset_index(drop=True, inplace=True)
    return raw


def _enrich_matches(matches: pd.DataFrame) -> pd.DataFrame:
    enriched = matches.copy()
    detail_cols = [
        "report_url",
        "attendance",
        "home_system_of_play",
        "away_system_of_play",
        "home_club_rank",
        "away_club_rank",
        "stadium",
        "referee",
        "home_manager",
        "away_manager",
        "home_goal_scorers",
        "away_goal_scorers",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
        "home_substitutions",
        "away_substitutions",
    ]
    for col in detail_cols:
        enriched[col] = None

    club_enrichment = _load_club_enrichment(int(enriched["season_id"].iloc[0]), str(enriched["phase"].iloc[0]))
    if not club_enrichment.empty:
        enriched = enriched.merge(
            club_enrichment,
            on=["matchday", "home_team", "away_team", "home_score", "away_score"],
            how="left",
            suffixes=("", "_club"),
        )
        # date/date_raw/time: el calendario general (gesamtspielplan) pierde la fecha en
        # varias filas por fallas de forward-fill entre celdas combinadas; la pagina de
        # club siempre trae la fecha completa y ya parseada con dayfirst=True, asi que
        # gana cuando esta disponible.
        club_priority_cols = {"date_raw", "date", "time"}
        for col in ["date_raw", "date", "time", "report_url", "attendance", "home_system_of_play", "away_system_of_play", "home_club_rank", "away_club_rank", "club_team", "club_venue", "source_url"]:
            merged_col = f"{col}_club"
            if merged_col in enriched.columns:
                if col in club_priority_cols:
                    enriched[col] = enriched[merged_col].where(enriched[merged_col].notna(), enriched[col]) if col in enriched.columns else enriched[merged_col]
                else:
                    enriched[col] = enriched[col].where(enriched[col].notna(), enriched[merged_col]) if col in enriched.columns else enriched[merged_col]
                enriched.drop(columns=[merged_col], inplace=True)

    for idx, row in enriched[enriched["status"] == "FINISHED"].iterrows():
        url = row.get("report_url")
        if pd.isna(url) or not str(url).strip() or str(url).strip().lower() == "nan":
            continue
        html = _get(str(url).strip())
        details = _parse_detail_page(html)
        for key, value in details.items():
            enriched.at[idx, key] = value
        time.sleep(0.5)

    return enriched


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
            "team": team,
            "played_games": played,
            "won": wins,
            "draw": draws,
            "lost": losses,
            "points": pts,
            "goals_for": gf,
            "goals_against": ga,
            "goal_difference": gf - ga,
            "win_rate": wins / played if played else 0.0,
            "points_per_game": pts / played if played else 0.0,
            "goals_for_per_game": gf / played if played else 0.0,
            "goals_against_per_game": ga / played if played else 0.0,
        })

    standings = pd.DataFrame(out)
    standings.sort_values(["points", "goal_difference", "goals_for", "team"], ascending=[False, False, False, True], inplace=True)
    standings.reset_index(drop=True, inplace=True)
    standings.insert(0, "position", range(1, len(standings) + 1))
    return standings


def build(force: bool = False) -> None:
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    STANDINGS_DIR.mkdir(parents=True, exist_ok=True)

    for page in PAGES:
        season_id = page["season_id"]
        phase = page["phase"]
        url = page["url"]
        raw_path = MATCHES_DIR / f"tm_raw_{phase}_{season_id}.json"
        matches_path = MATCHES_DIR / f"tm_matches_{phase}_{season_id}.csv"
        standings_path = STANDINGS_DIR / f"tm_standings_{phase}_{season_id}.csv"

        if raw_path.exists() and matches_path.exists() and standings_path.exists() and not force:
            print(f"OK cache existe: {phase} {season_id}")
            continue

        html = _get(url)
        matches = _match_rows_from_page(html, season_id=season_id, phase=phase, source_url=url)
        if matches.empty:
            raise RuntimeError(f"No se pudieron parsear partidos para {phase} {season_id}")

        matches = _enrich_matches(matches)

        matches.to_csv(matches_path, index=False)
        standings = _build_standings(matches)
        standings.to_csv(standings_path, index=False)

        raw_path.write_text(matches.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

        print(
            f"Liga MX {phase} {season_id}: rows={len(matches)} finished={int((matches['status']=='FINISHED').sum())} "
            f"teams={len(standings)} -> {matches_path.name}, {standings_path.name}"
        )


if __name__ == "__main__":
    build(force=True)
