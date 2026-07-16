"""Construir baseline de equipos recien ascendidos y factor de escala Championship->PL.

Genera data/promoted_baseline.json con:
  - "generic_baseline": promedio de la 1a temporada PL real de 6 equipos ascendidos
    (Burnley, Luton, Sheffield Utd, Ipswich, Leicester, Southampton). Usado como
    fallback para cualquier equipo sin PL ni Championship.
  - "scaling_factors": factor k = media(PL_primer_anio / ELC_temporada_ascenso) para
    puntos, win_rate, gf/partido y ga/partido, calibrado con 5 equipos reales:
      * Ipswich, Leicester, Southampton  (ELC 2023 -> PL 2024-25)
      * Leeds, Sunderland               (ELC 2024 -> PL 2025-26)
    (Burnley se excluye: tiene historia PL previa en nuestra ventana; ELC 2022 restringido)
  - "team_seeds": para cada equipo cero-historia, su seed PL = stats ELC de su temporada
    de ascenso x factores_k. Se calcula bajo demanda en feature_engineering; aqui solo
    dejamos los factores y el baseline.

Se ejecuta una vez; las predicciones leen el JSON.
"""
import os
import json
import pandas as pd

BASE = os.path.dirname(__file__)
DATA_CLEANED = os.path.join(BASE, "..", "data", "cleaned")
CACHE_PATH = os.path.join(BASE, "..", "data", "championship_seeds.json")
OUT_PATH = os.path.join(BASE, "..", "data", "promoted_baseline.json")

# Mapeo equipo -> (temporada PL de su 1a temporada, archivo standings, season ELC ascenso)
# Cohorte PL 2024-25 (archivo standings 2024) <- ELC 2023
COHORT_2024_25 = {
    "Ipswich Town FC": 2023,
    "Leicester City FC": 2023,
    "Southampton FC": 2023,
}
# Cohorte PL 2025-26 (archivo standings 2025) <- ELC 2024
COHORT_2025_26 = {
    "Leeds United FC": 2024,
    "Sunderland AFC": 2024,
}
# Para el baseline generico n=6 (1a temporada PL real)
BASELINE_FIRST = {
    "Burnley FC": ("2023", 2023),
    "Luton Town FC": ("2023", 2023),
    "Sheffield United FC": ("2023", 2023),
    "Ipswich Town FC": ("2024", 2023),
    "Leicester City FC": ("2024", 2023),
    "Southampton FC": ("2024", 2023),
}


def _load_standings(year: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_CLEANED, f"standings_{year}_cleaned.csv"))


def _champ_row(team: str, elc_season: int) -> dict:
    with open(CACHE_PATH) as f:
        seeds = json.load(f)
    return seeds["seasons"].get(str(elc_season), {}).get(team)


def _pl_first_season_stats(team: str, standings_year: str) -> dict:
    df = _load_standings(standings_year)
    row = df[df["team"] == team]
    if row.empty:
        return None
    r = row.iloc[0]
    pg = max(1, int(r["played_games"]))
    return {
        "points": float(r["points"]),
        "position": int(r["position"]),
        "win_rate": float(r["win_rate"]),
        "goals_for_per_game": float(r["goals_for_per_game"]),
        "goals_against_per_game": float(r["goals_against_per_game"]),
        "played_games": pg,
    }


def build(force: bool = False) -> dict:
    if os.path.exists(OUT_PATH) and not force:
        print(f"Baseline ya existe: {OUT_PATH}")
        with open(OUT_PATH) as f:
            return json.load(f)

    # --- Baseline generico n=6 ---
    agg = {"points_per_game": [], "win_rate": [], "gf_pg": [], "ga_pg": [], "position": []}
    for team, (syear, _) in BASELINE_FIRST.items():
        s = _pl_first_season_stats(team, syear)
        if not s:
            continue
        agg["points_per_game"].append(s["points"] / s["played_games"])
        agg["win_rate"].append(s["win_rate"])
        agg["gf_pg"].append(s["goals_for_per_game"])
        agg["ga_pg"].append(s["goals_against_per_game"])
        agg["position"].append(s["position"])

    generic = {
        "n_teams": len(agg["points_per_game"]),
        "points_per_game": sum(agg["points_per_game"]) / len(agg["points_per_game"]),
        "win_rate": sum(agg["win_rate"]) / len(agg["win_rate"]),
        "goals_for_per_game": sum(agg["gf_pg"]) / len(agg["gf_pg"]),
        "goals_against_per_game": sum(agg["ga_pg"]) / len(agg["ga_pg"]),
        "avg_position": sum(agg["position"]) / len(agg["position"]),
    }

    # --- Factor de escala k (5 equipos) ---
    calib = {}
    for team, elc_season in {**COHORT_2024_25, **COHORT_2025_26}.items():
        standings_year = "2024" if elc_season == 2023 else "2025"
        pl = _pl_first_season_stats(team, standings_year)
        ch = _champ_row(team, elc_season)
        if not pl or not ch:
            continue
        ch_pg = max(1, ch["played_games"])
        ch_ppg = ch["points"] / ch_pg
        ch_gf_pg = ch["goals_for"] / ch_pg
        ch_ga_pg = ch["goals_against"] / ch_pg
        ch_wr = ch["won"] / ch_pg
        calib[team] = {
            "k_points": (pl["points"] / pl["played_games"]) / ch_ppg,
            "k_winrate": pl["win_rate"] / ch_wr,
            "k_gf": pl["goals_for_per_game"] / ch_gf_pg,
            "k_ga": pl["goals_against_per_game"] / ch_ga_pg,
        }

    n = len(calib)
    scaling = {
        "n_teams": n,
        "k_points": sum(c["k_points"] for c in calib.values()) / n,
        "k_winrate": sum(c["k_winrate"] for c in calib.values()) / n,
        "k_gf": sum(c["k_gf"] for c in calib.values()) / n,
        "k_ga": sum(c["k_ga"] for c in calib.values()) / n,
    }

    out = {
        "generic_baseline": generic,
        "scaling_factors": scaling,
        "calibration_teams": calib,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Baseline guardado: {OUT_PATH}")
    print(f"  n baseline={generic['n_teams']} | n calibracion={n}")
    print(f"  k_points={scaling['k_points']:.3f} k_winrate={scaling['k_winrate']:.3f} "
          f"k_gf={scaling['k_gf']:.3f} k_ga={scaling['k_ga']:.3f}")
    print(f"  generic ppg={generic['points_per_game']:.2f} win_rate={generic['win_rate']:.3f} "
          f"avg_pos={generic['avg_position']:.1f}")
    return out


def team_seed(team: str, elc_season: int) -> dict:
    """Seed PL escalado para un equipo cero-historia a partir de su temporada ELC.

    Devuelve None si no hay datos de Championship para ese equipo/temporada.
    """
    with open(OUT_PATH) as f:
        base = json.load(f)
    ch = _champ_row(team, elc_season)
    if not ch:
        return None
    s = base["scaling_factors"]
    ch_pg = max(1, ch["played_games"])
    ch_ppg = ch["points"] / ch_pg
    ch_gf_pg = ch["goals_for"] / ch_pg
    ch_ga_pg = ch["goals_against"] / ch_pg
    ch_wr = ch["won"] / ch_pg

    ppg = ch_ppg * s["k_points"]
    gf_pg = ch_gf_pg * s["k_gf"]
    ga_pg = ch_ga_pg * s["k_ga"]
    wr = min(1.0, ch_wr * s["k_winrate"])

    return {
        "position": round(base["generic_baseline"]["avg_position"]),
        "points_per_game": ppg,
        "win_rate": wr,
        "goals_for_per_game": gf_pg,
        "goals_against_per_game": ga_pg,
        "source": f"ELC {elc_season} escalado",
        "raw_championship": {
            "position": ch["position"],
            "points": ch["points"],
            "won": ch["won"],
            "lost": ch["lost"],
        },
    }


if __name__ == "__main__":
    build(force=True)
