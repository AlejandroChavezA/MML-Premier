#!/usr/bin/env python3
"""Fase 1 del plan: ablación de features para Premier, incluyendo Elo como candidato
(no como asunción -- Elo ya está validado en Liga MX pero nunca se midió para Premier,
ver config/leagues/premier_league.yaml `extra_features: []`).

Entrena un OutcomeXGB (hiperparámetros default, sin tunear -- eso es la fase 2) por cada
columna quitada una a la vez, más una corrida quitando el grupo Elo completo, sobre el
mismo holdout cronológico que ya se usó para medir el 49.2% de producción. Cada resultado
se apendea al leaderboard; no se sobreescribe nada.
"""
from pathlib import Path

from core.models.outcome_xgb import OutcomeXGB
from experiments.premier_outcome import (
    CANDIDATES_DIR, evaluate_holdout, load_dataset, log_result,
)

ELO_COLS = ["home_elo", "away_elo", "elo_diff"]


def _xgb_proba_fn(model, fe):
    def _fn(match_row):
        result = model.predict_proba(match_row["home_team"], match_row["away_team"],
                                      match_row["date"], fe)
        if result.get("error"):
            return None
        return result["probabilities"]
    return _fn


def _train_and_eval(name: str, cols, data, tmp_dir: Path):
    model = OutcomeXGB(models_dir=str(tmp_dir))
    model.train(
        data["features_df"][cols][data["train_mask"]],
        data["outcome_targets"][data["train_mask"]],
        dates=data["dates"][data["train_mask"]],
    )
    metrics = evaluate_holdout(_xgb_proba_fn(model, data["fe"]), data["holdout_matches"])
    log_result("ablation", name, {"n_features": len(cols), "dropped": None}, metrics)
    return metrics


def run():
    print("Cargando dataset con Elo como candidato...")
    data = load_dataset(extra_features=["elo"])
    all_cols = list(data["features_df"].columns)
    tmp_dir = CANDIDATES_DIR / "ablation_tmp"

    print(f"\n== Baseline: {len(all_cols)} features (incluye Elo) ==")
    _train_and_eval("baseline_35_features_incl_elo", all_cols, data, tmp_dir)

    print("\n== Quitar grupo Elo completo ==")
    cols_no_elo = [c for c in all_cols if c not in ELO_COLS]
    _train_and_eval("drop_group_elo", cols_no_elo, data, tmp_dir)

    print("\n== Quitar una feature a la vez (32 base + 3 elo) ==")
    for col in all_cols:
        cols = [c for c in all_cols if c != col]
        _train_and_eval(f"drop_{col}", cols, data, tmp_dir)

    print("\nListo -- revisa experiments/premier_outcome_leaderboard.jsonl (phase=ablation).")


if __name__ == "__main__":
    run()
