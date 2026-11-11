#!/usr/bin/env python3
"""Chequeo puntual: la fase 2 usó el feature set 'sin Elo' (drop_group_elo, 45.4% en la
fase 1), pero el MEJOR resultado individual de la fase 1 fue drop_elo_diff (47.0%) --
quedarse con home_elo/away_elo y quitar solo elo_diff. Nunca se corrió el grid de
hiperparámetros+decay con ESE feature set específico. Esto lo cierra.
"""
from core.models.outcome_xgb import OutcomeXGB
from experiments.premier_outcome import CANDIDATES_DIR, evaluate_holdout, load_dataset, log_result
from experiments.premier_outcome_hparams import DECAY_HALF_LIVES, XGB_PRESETS, _decay_weights
from experiments.premier_outcome_ablation import _xgb_proba_fn


def run():
    print("Cargando dataset con Elo, pero dropeando SOLO elo_diff (home_elo/away_elo quedan)...")
    data = load_dataset(extra_features=["elo"])
    cols = [c for c in data["features_df"].columns if c != "elo_diff"]

    train_dates = data["dates"][data["train_mask"]].reset_index(drop=True)
    train_features = data["features_df"][cols][data["train_mask"]].reset_index(drop=True)
    train_targets = data["outcome_targets"][data["train_mask"]].reset_index(drop=True)

    best = None
    for preset_name, params in XGB_PRESETS.items():
        for half_life in DECAY_HALF_LIVES:
            sw = _decay_weights(train_dates, data["cutoff_date"], half_life)
            model = OutcomeXGB(models_dir=str(CANDIDATES_DIR / "combo_check_tmp"))
            model.train(train_features, train_targets, dates=train_dates,
                        xgb_params=params, sample_weight=sw)
            metrics = evaluate_holdout(_xgb_proba_fn(model, data["fe"]), data["holdout_matches"])
            name = f"xgb_noElodiff_{preset_name}_decay{half_life}"
            log_result("combo_check", name, {"preset": preset_name, "decay_half_life": half_life,
                                              "dropped": "elo_diff only"}, metrics)
            if metrics["accuracy"] is not None and (best is None or metrics["accuracy"] > best[1]["accuracy"]):
                best = (name, metrics)

    print("\n== Mejor con este feature set ==")
    print(best[0], best[1])
    print("\nComparar contra: fase 2 mejor (sin elo del todo) = 47.6% (xgb_default_decay365)")
    print("                  fase 5 mejor combinación real    = 49.2% empatado con legacy puro")
    return best


if __name__ == "__main__":
    run()
