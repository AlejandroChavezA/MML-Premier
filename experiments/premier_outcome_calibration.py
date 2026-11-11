#!/usr/bin/env python3
"""Fase 3 del plan: calibración.

OutcomeXGB ya calibra (CalibratedClassifierCV isotonic, cv=5) dentro de su propio
`.train()` -- todos los números de la fase 2 ya la incluyen, no hay nada nuevo que medir
ahí. El hueco real es WinnerPredictor (GB+RF), que no calibraba hasta el cambio de este
plan (`calibrate=True` opcional, default False). Esta fase compara, con el mejor decay de
la fase 2 (180 días), calibrar vs no calibrar -- midiendo accuracy pero sobre todo
log-loss/Brier (la calibración no debería cambiar mucho el accuracy "en seco", shape de
probabilidades sí).
"""
from src_v2.models.winner_predictor import WinnerPredictor
from experiments.premier_outcome import CANDIDATES_DIR, evaluate_holdout, load_dataset, log_result
from experiments.premier_outcome_hparams import _decay_weights


def _winner_proba_fn(model, fe):
    def _fn(match_row):
        result = model.predict(match_row["home_team"], match_row["away_team"],
                                match_row["date"], fe)
        return None if result.get("error") else result["probabilities"]
    return _fn


def run(best_decay_half_life: int = 180):
    print("Cargando dataset SIN Elo (mismo set que fases 1-2)...")
    data = load_dataset(extra_features=None)
    train_dates = data["dates"][data["train_mask"]].reset_index(drop=True)
    train_features = data["features_df"][data["train_mask"]].reset_index(drop=True)
    train_targets = data["outcome_targets"][data["train_mask"]].reset_index(drop=True)
    sw = _decay_weights(train_dates, data["cutoff_date"], best_decay_half_life)

    results = {}
    for calibrate in (False, True):
        model = WinnerPredictor(models_dir=str(CANDIDATES_DIR / f"calib_tmp_{calibrate}"))
        model.train(train_features, train_targets, dates=train_dates, sample_weight=sw,
                    calibrate=calibrate)
        metrics = evaluate_holdout(_winner_proba_fn(model, data["fe"]), data["holdout_matches"])
        name = f"winner_decay{best_decay_half_life}_calibrate{calibrate}"
        log_result("calibration", name, {"decay_half_life": best_decay_half_life,
                                          "calibrate": calibrate}, metrics)
        results[calibrate] = metrics

    print("\n== Comparación ==")
    print("Sin calibrar:", results[False])
    print("Calibrado:   ", results[True])
    return results


if __name__ == "__main__":
    run()
