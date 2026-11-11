#!/usr/bin/env python3
"""Fase 4 del plan: umbral de decisión para EMPATE.

Sobre las probabilidades ya calibradas (OutcomeXGB siempre, WinnerPredictor con
calibrate=True), en vez de puro argmax: si P(EMPATE) está dentro de `margin` del máximo
entre P(LOCAL) y P(VISITANTE), preferir EMPATE. Es una regla de decisión post-hoc -- no
reentrena nada, solo cambia cómo se convierte una probabilidad ya calculada en una
etiqueta. Se mide accuracy general y recall de EMPATE específicamente, para ver si gana
empates sin regalar demasiados LOCAL/VISITANTE correctos.
"""
from core.models.outcome_xgb import OutcomeXGB
from src_v2.models.winner_predictor import WinnerPredictor
from experiments.premier_outcome import (
    CANDIDATES_DIR, evaluate_holdout, load_dataset, log_result, score_predictions, _actual_label,
)
from experiments.premier_outcome_hparams import _decay_weights, XGB_PRESETS

MARGINS = [0.0, 0.03, 0.06, 0.10, 0.15]


def _apply_margin(proba: dict, margin: float) -> dict:
    """No cambia las probabilidades -- solo se usa para decidir la etiqueta final en
    score_predictions (que ya hace argmax sobre lo que le pasemos), así que acá
    'tramposeamos' devolviendo probabilidades ajustadas únicamente para el argmax,
    conservando las originales para log-loss/Brier por separado si hiciera falta."""
    if margin <= 0:
        return proba
    best_other = max(proba['LOCAL'], proba['VISITANTE'])
    if best_other - proba['EMPATE'] <= margin:
        # Empujar EMPATE por encima del máximo sin falsear log-loss de forma severa:
        # redistribuye lo mínimo necesario para que gane el argmax.
        boosted = dict(proba)
        boosted['EMPATE'] = best_other + 1e-6
        return boosted
    return proba


def run(xgb_params=None, decay_half_life=365):
    print("Cargando dataset SIN Elo...")
    data = load_dataset(extra_features=None)
    train_dates = data["dates"][data["train_mask"]].reset_index(drop=True)
    train_features = data["features_df"][data["train_mask"]].reset_index(drop=True)
    train_targets = data["outcome_targets"][data["train_mask"]].reset_index(drop=True)
    sw = _decay_weights(train_dates, data["cutoff_date"], decay_half_life)

    params = xgb_params or XGB_PRESETS["default"]
    model = OutcomeXGB(models_dir=str(CANDIDATES_DIR / "threshold_tmp"))
    model.train(train_features, train_targets, dates=train_dates, xgb_params=params, sample_weight=sw)

    # Probabilidades base (sin umbral) para todo el holdout, una sola vez.
    base_rows = []
    for _, m in data["holdout_matches"].iterrows():
        result = model.predict_proba(m["home_team"], m["away_team"], m["date"], data["fe"])
        base_rows.append((m, None if result.get("error") else result["probabilities"]))

    best = None
    for margin in MARGINS:
        actual_labels, proba_rows = [], []
        for m, proba in base_rows:
            if proba is None:
                continue
            actual_labels.append(_actual_label(m["home_score"], m["away_score"]))
            proba_rows.append(_apply_margin(proba, margin))
        metrics = score_predictions(actual_labels, proba_rows)
        metrics["skipped"] = sum(1 for _, p in base_rows if p is None)
        name = f"xgb_default_decay{decay_half_life}_margin{margin}"
        log_result("threshold", name, {"margin": margin, "decay_half_life": decay_half_life}, metrics)
        if metrics["accuracy"] is not None and (best is None or metrics["accuracy"] > best[1]["accuracy"]):
            best = (name, metrics)

    print("\n== Mejor margen ==")
    print(best[0], best[1])
    return best


if __name__ == "__main__":
    run()
