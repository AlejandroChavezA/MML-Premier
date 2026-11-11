#!/usr/bin/env python3
"""Fase 5 del plan: blend entre familias de modelos.

Combina, por partido del holdout, las probabilidades de 4 fuentes distintas:
- OutcomeXGB (mejor config de la fase 2: default + decay 365 días)
- WinnerPredictor (mejor de la fase 3: calibrado, decay 180 días)
- Legacy GB en producción (gradient_boosting_v2, vía menu.predictor -- NO se reentrena,
  es la referencia real que ya corre hoy)
- DixonColesModel.outcome_probs() (familia Poisson, nunca usada para 1X2 en este repo --
  entrenado solo con partidos anteriores al cutoff, igual que los demás)

Grid de pesos en pasos de 0.25 que suman 1 entre las 4 fuentes (35 combinaciones) --
el "probar combinaciones" explícito que pidió el usuario.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

from core.models.dixon_coles import DixonColesModel
from core.models.outcome_xgb import OutcomeXGB
from src_v2.models.winner_predictor import WinnerPredictor
from experiments.premier_outcome import (
    CANDIDATES_DIR, _CLASSES, _actual_label, load_dataset, log_result, score_predictions,
)
from experiments.premier_outcome_hparams import _decay_weights, XGB_PRESETS

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STEP = 0.25
SOURCES = ["xgb", "winner", "legacy", "dc"]


def _weight_grid():
    n_steps = int(1 / STEP)
    combos = []
    for combo in itertools.product(range(n_steps + 1), repeat=len(SOURCES)):
        if sum(combo) == n_steps:
            combos.append({s: c * STEP for s, c in zip(SOURCES, combo)})
    return combos


def _collect_source_probas(data):
    train_dates = data["dates"][data["train_mask"]].reset_index(drop=True)
    train_features = data["features_df"][data["train_mask"]].reset_index(drop=True)
    train_targets = data["outcome_targets"][data["train_mask"]].reset_index(drop=True)

    print("Entrenando OutcomeXGB (default, decay365)...")
    sw_xgb = _decay_weights(train_dates, data["cutoff_date"], 365)
    xgb_model = OutcomeXGB(models_dir=str(CANDIDATES_DIR / "blend_xgb"))
    xgb_model.train(train_features, train_targets, dates=train_dates,
                     xgb_params=XGB_PRESETS["default"], sample_weight=sw_xgb)

    print("Entrenando WinnerPredictor (calibrado, decay180)...")
    sw_winner = _decay_weights(train_dates, data["cutoff_date"], 180)
    winner_model = WinnerPredictor(models_dir=str(CANDIDATES_DIR / "blend_winner"))
    winner_model.train(train_features, train_targets, dates=train_dates,
                        sample_weight=sw_winner, calibrate=True)

    print("Entrenando Dixon-Coles (solo con partidos antes del cutoff)...")
    dc_train = data["all_matches"][data["all_matches"]["date"] < data["cutoff_date"]]
    dc_model = DixonColesModel()
    dc_model.fit(dc_train)

    print("Inicializando legacy (PredictionMenu, gradient_boosting_v2, sin reentrenar)...")
    from menu_interface import PredictionMenu  # noqa: E402
    menu = PredictionMenu()
    if not menu.initialize():
        raise RuntimeError("No se pudo inicializar el legacy PredictionMenu")

    print("Prediciendo holdout con las 4 fuentes...")
    rows = []
    for _, m in data["holdout_matches"].iterrows():
        entry = {"actual": _actual_label(m["home_score"], m["away_score"])}

        r = xgb_model.predict_proba(m["home_team"], m["away_team"], m["date"], data["fe"])
        entry["xgb"] = None if r.get("error") else r["probabilities"]

        r = winner_model.predict(m["home_team"], m["away_team"], m["date"], data["fe"])
        entry["winner"] = None if r.get("error") else r["probabilities"]

        lam_h, lam_a = dc_model.predict(m["home_team"], m["away_team"])
        dc_probs = dc_model.outcome_probs(lam_h, lam_a)
        entry["dc"] = {"LOCAL": dc_probs["home"], "EMPATE": dc_probs["draw"],
                        "VISITANTE": dc_probs["away"]}

        r = menu.predictor.predict_match(m["home_team"], m["away_team"], m["date"], menu.current_model)
        entry["legacy"] = None if r.get("error") else r["probabilities"]

        rows.append(entry)
    return rows


def run():
    print("Cargando dataset SIN Elo...")
    data = load_dataset(extra_features=None)
    rows = _collect_source_probas(data)
    usable = [r for r in rows if all(r[s] is not None for s in SOURCES)]
    print(f"{len(usable)}/{len(rows)} partidos con las 4 fuentes disponibles.")

    best = None
    for weights in _weight_grid():
        actual_labels, proba_rows = [], []
        for r in usable:
            blended = {c: sum(weights[s] * r[s][c] for s in SOURCES) for c in _CLASSES}
            actual_labels.append(r["actual"])
            proba_rows.append(blended)
        metrics = score_predictions(actual_labels, proba_rows)
        name = "blend_" + "_".join(f"{s}{weights[s]:.2f}" for s in SOURCES)
        log_result("blend", name, {"weights": weights}, metrics)
        if metrics["accuracy"] is not None and (best is None or metrics["accuracy"] > best[1]["accuracy"]):
            best = (name, metrics, weights)

    print("\n== Mejor blend ==")
    print(best[0], best[1], best[2])
    return best


if __name__ == "__main__":
    run()
