#!/usr/bin/env python3
"""Fase 2 del plan: hiperparámetros + recencia (decay), juntos.

Van juntos porque el peso por recencia cambia el paisaje de la pérdida -- tunear
hiperparámetros antes del decay optimizaría para el problema equivocado (ver plan).
Usa el set de features de la fase 1 (32, SIN Elo -- la ablación no mostró una ganancia
robusta para Premier, a diferencia de Liga MX).

Grid acotado (no Bayesiano completo) por costo: 3 presets de capacidad x 3 half-life de
decay, para OutcomeXGB y para el componente GB de WinnerPredictor por separado.
"""
import numpy as np
import pandas as pd

from core.models.outcome_xgb import OutcomeXGB
from src_v2.models.winner_predictor import WinnerPredictor
from experiments.premier_outcome import (
    CANDIDATES_DIR, evaluate_holdout, load_dataset, log_result,
)

XGB_PRESETS = {
    "default": dict(n_estimators=60, max_depth=3, learning_rate=0.02, subsample=0.6,
                     colsample_bytree=0.6, min_child_weight=20, reg_alpha=2.0, reg_lambda=4.0),
    "lighter_reg": dict(n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.7,
                         colsample_bytree=0.7, min_child_weight=10, reg_alpha=0.5, reg_lambda=1.0),
    "higher_capacity": dict(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.5, reg_lambda=1.0),
}

DECAY_HALF_LIVES = [None, 180, 365]  # días; None = sin ponderar por recencia


def _decay_weights(dates: pd.Series, cutoff_date, half_life_days):
    if half_life_days is None:
        return None
    age_days = (cutoff_date - pd.to_datetime(dates)).dt.total_seconds() / 86400.0
    return np.power(0.5, age_days / half_life_days)


def _xgb_proba_fn(model, fe):
    def _fn(match_row):
        result = model.predict_proba(match_row["home_team"], match_row["away_team"],
                                      match_row["date"], fe)
        return None if result.get("error") else result["probabilities"]
    return _fn


def _winner_proba_fn(model, fe):
    def _fn(match_row):
        result = model.predict(match_row["home_team"], match_row["away_team"],
                                match_row["date"], fe)
        return None if result.get("error") else result["probabilities"]
    return _fn


def run():
    print("Cargando dataset SIN Elo (feature set elegido en fase 1)...")
    data = load_dataset(extra_features=None)
    train_dates = data["dates"][data["train_mask"]].reset_index(drop=True)
    train_features = data["features_df"][data["train_mask"]].reset_index(drop=True)
    train_targets = data["outcome_targets"][data["train_mask"]].reset_index(drop=True)

    best = {"xgb": None, "winner": None}

    print("\n== OutcomeXGB: preset x decay ==")
    for preset_name, params in XGB_PRESETS.items():
        for half_life in DECAY_HALF_LIVES:
            sw = _decay_weights(train_dates, data["cutoff_date"], half_life)
            model = OutcomeXGB(models_dir=str(CANDIDATES_DIR / "hparams_tmp"))
            model.train(train_features, train_targets, dates=train_dates,
                        xgb_params=params, sample_weight=sw)
            metrics = evaluate_holdout(_xgb_proba_fn(model, data["fe"]), data["holdout_matches"])
            name = f"xgb_{preset_name}_decay{half_life}"
            log_result("hparam_decay", name, {"preset": preset_name, "decay_half_life": half_life,
                                               "params": params}, metrics)
            if metrics["accuracy"] is not None and (
                best["xgb"] is None or metrics["accuracy"] > best["xgb"][1]["accuracy"]):
                best["xgb"] = (name, metrics, params, half_life)

    print("\n== WinnerPredictor (GB+RF): decay only (hiperparámetros fijos del ensamble) ==")
    for half_life in DECAY_HALF_LIVES:
        sw = _decay_weights(train_dates, data["cutoff_date"], half_life)
        model = WinnerPredictor(models_dir=str(CANDIDATES_DIR / "hparams_tmp_winner"))
        model.train(train_features, train_targets, dates=train_dates, sample_weight=sw)
        metrics = evaluate_holdout(_winner_proba_fn(model, data["fe"]), data["holdout_matches"])
        name = f"winner_decay{half_life}"
        log_result("hparam_decay", name, {"decay_half_life": half_life}, metrics)
        if metrics["accuracy"] is not None and (
            best["winner"] is None or metrics["accuracy"] > best["winner"][1]["accuracy"]):
            best["winner"] = (name, metrics, half_life)

    print("\n== Mejor de cada familia ==")
    print("OutcomeXGB:", best["xgb"][0], best["xgb"][1])
    print("WinnerPredictor:", best["winner"][0], best["winner"][1])
    return best


if __name__ == "__main__":
    run()
