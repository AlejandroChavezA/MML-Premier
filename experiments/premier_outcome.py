#!/usr/bin/env python3
"""Harness de experimentos para mejorar 1X2 de Premier League sin tocar producción.

Reusa core.train_ensemble.build_training_dataset + core.eval_utils.chronological_cutoff para
que todas las fases de experimentación queden en la MISMA partición train/holdout
(holdout_frac=0.15) que ya se usó para medir el 49.2% honesto del modelo en producción
(gradient_boosting_v2) vía `python -m core.evaluate_winner_ensemble --league premier_league`
(ver CLAUDE.md, sección "The '66.1%' figure..."). Todo resultado se compara contra esa cifra,
nunca contra una medición nueva no comparable.

No escribe nada en `models/` -- los artefactos que vale la pena conservar van a
`experiments/candidates/`, y cada corrida (ganadora o no) se apendea a
`experiments/premier_outcome_leaderboard.jsonl` para no perder el historial de intentos.

Uso:
    python -m experiments.premier_outcome seed         # siembra el leaderboard con los numeros ya medidos
    python -m experiments.premier_outcome ablation      # fase 1
    python -m experiments.premier_outcome report        # top-5 actual del leaderboard
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src_v2.features.feature_engineer import FeatureEngineer  # noqa: E402

from core.eval_utils import chronological_cutoff  # noqa: E402
from core.train_ensemble import build_training_dataset  # noqa: E402
from core.models.outcome_xgb import OutcomeXGB  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).parent
LEADERBOARD_PATH = EXPERIMENTS_DIR / "premier_outcome_leaderboard.jsonl"
CANDIDATES_DIR = EXPERIMENTS_DIR / "candidates"
DATA_DIR = "data/cleaned"
HOLDOUT_FRAC = 0.15  # mismo corte que core/evaluate_ensemble.py y core/evaluate_winner_ensemble.py

_CLASSES = ['VISITANTE', 'EMPATE', 'LOCAL']  # mismo orden/mapeo que OutcomeXGB.RESULT_MAP


# ---------------------------------------------------------------------------
# Dataset compartido (mismo corte cronológico para todas las fases)
# ---------------------------------------------------------------------------

def load_dataset(extra_features: Optional[List[str]] = None) -> Dict:
    """Carga Premier con el mismo split cronológico (holdout_frac=0.15) que ya se usó
    para medir el 49.2% de producción. `extra_features=["elo"]` activa el feature opcional
    que hoy está apagado para Premier (ver config/leagues/premier_league.yaml)."""
    fe = FeatureEngineer(DATA_DIR, extra_features=extra_features)
    if not fe.load_data():
        raise RuntimeError(f"No se pudo cargar data_dir={DATA_DIR}")

    features_df, outcome_targets, goals_targets, dates, all_matches = build_training_dataset(fe)
    # Cutoff sobre TODOS los FINISHED (no solo los que sobrevivieron a create_match_features),
    # igual que core/evaluate_ensemble.py -- así el corte de fecha es idéntico al ya medido.
    cutoff_date = chronological_cutoff(all_matches['date'], HOLDOUT_FRAC)
    train_mask = (dates < cutoff_date).values
    holdout_matches = all_matches[all_matches['date'] >= cutoff_date]

    return {
        "fe": fe,
        "features_df": features_df,
        "outcome_targets": outcome_targets,
        "dates": dates,
        "all_matches": all_matches,
        "cutoff_date": cutoff_date,
        "train_mask": train_mask,
        "holdout_matches": holdout_matches,
    }


# ---------------------------------------------------------------------------
# Métricas (iguales para todas las fases, así todo es comparable)
# ---------------------------------------------------------------------------

def _actual_label(home_score: float, away_score: float) -> str:
    if home_score > away_score:
        return 'LOCAL'
    if home_score < away_score:
        return 'VISITANTE'
    return 'EMPATE'


def score_predictions(actual_labels: List[str], proba_rows: List[Dict[str, float]]) -> Dict:
    """Accuracy, log-loss, Brier (multiclase, promedio one-vs-rest) y recall de EMPATE."""
    n = len(actual_labels)
    if n == 0:
        return {"n": 0, "accuracy": None, "log_loss": None, "brier": None, "draw_recall": None}

    preds = [max(row, key=row.get) for row in proba_rows]
    correct = sum(1 for a, p in zip(actual_labels, preds) if a == p)

    y_true_idx = [_CLASSES.index(a) for a in actual_labels]
    y_proba = np.array([[max(row.get(c, 1e-6), 1e-6) for c in _CLASSES] for row in proba_rows])
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # por si no suman exactamente 1
    ll = float(log_loss(y_true_idx, y_proba, labels=list(range(len(_CLASSES)))))

    one_hot = np.eye(len(_CLASSES))[y_true_idx]
    brier = float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))

    draw_idx = [i for i, a in enumerate(actual_labels) if a == 'EMPATE']
    draw_recall = (sum(1 for i in draw_idx if preds[i] == 'EMPATE') / len(draw_idx)) if draw_idx else None

    return {"n": n, "accuracy": correct / n, "log_loss": ll, "brier": brier, "draw_recall": draw_recall}


def evaluate_holdout(proba_fn: Callable[[pd.Series], Optional[Dict[str, float]]],
                      holdout_matches: pd.DataFrame) -> Dict:
    """`proba_fn(match_row) -> {'LOCAL':p,'EMPATE':p,'VISITANTE':p} | None` (None = se salta)."""
    actual_labels, proba_rows = [], []
    skipped = 0
    for _, m in holdout_matches.iterrows():
        proba = proba_fn(m)
        if proba is None:
            skipped += 1
            continue
        actual_labels.append(_actual_label(m['home_score'], m['away_score']))
        proba_rows.append(proba)
    metrics = score_predictions(actual_labels, proba_rows)
    metrics["skipped"] = skipped
    return metrics


# ---------------------------------------------------------------------------
# Leaderboard (append-only -- nunca se pierde un intento anterior)
# ---------------------------------------------------------------------------

def log_result(phase: str, name: str, config: Dict, metrics: Dict) -> Dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "name": name,
        "config": config,
        "metrics": metrics,
    }
    with open(LEADERBOARD_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    acc = metrics.get("accuracy")
    ll = metrics.get("log_loss")
    dr = metrics.get("draw_recall")
    acc_s = f"{acc:.1%}" if acc is not None else "n/a"
    ll_s = f"{ll:.4f}" if ll is not None else "n/a"
    dr_s = f"{dr:.1%}" if dr is not None else "n/a"
    print(f"[{phase}] {name}: accuracy={acc_s} log_loss={ll_s} draw_recall={dr_s} n={metrics.get('n')}")
    return entry


def read_leaderboard() -> List[Dict]:
    if not LEADERBOARD_PATH.exists():
        return []
    with open(LEADERBOARD_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Siembra: los números ya medidos y validados en la sesión anterior, para que
# toda mejora futura se compare contra algo real, no contra cero.
# ---------------------------------------------------------------------------

def seed_baseline():
    baseline_rows = [
        ("legacy_gradient_boosting_v2", {"accuracy": 0.492, "n": 185}),
        ("core_xgboost", {"accuracy": 0.454, "n": 185}),
        ("gb_rf_winner_predictor_sin_tuning", {"accuracy": 0.438, "n": 185}),
        ("baseline_naive_siempre_local", {"accuracy": 0.411, "n": 185}),
    ]
    for name, metrics in baseline_rows:
        log_result("baseline", name, {"note": "medido 2026-11-05 via core.evaluate_winner_ensemble"}, metrics)


def cmd_report(top_n: int = 5):
    rows = read_leaderboard()
    if not rows:
        print("Leaderboard vacío -- corré `seed` primero.")
        return
    with_acc = [r for r in rows if r["metrics"].get("accuracy") is not None]
    by_acc = sorted(with_acc, key=lambda r: r["metrics"]["accuracy"], reverse=True)[:top_n]
    with_ll = [r for r in rows if r["metrics"].get("log_loss") is not None]
    by_ll = sorted(with_ll, key=lambda r: r["metrics"]["log_loss"])[:top_n]

    print(f"\nTop {top_n} por accuracy:")
    for r in by_acc:
        print(f"  {r['metrics']['accuracy']:.1%}  [{r['phase']}] {r['name']}")

    print(f"\nTop {top_n} por log-loss (menor = mejor calibrado):")
    for r in by_ll:
        print(f"  {r['metrics']['log_loss']:.4f}  [{r['phase']}] {r['name']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["seed", "ablation", "report"])
    args = parser.parse_args()

    if args.phase == "seed":
        seed_baseline()
    elif args.phase == "ablation":
        from experiments import premier_outcome_ablation
        premier_outcome_ablation.run()
    elif args.phase == "report":
        cmd_report()


if __name__ == "__main__":
    main()
