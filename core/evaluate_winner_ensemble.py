#!/usr/bin/env python3
"""Backtest del candidato GB+RF (src_v2.models.winner_predictor.WinnerPredictor)
contra lo que ya existe para una liga: el ensamble XGBoost de core/ y el
legacy/baseline que ya evalúa core/evaluate_ensemble.py.

Mismo holdout cronológico para los tres, para que el número sea comparable
(ver core/eval_utils.py y docs/plan_5_ligas_ligamx.md).

Uso:
    python -m core.evaluate_winner_ensemble --league premier_league
    python -m core.evaluate_winner_ensemble --league liga_mx
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src_v2.features.feature_engineer import FeatureEngineer  # noqa: E402

from core.eval_utils import chronological_cutoff  # noqa: E402
from core.evaluate_ensemble import evaluate as evaluate_xgb_ensemble  # noqa: E402
from core.train_ensemble import build_training_dataset  # noqa: E402
from src_v2.models.winner_predictor import WinnerPredictor  # noqa: E402


def _evaluate_winner_ensemble(data_dir: str, holdout_frac: float) -> dict:
    fe = FeatureEngineer(data_dir)
    if not fe.load_data():
        raise RuntimeError(f"No se pudo cargar data_dir={data_dir}")

    all_matches = pd.concat([df[df['status'] == 'FINISHED'] for df in fe.matches_by_season.values()])
    cutoff_date = chronological_cutoff(all_matches['date'], holdout_frac)
    holdout_matches = all_matches[all_matches['date'] >= cutoff_date]

    features_df, outcome_targets, _, dates, _ = build_training_dataset(fe)
    train_mask = (dates < cutoff_date).values

    winner = WinnerPredictor(models_dir="/tmp/winner_ensemble_eval_tmp")
    winner.train(features_df[train_mask], outcome_targets[train_mask], dates=dates[train_mask])

    correct, total = 0, 0
    for _, m in holdout_matches.iterrows():
        result = winner.predict(m['home_team'], m['away_team'], m['date'], fe)
        if result.get('error'):
            continue
        actual = 'LOCAL' if m['home_score'] > m['away_score'] else (
            'VISITANTE' if m['home_score'] < m['away_score'] else 'EMPATE')
        total += 1
        if result['predicted'] == actual:
            correct += 1

    return {
        'holdout_matches': len(holdout_matches),
        'winner_ensemble_accuracy': correct / total if total else 0.0,
        'winner_ensemble_n': total,
    }


def main():
    parser = argparse.ArgumentParser(description="Compara GB+RF vs XGBoost (core/) vs legacy/baseline")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--league", default=None,
                        help="Slug de config/leagues/{league}.yaml -- define data-dir y si hay legacy")
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    run_legacy = not args.skip_legacy

    if args.league:
        from core.league_config import load_league_config
        cfg = load_league_config(args.league)
        if data_dir is None:
            data_dir = str(cfg.data_dir_path)
        if not args.skip_legacy:
            run_legacy = cfg.has_legacy_baseline

    if data_dir is None:
        data_dir = "data/cleaned"

    xgb_result = evaluate_xgb_ensemble(data_dir, args.holdout_frac, run_legacy=run_legacy)
    winner_result = _evaluate_winner_ensemble(data_dir, args.holdout_frac)

    print(f"\n{'Contendiente':<30} {'Accuracy':<12} {'N':<6}")
    print("-" * 50)
    print(f"{'XGBoost (core/)':<30} {xgb_result['ensemble_accuracy']:<12.1%} {xgb_result['ensemble_n']:<6}")
    print(f"{'GB+RF (src_v2, nuevo)':<30} {winner_result['winner_ensemble_accuracy']:<12.1%} {winner_result['winner_ensemble_n']:<6}")
    if xgb_result.get('legacy_model'):
        print(f"{'Legacy (' + xgb_result['legacy_model'] + ')':<30} {xgb_result['legacy_accuracy']:<12.1%} {xgb_result['legacy_n']:<6}")
        reference_acc = xgb_result['legacy_accuracy']
        reference_label = xgb_result['legacy_model']
    else:
        print(f"{'Baseline naive (siempre local)':<30} {xgb_result['baseline_accuracy']:<12.1%} {'-':<6}")
        reference_acc = xgb_result['baseline_accuracy']
        reference_label = 'baseline naive'

    best_new = max(
        ('XGBoost (core/)', xgb_result['ensemble_accuracy']),
        ('GB+RF (src_v2)', winner_result['winner_ensemble_accuracy']),
        key=lambda x: x[1],
    )
    print(f"\nMejor contendiente nuevo: {best_new[0]} ({best_new[1]:.1%})")
    if best_new[1] >= reference_acc:
        print(f"=> Supera a {reference_label} ({reference_acc:.1%}): candidato a promover.")
    else:
        print(f"=> Ninguno supera a {reference_label} ({reference_acc:.1%}) todavía.")


if __name__ == "__main__":
    main()
