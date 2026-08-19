#!/usr/bin/env python3
"""Entrena WinnerPredictor (GB+RF) para una liga, con split cronológico.

Reusa core.train_ensemble.build_training_dataset() -- es funcionalmente idéntico
a FeatureEngineer.create_training_dataset() salvo que ya devuelve las fechas
alineadas fila-a-fila, necesarias para el split cronológico (ver core/eval_utils.py).
No duplica ese loop acá.

Uso:
    python -m src_v2.train_winner_ensemble --data-dir data/cleaned --models-dir models/premier_league_winner_candidate
    python -m src_v2.train_winner_ensemble --data-dir data/ligamx/cleaned --models-dir models/liga_mx
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src_v2.features.feature_engineer import FeatureEngineer  # noqa: E402

from core.train_ensemble import build_training_dataset  # noqa: E402
from src_v2.models.winner_predictor import WinnerPredictor  # noqa: E402


def train_winner_ensemble(data_dir: str, models_dir: str) -> dict:
    fe = FeatureEngineer(data_dir)
    if not fe.load_data():
        raise RuntimeError(f"No se pudo cargar data_dir={data_dir}")

    features_df, outcome_targets, _, dates, _ = build_training_dataset(fe)
    if features_df.empty:
        raise RuntimeError("Dataset de entrenamiento vacío")

    winner = WinnerPredictor(models_dir=models_dir)
    winner.train(features_df, outcome_targets, dates=dates)

    return {
        'n_samples': len(features_df),
        'performance': winner.get_performance(),
    }


def main():
    parser = argparse.ArgumentParser(description="Entrena el ensamble GB+RF (WinnerPredictor) para una liga")
    parser.add_argument("--data-dir", default="data/cleaned")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    result = train_winner_ensemble(args.data_dir, args.models_dir)
    print(f"\nSamples: {result['n_samples']}")
    print(f"Performance: {result['performance']}")


if __name__ == "__main__":
    main()
