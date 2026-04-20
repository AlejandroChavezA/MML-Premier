#!/usr/bin/env python3
"""
Train v2
=======
Script simplificado para entrenar modelos.

Uso:
    python train.py
"""

import sys
import pandas as pd
from pathlib import Path

# Add parent to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.features.feature_engineer import get_feature_engineer
from src_v2.features.competitiveness import get_competitiveness
from src_v2.models.winner_predictor import get_winner_predictor
from src_v2.models.goals_predictor import get_goals_predictor


def main():
    """Entrenar todos los modelos"""
    print("🏋️ ENTRENAMIENTO v2")
    print("=" * 50)
    
    # 1. Load data
    print("\n[1/4] Cargando datos...")
    fe = get_feature_engineer("data/cleaned")
    if not fe.load_data():
        print("❌ Error cargando datos")
        return 1
    
    # 2. Competitiveness
    comp = get_competitiveness("data/cleaned")
    comp.print_summary()
    
    # 3. Create dataset
    print("\n[2/4] Creando dataset...")
    features_df, targets_df = fe.create_training_dataset()
    
    if features_df.empty:
        print("❌ Dataset vacío")
        return 1
    
    print(f"  Samples: {len(features_df)}, Features: {len(features_df.columns)}")
    
    # Distribucion de targets
    print("\n  Target distribution:")
    for label, code in [('VISITANTE', 0), ('EMPATE', 1), ('LOCAL', 2)]:
        count = (targets_df == code).sum()
        pct = count / len(targets_df) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")
    
    # 4. Train winner model
    print("\n[3/4] Entrenando WinnerPredictor...")
    winner = get_winner_predictor("models")
    winner.train(features_df, targets_df, use_time_series=False)
    
    perf = winner.get_performance()
    print(f"\n  Results:")
    print(f"    Train: {perf['train']:.1%}")
    print(f"    Test:  {perf['test']:.1%}")
    print(f"    CV:    {perf['cv_mean']:.1%} ± {perf['cv_std']:.1%}")
    
    # Check overfitting
    gap = perf['train'] - perf['test']
    print(f"\n  Gap: {gap:.1%}")
    if gap > 0.25:
        print("  ⚠️  OVERFITTING! reducer complejidad")
    
    # 5. Train goals model
    print("\n[4/4] Entrenando GoalsPredictor...")
    goals = get_goals_predictor("models")
    
    # Create goals targets from original match data
    all_matches = pd.concat([
        fe.matches_2023[fe.matches_2023['status'] == 'FINISHED'],
        fe.matches_2024[fe.matches_2024['status'] == 'FINISHED'],
        fe.matches_2025[fe.matches_2025['status'] == 'FINISHED'],
    ])
    targets_goals = all_matches['home_score'] + all_matches['away_score']
    
    if len(targets_goals) > 0:
        goals.train(features_df, targets_goals)
    
    print("\n✅ Entrenamiento completo!")
    print(f"  Modelos guardados en: models/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())