#!/usr/bin/env python3
"""Entrena los 3 modelos del ensamble (OutcomeXGB, GoalsGLM, DixonColesModel)
para una liga (ver docs/plan_5_ligas_ligamx.md, Paso 3).

Uso:
    python -m core.train_ensemble --data-dir data/cleaned --models-dir models/premier_league
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src_v2.features.feature_engineer import FeatureEngineer  # noqa: E402

from core.models.dixon_coles import DixonColesModel  # noqa: E402
from core.models.goals_glm import GoalsGLM  # noqa: E402
from core.models.outcome_xgb import OutcomeXGB  # noqa: E402


def build_training_dataset(fe: FeatureEngineer):
    """Features + targets (outcome y goles) + fecha, alineados fila-a-fila.

    A diferencia de src_v2/train.py (que arma el target de goles con un
    pd.concat aparte de create_training_dataset(), sin garantía de que las
    filas queden alineadas si algún partido falla en create_match_features),
    acá se arma todo en un único loop para que features_df, outcome_targets,
    goals_targets y dates queden siempre alineados por construcción -- incluyendo
    la fecha permite hacer split cronológico exacto (core.eval_utils) en vez de
    aproximar por posición contra `all_matches` (que puede tener filas de más
    por partidos que create_match_features descarta).
    """
    all_matches = pd.concat([
        df[df['status'] == 'FINISHED'].assign(_season_year=year)
        for year, df in fe.matches_by_season.items()
    ])

    features_list, outcome_targets, goals_targets, dates = [], [], [], []
    for _, m in all_matches.iterrows():
        try:
            phase = m['phase'] if 'phase' in m.index and pd.notna(m['phase']) else None
            feats = fe.create_match_features(m['home_team'], m['away_team'], m['date'],
                                              season_year=int(m['_season_year']), phase=phase)
            numeric_feats = {k: v for k, v in feats.items() if k not in ('home_team', 'away_team', 'date')}
        except Exception:
            continue
        features_list.append(numeric_feats)
        if m['home_score'] > m['away_score']:
            outcome_targets.append(2)
        elif m['home_score'] < m['away_score']:
            outcome_targets.append(0)
        else:
            outcome_targets.append(1)
        goals_targets.append(m['home_score'] + m['away_score'])
        dates.append(m['date'])

    features_df = pd.DataFrame(features_list)
    return (features_df, pd.Series(outcome_targets, name='result'),
            pd.Series(goals_targets, name='total_goals'),
            pd.Series(dates, name='date'), all_matches)


def train_ensemble(data_dir: str, models_dir: str) -> dict:
    fe = FeatureEngineer(data_dir)
    if not fe.load_data():
        raise RuntimeError(f"No se pudo cargar data_dir={data_dir}")

    features_df, outcome_targets, goals_targets, dates, all_matches = build_training_dataset(fe)
    if features_df.empty:
        raise RuntimeError("Dataset de entrenamiento vacío")

    outcome_model = OutcomeXGB(models_dir=models_dir)
    outcome_model.train(features_df, outcome_targets, dates=dates)

    goals_model = GoalsGLM(models_dir=models_dir)
    goals_model.train(features_df, goals_targets, dates=dates)

    # Dixon-Coles se entrena directo sobre los partidos crudos (no usa
    # FeatureEngineer): usa todos los FINISHED disponibles, no solo los que
    # sobrevivieron a create_match_features.
    dc_model = DixonColesModel()
    dc_fit_result = dc_model.fit(all_matches)
    dc_path = Path(models_dir)
    dc_path.mkdir(parents=True, exist_ok=True)
    dc_model.save(dc_path / "dixon_coles.json")

    return {
        'n_samples': len(features_df),
        'outcome_performance': outcome_model.get_performance(),
        'goals_performance': goals_model.get_performance(),
        'dixon_coles_fit': dc_fit_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Entrena el ensamble de 3 modelos para una liga")
    parser.add_argument("--data-dir", default="data/cleaned")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    result = train_ensemble(args.data_dir, args.models_dir)
    print(f"Samples: {result['n_samples']}")
    print(f"Outcome (XGBoost): {result['outcome_performance']}")
    print(f"Goals (GLM Poisson): {result['goals_performance']}")
    print(f"Dixon-Coles: {result['dixon_coles_fit']}")


if __name__ == "__main__":
    main()
