"""Split cronológico compartido para entrenamiento y evaluación del ensamble.

Un train_test_split al azar sobre partidos de fútbol deja que un partido
"futuro" caiga en train mientras uno "pasado" cae en test -- filtra
información que el modelo no tendría en producción (ver
docs/plan_5_ligas_ligamx.md y el análisis de leakage en src/prediction_models.py).
Acá se corta por fecha: todo lo anterior al corte es train, el resto es test.
"""
from typing import Tuple

import pandas as pd


def chronological_cutoff(dates: pd.Series, holdout_frac: float = 0.2) -> pd.Timestamp:
    """Fecha de corte tal que aprox. `holdout_frac` de las filas quedan después."""
    dates = pd.Series(dates).reset_index(drop=True)
    order = dates.sort_values().index
    cutoff_idx = int(len(order) * (1 - holdout_frac))
    return dates.loc[order[cutoff_idx]]


def chronological_split(
    features_df: pd.DataFrame, targets: pd.Series, dates: pd.Series, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Timestamp]:
    """Divide (features, targets) en train/test cortando por `dates`, no al azar.

    `features_df`, `targets` y `dates` deben venir alineados por posición (mismo
    orden de filas) -- así es como los arma core.train_ensemble.build_training_dataset.
    """
    features_df = features_df.reset_index(drop=True)
    targets = pd.Series(targets).reset_index(drop=True)
    dates = pd.Series(dates).reset_index(drop=True)

    cutoff_date = chronological_cutoff(dates, test_size)
    train_mask = (dates < cutoff_date).values

    return (
        features_df[train_mask], features_df[~train_mask],
        targets[train_mask], targets[~train_mask],
        cutoff_date,
    )
