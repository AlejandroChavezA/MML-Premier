#!/usr/bin/env python3
"""Gate shadow: compara accuracy del ensamble nuevo contra el modelo legacy
(cuando existe) en un holdout cronológico, antes de promoverlo a default del
dashboard.

Ver docs/plan_5_ligas_ligamx.md Paso 3 y la decisión acordada con el usuario:
el ensamble NO reemplaza a los modelos sklearn existentes a ciegas -- solo se
vuelve default si iguala o supera su accuracy. Este script hace la comparación
inicial con un holdout cronológico (últimos N partidos, nunca vistos por el
outcome_xgb entrenado acá); la vigilancia en producción sigue después vía
src_v2/evaluation/evaluator.py sobre prediction_history.json a medida que
se acumulan jornadas reales.

Nota: el modelo legacy (src/prediction_models.py) NO se reentrena excluyendo
el holdout -- ya está entrenado sobre todos los datos disponibles (no hace
split cronológico). Esto sesga la comparación A FAVOR del legacy (pudo haber
visto esos partidos en entrenamiento). Si el ensamble iguala/supera al legacy
de todas formas bajo ese sesgo, es evidencia más fuerte a favor de promoverlo.

Ligas sin modelo legacy (p. ej. Liga MX, ver config/leagues/liga_mx.yaml
`has_legacy_baseline: false`): el legacy de Premier (src/menu_interface.PredictionMenu)
está entrenado solo con equipos ingleses, así que compararlo contra otra liga
fallaba en silencio (legacy_n=0, legacy_acc=0.0). En esos casos se compara el
ensamble contra un baseline naive ("siempre gana el local") en vez de forzar
una comparación sin sentido.

Uso:
    python -m core.evaluate_ensemble --data-dir data/cleaned --holdout-frac 0.15
    python -m core.evaluate_ensemble --league liga_mx
    python -m core.evaluate_ensemble --league premier_league --skip-legacy
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src_v2.features.feature_engineer import FeatureEngineer  # noqa: E402

from core.eval_utils import chronological_cutoff  # noqa: E402
from core.models.outcome_xgb import RESULT_MAP, OutcomeXGB  # noqa: E402
from core.train_ensemble import build_training_dataset  # noqa: E402

_CLASSES = [RESULT_MAP[i] for i in range(3)]  # ['VISITANTE', 'EMPATE', 'LOCAL']


def _naive_prior_log_loss(train_targets: pd.Series, actual_labels: list) -> float:
    """Log-loss de predecir, para todo el holdout, la distribución de clases
    observada en train -- piso de referencia probabilístico (nunca "ve" el
    holdout, a diferencia de usar las frecuencias del propio holdout).

    Es el análogo en log-loss del baseline naive "siempre gana el local" que
    ya se usa para accuracy: ese baseline nunca predice EMPATE, así que
    superarlo en accuracy no exige que el modelo aprenda a distinguir
    empates. Log-loss con este prior sí lo exige, porque penaliza fuerte
    (log(p) -> -inf si p->0) cualquier resultado al que el modelo/baseline le
    asigne probabilidad cercana a cero.
    """
    counts = train_targets.value_counts(normalize=True)
    prior = [counts.get(i, 1e-6) for i in range(3)]
    y_proba = [prior] * len(actual_labels)
    return log_loss(actual_labels, y_proba, labels=_CLASSES)


def _naive_home_baseline(holdout_matches: pd.DataFrame) -> float:
    """Accuracy de "siempre gana el local" en el holdout -- piso de referencia
    cuando no hay modelo legacy contra el cual comparar."""
    if len(holdout_matches) == 0:
        return 0.0
    home_wins = (holdout_matches['home_score'] > holdout_matches['away_score']).sum()
    return float(home_wins) / len(holdout_matches)


def evaluate(data_dir: str, holdout_frac: float = 0.15, run_legacy: bool = True,
             outcome_xgb_params: dict = None, extra_features: list = None) -> dict:
    fe = FeatureEngineer(data_dir, extra_features=extra_features)
    if not fe.load_data():
        raise RuntimeError(f"No se pudo cargar data_dir={data_dir}")

    all_matches = pd.concat([df[df['status'] == 'FINISHED'] for df in fe.matches_by_season.values()])
    cutoff_date = chronological_cutoff(all_matches['date'], holdout_frac)
    holdout_matches = all_matches[all_matches['date'] >= cutoff_date]
    print(f"Holdout: {len(holdout_matches)} partidos desde {cutoff_date}")

    # -- Ensamble: entrenar OutcomeXGB solo con partidos ANTERIORES al holdout --
    features_df, outcome_targets, _, dates, _ = build_training_dataset(fe)
    train_mask = (dates < cutoff_date).values

    outcome_model = OutcomeXGB(models_dir="/tmp/ensemble_eval_tmp")
    outcome_model.train(
        features_df[train_mask], outcome_targets[train_mask], dates=dates[train_mask],
        xgb_params=outcome_xgb_params,
    )

    ensemble_correct = 0
    ensemble_total = 0
    actual_labels = []
    predicted_probas = []
    for _, m in holdout_matches.iterrows():
        pred = outcome_model.predict_proba(m['home_team'], m['away_team'], m['date'], fe)
        if pred.get('error'):
            continue
        actual = 'LOCAL' if m['home_score'] > m['away_score'] else (
            'VISITANTE' if m['home_score'] < m['away_score'] else 'EMPATE')
        ensemble_total += 1
        if pred['predicted_result'] == actual:
            ensemble_correct += 1
        actual_labels.append(actual)
        predicted_probas.append([pred['probabilities'][c] for c in _CLASSES])

    ensemble_acc = ensemble_correct / ensemble_total if ensemble_total else 0.0
    ensemble_log_loss = log_loss(actual_labels, predicted_probas, labels=_CLASSES) if ensemble_total else None
    naive_log_loss = _naive_prior_log_loss(outcome_targets[train_mask], actual_labels) if ensemble_total else None

    result = {
        'holdout_matches': len(holdout_matches),
        'ensemble_accuracy': ensemble_acc,
        'ensemble_n': ensemble_total,
        'ensemble_log_loss': ensemble_log_loss,
        'naive_log_loss': naive_log_loss,
    }

    if not run_legacy:
        baseline_acc = _naive_home_baseline(holdout_matches)
        result.update({
            'legacy_accuracy': None,
            'legacy_n': 0,
            'legacy_model': None,
            'baseline_accuracy': baseline_acc,
            'promote_ensemble': ensemble_acc >= baseline_acc,
            'note': ('Sin modelo legacy para esta liga: comparado contra baseline naive '
                     '(siempre gana el local), no contra un legacy real.'),
        })
        return result

    # -- Legacy: modelo ya entrenado (sklearn, src/prediction_models.py) --
    from menu_interface import PredictionMenu  # noqa: E402

    menu = PredictionMenu()
    if not menu.initialize():
        raise RuntimeError("No se pudo inicializar el legacy PredictionMenu")

    legacy_correct = 0
    legacy_total = 0
    for _, m in holdout_matches.iterrows():
        pred = menu.predictor.predict_match(m['home_team'], m['away_team'], m['date'], menu.current_model)
        if pred.get('error'):
            continue
        actual = 'LOCAL' if m['home_score'] > m['away_score'] else (
            'VISITANTE' if m['home_score'] < m['away_score'] else 'EMPATE')
        legacy_total += 1
        if pred['predicted_result'] == actual:
            legacy_correct += 1

    legacy_acc = legacy_correct / legacy_total if legacy_total else 0.0

    result.update({
        'legacy_accuracy': legacy_acc,
        'legacy_n': legacy_total,
        'legacy_model': menu.current_model,
        'promote_ensemble': ensemble_acc >= legacy_acc,
    })
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compara ensamble vs legacy (o baseline naive) en un holdout cronológico")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--league", default=None,
                        help="Slug de config/leagues/{league}.yaml -- define data-dir "
                             "y si corresponde comparar contra legacy (has_legacy_baseline)")
    parser.add_argument("--skip-legacy", action="store_true",
                        help="Forzar comparación contra baseline naive en vez de legacy")
    args = parser.parse_args()

    data_dir = args.data_dir
    run_legacy = not args.skip_legacy
    outcome_xgb_params = None
    extra_features = None

    if args.league:
        from core.league_config import load_league_config
        cfg = load_league_config(args.league)
        if data_dir is None:
            data_dir = str(cfg.data_dir_path)
        if not args.skip_legacy:
            run_legacy = cfg.has_legacy_baseline
        outcome_xgb_params = cfg.outcome_xgb.model_dump()
        extra_features = cfg.extra_features

    if data_dir is None:
        data_dir = "data/cleaned"

    result = evaluate(data_dir, args.holdout_frac, run_legacy=run_legacy, outcome_xgb_params=outcome_xgb_params,
                       extra_features=extra_features)
    print(f"\nEnsamble (OutcomeXGB):  {result['ensemble_accuracy']:.1%} ({result['ensemble_n']} partidos)")
    print(f"Log-loss ensamble: {result['ensemble_log_loss']:.4f}  |  "
          f"log-loss prior naive (frecuencias de train, sin ver empates como imposibles): "
          f"{result['naive_log_loss']:.4f}  "
          f"({'mejor' if result['ensemble_log_loss'] < result['naive_log_loss'] else 'NO mejor'} que el prior)")

    if result.get('legacy_model'):
        print(f"Legacy ({result['legacy_model']}): {result['legacy_accuracy']:.1%} ({result['legacy_n']} partidos)")
    else:
        print(f"Baseline naive (siempre local): {result['baseline_accuracy']:.1%}")
        print(result.get('note', ''))

    if result['promote_ensemble']:
        print("\n=> El ensamble iguala o supera la referencia: apto para promover a default.")
    else:
        print("\n=> El ensamble NO supera la referencia todavía: se mantiene el legacy/baseline.")


if __name__ == "__main__":
    main()
