"""Corre los 3 modelos del ensamble para un partido y los reconcilia.

Función compartida entre la verificación del Paso 3 y core/predict_week.py
(Paso 4) -- un solo lugar que sabe cómo invocar OutcomeXGB + DixonColesModel +
GoalsGLM y pasarlos por core.ensemble.reconcile().
"""
from typing import Dict

from core.ensemble import reconcile
from core.models.dixon_coles import DixonColesModel
from core.models.goals_glm import GoalsGLM
from core.models.outcome_xgb import OutcomeXGB


def predict_match_ensemble(outcome_model: OutcomeXGB, dc_model: DixonColesModel, glm_model: GoalsGLM,
                            feature_engineer, home_team: str, away_team: str, match_date) -> Dict:
    outcome_result = outcome_model.predict_proba(home_team, away_team, match_date, feature_engineer)
    if outcome_result.get('error'):
        return {'error': f"outcome: {outcome_result['error']}"}

    dc_result = dc_model.full_predict(home_team, away_team)

    glm_result = glm_model.predict(home_team, away_team, match_date, feature_engineer)
    if glm_result.get('error'):
        return {'error': f"glm: {glm_result['error']}"}

    return reconcile(outcome_result, dc_result, glm_result)
