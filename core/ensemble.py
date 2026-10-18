"""Reconciliación del ensamble de 3 modelos en una predicción única.

Ver docs/plan_5_ligas_ligamx.md, Paso 3. Cada modelo es autoritativo para
una parte de la predicción final:

- OutcomeXGB (XGBoost)     -> predicted_result, confidence, probabilities
- DixonColesModel          -> marcador más probable / scorelines candidatos
- GoalsGLM (Poisson)       -> Over/Under, goles esperados

La única regla dura: el marcador elegido nunca puede contradecir el
predicted_result de XGBoost. Si el top-1 de Dixon-Coles contradice al
outcome, se re-rankea la lista de scorelines de DC y se toma el de mayor
probabilidad cuyo resultado implícito sí coincida.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_OU_DISAGREEMENT_THRESHOLD = 1.0  # goles


def _implied_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return 'LOCAL'
    if home_goals < away_goals:
        return 'VISITANTE'
    return 'EMPATE'


def _select_scoreline(outcome: str, dc_result: Dict) -> Optional[Dict]:
    """Marcador de mayor probabilidad de DC cuyo resultado implícito == outcome."""
    for candidate in dc_result.get('topScorelines', []):
        if _implied_outcome(candidate['home_goals'], candidate['away_goals']) == outcome:
            return candidate
    return None


def reconcile(outcome_result: Dict, dc_result: Dict, glm_result: Dict) -> Dict:
    """Combina las 3 salidas en una predicción única y no contradictoria."""
    for name, result in (('outcome', outcome_result), ('dixon_coles', dc_result), ('glm', glm_result)):
        if result.get('error'):
            return {'error': f"{name}: {result['error']}"}

    probabilities = outcome_result['probabilities']
    predicted_result = outcome_result['predicted_result']
    confidence = probabilities[predicted_result]

    scoreline = _select_scoreline(predicted_result, dc_result)
    if scoreline is None:
        # DC no tiene ningún candidato con ese resultado implícito en su top-N;
        # se mantiene el resultado de XGBoost como autoritativo y se cae al
        # marcador más probable "genérico" para ese resultado.
        home_goals, away_goals = 1, 0
        if predicted_result == 'VISITANTE':
            home_goals, away_goals = 0, 1
        elif predicted_result == 'EMPATE':
            home_goals, away_goals = 1, 1
        scoreline = {'score': f'{home_goals}-{away_goals}', 'prob': None,
                     'home_goals': home_goals, 'away_goals': away_goals}
        logger.warning(
            "Sin marcador de Dixon-Coles consistente con %s (%s vs %s); usando fallback %s",
            predicted_result, outcome_result['home_team'], outcome_result['away_team'], scoreline['score'],
        )

    glm_total = glm_result['expected_goals']
    dc_xg = dc_result['expectedGoals']
    dc_total = dc_xg['total']
    if abs(glm_total - dc_total) > _OU_DISAGREEMENT_THRESHOLD:
        logger.warning(
            "GLM y Dixon-Coles difieren en goles esperados por >%.1f (%.2f vs %.2f) en %s vs %s",
            _OU_DISAGREEMENT_THRESHOLD, glm_total, dc_total,
            outcome_result['home_team'], outcome_result['away_team'],
        )

    # O/U y xG salen de Dixon-Coles (ataque/defensa por equipo). El GLM Poisson
    # de totales se colapsa a la media de la liga (std=0 en Liga MX J11: 2.66
    # en los 9 partidos → todo OVER 2.5), así que no se usa para el mercado.
    over_under = {}
    for key, val in (dc_result.get('overUnder') or {}).items():
        over_p = float(val.get('prob', 0.5))
        over_under[key] = {
            'over_prob': over_p,
            'under_prob': 1.0 - over_p,
            'prediction': val.get('prediction') or ('OVER' if over_p >= 0.5 else 'UNDER'),
            'confidence': abs(over_p - 0.5) * 2,
        }

    return {
        'home_team': outcome_result['home_team'],
        'away_team': outcome_result['away_team'],
        'predicted_result': predicted_result,
        'confidence': confidence,
        'probabilities': probabilities,
        'most_likely_scoreline': scoreline['score'],
        'top_scorelines': dc_result.get('topScorelines', []),
        'expected_goals': dc_total,
        'expected_goals_home': dc_xg.get('home'),
        'expected_goals_away': dc_xg.get('away'),
        'over_under': over_under,
        'raw_features': outcome_result.get('raw_features') or {},
    }
