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

_TIE_EPSILON = 0.03  # 3pp
_OU_DISAGREEMENT_THRESHOLD = 1.0  # goles


def _implied_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return 'LOCAL'
    if home_goals < away_goals:
        return 'VISITANTE'
    return 'EMPATE'


def _break_outcome_tie(probabilities: Dict[str, float]) -> str:
    """Si los dos resultados con mayor probabilidad están a <3pp, preferir el no-empate.

    Los clasificadores tienden a subestimar la confianza en empates -- ver
    docs/plan_5_ligas_ligamx.md Paso 3.
    """
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top, second = ranked[0], ranked[1]
    if (top[1] - second[1]) >= _TIE_EPSILON:
        return top[0]
    non_draw_candidates = [kv for kv in (top, second) if kv[0] != 'EMPATE']
    if non_draw_candidates:
        return max(non_draw_candidates, key=lambda kv: kv[1])[0]
    return top[0]


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
    predicted_result = _break_outcome_tie(probabilities)
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
    dc_total = dc_result['expectedGoals']['total']
    if abs(glm_total - dc_total) > _OU_DISAGREEMENT_THRESHOLD:
        logger.warning(
            "GLM y Dixon-Coles difieren en goles esperados por >%.1f (%.2f vs %.2f) en %s vs %s",
            _OU_DISAGREEMENT_THRESHOLD, glm_total, dc_total,
            outcome_result['home_team'], outcome_result['away_team'],
        )

    return {
        'home_team': outcome_result['home_team'],
        'away_team': outcome_result['away_team'],
        'predicted_result': predicted_result,
        'confidence': confidence,
        'probabilities': probabilities,
        'most_likely_scoreline': scoreline['score'],
        'top_scorelines': dc_result.get('topScorelines', []),
        'expected_goals': glm_result['expected_goals'],
        'over_under': glm_result['markets'],
    }
