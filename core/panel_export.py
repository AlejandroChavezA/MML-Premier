"""Payload de dashboard para predicciones de core.predict_week() (Paso 5).

Contraparte de PredictionMenu.transform_to_panel_format() (legacy, atado a la
forma del dict que devuelve src/prediction_models.py) para las predicciones
que salen del ensamble multi-liga -- forma de dict distinta
(most_likely_scoreline, expected_goals, over_under en vez de model_used).
Reusa el mismo schemas.MatchPrediction (Paso 1) y el mismo TEAM_METADATA por
liga que ya usa menu_interface.py (Paso 5).
"""
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas import MatchPrediction  # noqa: E402

from core.league_config import LeagueConfig  # noqa: E402


def _risk_level(confidence_pct: int) -> str:
    if confidence_pct >= 75:
        return 'low'
    if confidence_pct >= 55:
        return 'medium'
    return 'high'


def _team_code(team_name: str, team_codes: Dict[str, str]) -> str:
    if team_name in team_codes:
        return team_codes[team_name]
    return team_name[:3].upper() if len(team_name) >= 3 else team_name.upper()


def ensemble_predictions_to_panel_format(predictions: List[Dict], cfg: LeagueConfig,
                                          team_codes: Dict[str, str],
                                          team_logos: Dict[str, str]) -> List[Dict]:
    """Transforma la salida de core.predict_week() al formato de safesports-panel,
    validando cada predicción contra schemas.MatchPrediction (falla ruidosa)."""
    panel_predictions = []

    for pred in predictions:
        home_code = _team_code(pred['home_team'], team_codes)
        away_code = _team_code(pred['away_team'], team_codes)

        result_map = {'LOCAL': home_code, 'VISITANTE': away_code, 'EMPATE': 'DRAW'}
        predicted_winner = result_map.get(pred['predicted_result'], 'DRAW')

        confidence_pct = int(pred['confidence'] * 100)
        match_date = pred.get('match_date')
        game_date = match_date.isoformat() if hasattr(match_date, 'isoformat') else str(match_date)

        probs = pred.get('probabilities', {})

        panel_pred = {
            'sport': 'soccer',
            'homeTeam': home_code,
            'homeTeamFullName': pred['home_team'],
            'homeTeamLogo': team_logos.get(home_code, ''),
            'awayTeam': away_code,
            'awayTeamFullName': pred['away_team'],
            'awayTeamLogo': team_logos.get(away_code, ''),
            'predictedWinner': predicted_winner,
            'confidence': confidence_pct,
            'riskLevel': _risk_level(confidence_pct),
            'gameDate': game_date,
            'status': 'active',
            'soccerLeague': cfg.panel_slug,
            'notes': f"{cfg.name}\n"
                     f"Marcador probable: {pred.get('most_likely_scoreline', 'N/A')}\n"
                     f"Probabilidades: Local {probs.get('LOCAL', 0):.1%}, "
                     f"Empate {probs.get('EMPATE', 0):.1%}, "
                     f"Visitante {probs.get('VISITANTE', 0):.1%}",
            'arguments': {
                'forWinner': [f"Confianza del ensamble: {confidence_pct}%"],
                'forLoser': [f"Factor de riesgo: {100 - confidence_pct}%"],
                'summary': {
                    'winnerFactors': int(pred['confidence'] * 10),
                    'loserFactors': int((1 - pred['confidence']) * 10),
                    'matchupType': cfg.slug,
                    'betRecommendation': f"{predicted_winner} with {confidence_pct}% confidence",
                },
            },
        }
        MatchPrediction(**panel_pred)  # falla ruidosa si el payload no cumple el contrato
        panel_predictions.append(panel_pred)

    return panel_predictions
