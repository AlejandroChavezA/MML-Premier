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

from schemas import MatchPrediction, soccer_predicted_score  # noqa: E402

from core.league_config import LeagueConfig  # noqa: E402

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
import menu_actions  # noqa: E402


def _risk_level(confidence_pct: int) -> str:
    if confidence_pct >= 75:
        return 'low'
    if confidence_pct >= 55:
        return 'medium'
    return 'high'


def _pct(prob: float) -> int:
    return int(round(max(0.0, min(1.0, float(prob or 0))) * 100))


def soccer_data_from_pred(pred: Dict, predicted_winner: str = '', confidence_pct: int = 0) -> Dict:
    """soccer-v1: probs 0-100, xG, marcador entero, Over/Under."""
    probs = pred.get('probabilities') or {}
    score = soccer_predicted_score(
        pred.get('most_likely_scoreline'), pred.get('predicted_result', 'EMPATE'))
    home_xg = pred.get('expected_goals_home')
    away_xg = pred.get('expected_goals_away')
    total_xg = pred.get('expected_goals')
    if home_xg is None or away_xg is None:
        home_xg = float(score['home'])
        away_xg = float(score['away'])
        if total_xg is not None and (home_xg + away_xg) > 0:
            share_home = home_xg / (home_xg + away_xg)
            home_xg = float(total_xg) * share_home
            away_xg = float(total_xg) * (1 - share_home)
    home_xg = round(float(home_xg), 2)
    away_xg = round(float(away_xg), 2)
    total_xg = round(float(total_xg if total_xg is not None else home_xg + away_xg), 2)

    ou = (pred.get('over_under') or {}).get('over_2.5') or {}
    over_prob = ou.get('over_prob', ou.get('prob'))
    side = ou.get('prediction')
    if side not in ('OVER', 'UNDER') and over_prob is not None:
        side = 'OVER' if float(over_prob) >= 0.5 else 'UNDER'
    ou_prob = None
    if side in ('OVER', 'UNDER') and over_prob is not None:
        ou_prob = _pct(over_prob if side == 'OVER' else (1 - float(over_prob)))
    elif total_xg is not None:
        # Premier: Poisson sobre xG, no sobre el marcador entero (2-1 siempre era OVER).
        from core.models.goals_glm import _poisson_over
        over_p = _poisson_over(float(total_xg), 2.5)
        side = 'OVER' if over_p >= 0.5 else 'UNDER'
        ou_prob = _pct(over_p if side == 'OVER' else (1 - over_p))

    data = {
        'awayWinPct': _pct(probs.get('VISITANTE', 0)),
        'homeWinPct': _pct(probs.get('LOCAL', 0)),
        'drawPct': _pct(probs.get('EMPATE', 0)),
        'awayScore': away_xg,
        'homeScore': home_xg,
        'totalScore': total_xg,
        'predictedScore': score,
        'betLine': 2.5,
        'betSide': side if side in ('OVER', 'UNDER') else None,
        'ouProb': ou_prob,
    }
    return {k: v for k, v in data.items() if v is not None}


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
        arguments = menu_actions.build_match_arguments(
            pred.get('predicted_result', 'EMPATE'), confidence_pct, pred.get('raw_features') or {})
        soccer_data = soccer_data_from_pred(pred, predicted_winner, confidence_pct)
        bet_rec = f"{soccer_data['betSide']} {soccer_data['betLine']}" if soccer_data.get('betSide') else (
            f"{predicted_winner} {confidence_pct}%")

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
                'forWinner': arguments['forWinner'],
                'forLoser': arguments['forLoser'],
                'summary': {
                    'winnerFactors': len(arguments['forWinner']),
                    'loserFactors': len(arguments['forLoser']),
                    'matchupType': cfg.slug,
                    'betRecommendation': bet_rec,
                },
            },
            'soccerData': soccer_data,
        }
        MatchPrediction(**panel_pred)  # falla ruidosa si el payload no cumple el contrato
        panel_predictions.append(panel_pred)

    return panel_predictions
