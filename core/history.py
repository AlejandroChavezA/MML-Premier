"""Historial de predicciones de producción, multi-liga.

Comparte data/prediction_history.json con Premier
(src/menu_interface.py:_save_prediction_to_history), agregando el campo
`league` para poder filtrar por liga más adelante (ver
docs/plan_5_ligas_ligamx.md, Paso 5). Las entradas viejas sin el campo son de
Premier -- era la única liga con historial hasta que Liga MX se expuso vía
`run.py jornada --league liga_mx`.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_HISTORY_PATH = Path(__file__).parent.parent / "data" / "prediction_history.json"


def _load(history_path: Path) -> List[Dict]:
    if not history_path.exists():
        return []
    try:
        with open(history_path) as f:
            return json.load(f)
    except Exception:
        return []


def append_predictions(league_slug: str, matchday: int, season: int, predictions: List[Dict],
                        history_path: Optional[Path] = None) -> None:
    """Agrega las predicciones de una jornada al historial compartido, tageadas por liga.

    Idempotente por (league, home_team, away_team, match_date, matchday): correr
    la misma jornada dos veces actualiza la entrada existente en vez de duplicarla.
    """
    history_path = history_path or DEFAULT_HISTORY_PATH
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = _load(history_path)

    for pred in predictions:
        if pred.get('error'):
            continue
        # No loguear backfill retroactivo: si el partido ya se jugó, esto ya no es
        # una predicción, es el modelo reconociendo un resultado que pudo haber visto
        # en su propio entrenamiento. Encontrado en Premier: el 100% de las 416
        # entradas viejas de data/prediction_history.json eran así (guardadas en
        # promedio 119+ días después del partido) e inflaban el accuracy "real" de
        # gradient_boosting_v2. Ver CLAUDE.md.
        if pred.get('status') == 'FINISHED':
            continue
        match_date = str(pred.get('match_date', ''))[:10] if pred.get('match_date') else None
        dedupe_key = (league_slug, pred.get('home_team'), pred.get('away_team'), match_date, matchday)
        history = [
            e for e in history
            if (e.get('league', 'premier_league'), e.get('home_team'), e.get('away_team'),
                e.get('match_date'), e.get('matchday')) != dedupe_key
        ]
        history.append({
            'timestamp': datetime.now().isoformat(),
            'league': league_slug,
            'home_team': pred.get('home_team'),
            'away_team': pred.get('away_team'),
            'match_date': match_date,
            'matchday': matchday,
            'season': season,
            'model': 'ensemble',
            'predicted_1x2': pred.get('predicted_result'),
            'confidence': pred.get('confidence'),
            'predicted_scoreline': pred.get('most_likely_scoreline'),
        })

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
