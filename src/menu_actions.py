"""Funciones de negocio "puras" extraídas de PredictionMenu (menu_interface.py).

Reciben los componentes ya inicializados (feature_engineer, predictor, etc.) como
parámetros explícitos y devuelven datos, sin print()/input(). Tanto el menú
interactivo (menu_interface.py) como el CLI (run.py) llaman a estas mismas
funciones para que nunca diverjan en comportamiento.
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def get_jornada_status(project_root: Path, season: int, data_dir: Optional[Path] = None,
                        phase: Optional[str] = None) -> Dict:
    """Jornadas disponibles/completadas/pendientes de una temporada.

    `data_dir` es opcional (default: data/cleaned de Premier, comportamiento
    histórico sin cambios). `phase` es para ligas split (Apertura/Clausura,
    ej. Liga MX) donde el número de jornada se repite por fase dentro del
    mismo año -- sin filtrar por fase, jornada 1 de Apertura y jornada 1 de
    Clausura se mezclarían como si fueran el mismo bloque.
    """
    data_dir = Path(data_dir) if data_dir is not None else Path(project_root) / "data" / "cleaned"
    matches_path = data_dir / f"matches_{season}_cleaned.csv"
    matches = pd.read_csv(matches_path)
    if phase is not None:
        matches = matches[matches['phase'].astype(str).str.lower().str.startswith(phase.lower())]
    available = sorted(matches['matchday'].unique())

    finished = []
    upcoming = []
    for jornada in available:
        jornada_matches = matches[matches['matchday'] == jornada]
        if all(jornada_matches['status'] == 'FINISHED'):
            finished.append(jornada)
        elif all(jornada_matches['status'] == 'TIMED'):
            upcoming.append(jornada)

    return {
        'matches': matches,
        'available_matchdays': available,
        'finished_jornadas': finished,
        'upcoming_jornadas': upcoming,
    }


def resolve_next_matchday(status: Dict) -> Optional[int]:
    """Primera jornada que todavía no está 100% FINISHED."""
    matches = status['matches']
    for jornada in status['available_matchdays']:
        jornada_matches = matches[matches['matchday'] == jornada]
        if not all(jornada_matches['status'] == 'FINISHED'):
            return jornada
    return None


def resolve_last_finished_matchday(status: Dict) -> Optional[int]:
    finished = status['finished_jornadas']
    return max(finished) if finished else None


def predict_jornada(predictor, matchday: int, season: int, model: str) -> List[Dict]:
    """Predicciones de todos los partidos de una jornada."""
    return predictor.predict_week_matches(matchday, season, model)


def summarize_jornada(predictions: List[Dict], model: str) -> Dict:
    """Resumen agregado de una jornada ya predicha: conteos, confianza, competitividad."""
    valid = [p for p in predictions if 'error' not in p]
    if not valid:
        return {}

    return {
        'competitiveness_level': valid[0].get('competitiveness_level', 'N/A'),
        'competitiveness_score': valid[0].get('competitiveness', 0),
        'local_wins': sum(1 for p in valid if p['predicted_result'] == 'LOCAL'),
        'away_wins': sum(1 for p in valid if p['predicted_result'] == 'VISITANTE'),
        'draws': sum(1 for p in valid if p['predicted_result'] == 'EMPATE'),
        'total_matches': len(valid),
        'avg_confidence': sum(p['confidence'] for p in valid) / len(valid),
        'model': model,
    }


def predict_single_match(predictor, home_team: str, away_team: str, match_date, model: str) -> Dict:
    return predictor.predict_match(home_team, away_team, match_date, model)


def get_team_statistics(feature_engineer, team: str, as_of=None) -> Dict:
    """Forma reciente + rendimiento local/visitante + posición en tabla de un equipo."""
    as_of = as_of if as_of is not None else pd.Timestamp.now()
    return {
        'team': team,
        'form': feature_engineer.calculate_team_form(team, as_of),
        'home_performance': feature_engineer.get_home_away_performance(team, 'home', as_of),
        'away_performance': feature_engineer.get_home_away_performance(team, 'away', as_of),
        'standings': feature_engineer.get_current_standings_position(team),
    }


def load_standings(project_root: Path, season: int) -> pd.DataFrame:
    standings_path = Path(project_root) / "data" / "cleaned" / f"standings_{season}_cleaned.csv"
    return pd.read_csv(standings_path)


def list_models_with_accuracy(predictor, current_model: str) -> List[Dict]:
    performance = predictor.get_model_performance()
    return [
        {
            'name': model,
            'is_current': model == current_model,
            'test_accuracy': performance.get(model, {}).get('test_accuracy', 0),
        }
        for model in predictor.models.keys()
    ]


def get_model_performance_report(predictor, historical_accuracy: Optional[Dict] = None) -> Dict:
    performance = predictor.get_model_performance()
    if not performance:
        return {}
    best_model = predictor.get_best_model()
    return {
        'performance': performance,
        'historical_accuracy': historical_accuracy or {},
        'best_model': best_model,
        'best_metrics': performance.get(best_model, {}),
    }
