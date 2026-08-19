"""Orquestación multi-liga del ensamble (Paso 4).

predict_week(league, week, season, phase=None) es lo que tanto el menú
refactorizado (Paso 2) como cualquier CLI futuro llaman para obtener
predicciones ya reconciliadas y validadas contra schemas.MatchPrediction.

`core/` (no src_v2/) es dueño de esta orquestación porque es la capa que le
da noción de "liga" a un stack (FeatureEngineer + 3 modelos) que de por sí es
mono-liga -- ver docs/plan_5_ligas_ligamx.md Paso 4.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src_v2.features.feature_engineer import FeatureEngineer  # noqa: E402

from core.league_config import LeagueConfig, load_league_config  # noqa: E402
from core.models.dixon_coles import DixonColesModel  # noqa: E402
from core.models.goals_glm import GoalsGLM  # noqa: E402
from core.models.outcome_xgb import OutcomeXGB  # noqa: E402
from core.predict_match import predict_match_ensemble  # noqa: E402


def _filter_matchday(matches, cfg: LeagueConfig, week: int, phase: Optional[str]):
    week_matches = matches[matches['matchday'] == week]
    if cfg.season_type == "split":
        if phase is None:
            raise ValueError(f"'{cfg.slug}' es una liga split (Apertura/Clausura); falta el parámetro phase")
        # Los datos de Liga MX usan 'apertura'/'clausura' completos, no las
        # siglas AP/CL -- se compara por prefijo case-insensitive.
        week_matches = week_matches[
            week_matches['phase'].astype(str).str.lower().str.startswith(phase.lower())
        ]
    return week_matches


def predict_week(league: str, week: int, season: int, phase: Optional[str] = None) -> List[Dict]:
    """Predicciones reconciliadas para una jornada de una liga, ya validadas
    contra schemas.MatchPrediction en el punto de export (menu_interface.py)."""
    cfg = load_league_config(league)

    fe = FeatureEngineer(str(cfg.data_dir_path))
    if not fe.load_data():
        raise RuntimeError(f"No se pudo cargar data para '{league}' desde {cfg.data_dir_path}")

    matches = fe.matches_by_season.get(season)
    if matches is None:
        raise ValueError(f"Temporada {season} no disponible para '{league}'. "
                          f"Disponibles: {sorted(fe.matches_by_season.keys())}")

    week_matches = _filter_matchday(matches, cfg, week, phase)
    if week_matches.empty:
        raise ValueError(f"No hay partidos para {league} jornada {week} temporada {season}"
                          + (f" fase {phase}" if phase else ""))

    outcome_model = OutcomeXGB(models_dir=str(cfg.models_dir_path))
    glm_model = GoalsGLM(models_dir=str(cfg.models_dir_path))
    dc_path = cfg.models_dir_path / "dixon_coles.json"

    if not outcome_model.load():
        raise RuntimeError(f"No hay OutcomeXGB entrenado en {cfg.models_dir_path}. "
                            f"Correr: python -m core.train_ensemble --data-dir {cfg.data_dir} "
                            f"--models-dir {cfg.models_dir}")
    if not glm_model.load():
        raise RuntimeError(f"No hay GoalsGLM entrenado en {cfg.models_dir_path}")
    if not dc_path.exists():
        raise RuntimeError(f"No hay DixonColesModel entrenado en {dc_path}")
    dc_model = DixonColesModel.load(dc_path)

    predictions = []
    for _, m in week_matches.iterrows():
        result = predict_match_ensemble(outcome_model, dc_model, glm_model, fe,
                                         m['home_team'], m['away_team'], m['date'])
        result['match_date'] = m['date']
        result['status'] = m['status']  # usado por core.history para no loguear backfill retroactivo
        predictions.append(result)

    return predictions
