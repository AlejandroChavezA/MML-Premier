"""Config por liga (Paso 4). Ver docs/plan_5_ligas_ligamx.md."""
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent.parent
LEAGUES_DIR = PROJECT_ROOT / "config" / "leagues"


class WinterBreak(BaseModel):
    active: bool = False
    start: Optional[str] = None  # "MM-DD"
    end: Optional[str] = None    # "MM-DD"


class LowerLeague(BaseModel):
    enabled: bool = False
    data_dir: Optional[str] = None


class PromotedTeamFallback(BaseModel):
    strategy: Literal["lower_league_stats", "league_median", "fixed_prior"] = "league_median"


class EloConfig(BaseModel):
    k_factor: float = 20.0
    home_advantage: float = 60.0


class AperturaClausura(BaseModel):
    cutover_month: int = 6


class LeagueConfig(BaseModel):
    # str, no int: cada liga puede venir de un proveedor distinto con su propio
    # esquema de IDs (football-data.org usa códigos alfa como "PL"; API-Football
    # usa numéricos como "262" para Liga MX).
    league_id: str
    name: str
    slug: str
    # Slug que espera safesports-panel en el campo soccerLeague (distinto del
    # `slug` interno -- el panel usa "premier"/"ligamx", no "premier_league"/"liga_mx").
    # Ver safesports-panel/components/admin/PredictionForm.tsx:SOCCER_LEAGUE_SLUG.
    panel_slug: str
    country: str
    data_dir: str
    models_dir: str
    teams_expected: int = 20
    jornadas: int = 38
    season_type: Literal["single", "split"] = "single"
    season_cutover_month: int = 7  # solo relevante si season_type == "single"
    apertura_clausura: Optional[AperturaClausura] = None  # solo si season_type == "split"
    winter_break: WinterBreak = Field(default_factory=WinterBreak)
    lower_league: LowerLeague = Field(default_factory=LowerLeague)
    # Solo Premier tiene un modelo legacy sklearn (src/prediction_models.py) entrenado
    # sobre sus datos -- para ligas sin ese baseline, core.evaluate_ensemble.py no debe
    # intentar cargar PredictionMenu (comparaba contra un modelo entrenado con equipos
    # de otra liga y fallaba en silencio). Ver docs/plan_5_ligas_ligamx.md.
    has_legacy_baseline: bool = False
    promoted_team_fallback: PromotedTeamFallback = Field(default_factory=PromotedTeamFallback)
    elo: EloConfig = Field(default_factory=EloConfig)
    extra_features: List[str] = Field(default_factory=list)

    @property
    def data_dir_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def models_dir_path(self) -> Path:
        return PROJECT_ROOT / self.models_dir


def load_league_config(league: str) -> LeagueConfig:
    """Lee config/leagues/{league}.yaml. `league` es el slug (ej. 'premier_league', 'liga_mx')."""
    path = LEAGUES_DIR / f"{league}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No existe config de liga: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return LeagueConfig(**data)


def list_available_leagues() -> List[str]:
    if not LEAGUES_DIR.exists():
        return []
    return sorted(p.stem for p in LEAGUES_DIR.glob("*.yaml"))
