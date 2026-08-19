"""Utilidades para determinar la temporada activa de una liga.

Por defecto (sin `league_cfg`, o con una liga `season_type="single"` como
Premier) se usa el corte de temporada clásico europeo: la API de
football-data.org identifica la temporada por el año de inicio (p.ej. la
temporada 2025/26 se referencia como "2025"), y como la Premier League
arranca a fines de julio/agosto, el corte se adelanta a julio:
  - De julio (mes 7) a diciembre  -> temporada = año actual
  - De enero a junio              -> temporada = año actual - 1

Ligas `season_type="split"` (Apertura/Clausura, ej. Liga MX) no tienen "año
de inicio de temporada" -- tienen dos medias temporadas por año calendario.
Para esas, `get_current_season` devuelve el año calendario del archivo
matches_{year}_cleaned.csv (que contiene ambas fases) y `get_current_phase`
devuelve cuál de las dos ("AP"/"CL") está activa. Ver
docs/plan_5_ligas_ligamx.md Paso 5 y core/league_config.py.
"""
from datetime import datetime
from typing import Optional


def get_current_season(now: datetime = None, league_cfg=None) -> int:
    """Devuelve el año de la temporada activa (o del archivo activo, en ligas split)."""
    now = now or datetime.now()
    if league_cfg is not None and getattr(league_cfg, "season_type", "single") == "split":
        return now.year
    cutover_month = getattr(league_cfg, "season_cutover_month", 7) if league_cfg is not None else 7
    return now.year if now.month >= cutover_month else now.year - 1


def get_current_phase(league_cfg, now: datetime = None) -> Optional[str]:
    """Para ligas split (Apertura/Clausura): devuelve "AP" o "CL" según el mes actual.

    None para ligas de temporada única (no aplica).
    """
    if league_cfg is None or getattr(league_cfg, "season_type", "single") != "split":
        return None
    now = now or datetime.now()
    ac = getattr(league_cfg, "apertura_clausura", None)
    cutover_month = ac.cutover_month if ac is not None else 6
    return "AP" if now.month >= cutover_month else "CL"


def get_recent_seasons(n: int = 3, now: datetime = None):
    """Devuelve las últimas `n` temporadas (de la más antigua a la actual)."""
    current = get_current_season(now)
    return [current - i for i in reversed(range(n))]


def get_latest_finished_season(available_seasons: list, now: datetime = None,
                                data_dir: str = "data/cleaned") -> int:
    """Devuelve la mayor temporada cargada que tenga partidos finalizados.

    Hoy (jul 2026) la temporada 2026 esta vacia (0 finalizados) aunque sea la maxima
    cargada; por eso la referencia para predicciones/standings es 2025 (la ultima con
    resultados reales).
    """
    finished = {}
    for y in available_seasons:
        try:
            import pandas as pd
            df = pd.read_csv(f"{data_dir}/matches_{y}_cleaned.csv")
            finished[y] = int((df['status'].astype(str).str.upper() == 'FINISHED').sum())
        except Exception:
            finished[y] = 0
    candidates = [y for y, n in finished.items() if n > 0]
    return max(candidates) if candidates else (max(available_seasons) if available_seasons else get_current_season(now))
