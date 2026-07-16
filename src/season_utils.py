"""Utilidades para determinar la temporada activa de la Premier League.

La API de football-data.org identifica la temporada por el año de inicio
(p.ej. la temporada 2025/26 se referencia como "2025"). La Premier League
arranca a fines de julio/agosto, así que el corte de temporada se adelanta a
julio para que la nueva temporada pase a ser "actual" apenas empiean las
pretemporadas:
  - De julio (mes 7) a diciembre  -> temporada = año actual
  - De enero a junio              -> temporada = año actual - 1
"""
from datetime import datetime


def get_current_season(now: datetime = None) -> int:
    """Devuelve el año de inicio de la temporada activa."""
    now = now or datetime.now()
    return now.year if now.month >= 7 else now.year - 1


def get_recent_seasons(n: int = 3, now: datetime = None):
    """Devuelve las últimas `n` temporadas (de la más antigua a la actual)."""
    current = get_current_season(now)
    return [current - i for i in reversed(range(n))]


def get_latest_finished_season(available_seasons: list, now: datetime = None) -> int:
    """Devuelve la mayor temporada cargada que tenga partidos finalizados.

    Hoy (jul 2026) la temporada 2026 esta vacia (0 finalizados) aunque sea la maxima
    cargada; por eso la referencia para predicciones/standings es 2025 (la ultima con
    resultados reales).
    """
    finished = {}
    for y in available_seasons:
        try:
            import pandas as pd
            df = pd.read_csv(f"data/cleaned/matches_{y}_cleaned.csv")
            finished[y] = int((df['status'].astype(str).str.upper() == 'FINISHED').sum())
        except Exception:
            finished[y] = 0
    candidates = [y for y, n in finished.items() if n > 0]
    return max(candidates) if candidates else (max(available_seasons) if available_seasons else get_current_season(now))
