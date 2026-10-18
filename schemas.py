"""Contrato de exportación de predicciones hacia safesports-panel.

Ver .claude/CLAUDE.md para el formato de payload documentado y
docs/plan_5_ligas_ligamx.md (Paso 1) para el motivo de este módulo:
toda predicción exportada debe pasar por MatchPrediction antes de
escribirse o postearse, para fallar ruidosamente si falta un campo
en vez de mandar un payload roto en silencio.
"""
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class PredictionSummary(BaseModel):
    winnerFactors: int
    loserFactors: int
    matchupType: str
    betRecommendation: str


class PredictionArguments(BaseModel):
    forWinner: List[str]
    forLoser: List[str]
    summary: PredictionSummary


class PredictedScore(BaseModel):
    home: int = Field(ge=0)
    away: int = Field(ge=0)


class SoccerData(BaseModel):
    predictedScore: PredictedScore
    awayWinPct: Optional[int] = Field(default=None, ge=0, le=100)
    homeWinPct: Optional[int] = Field(default=None, ge=0, le=100)
    drawPct: Optional[int] = Field(default=None, ge=0, le=100)
    awayScore: Optional[float] = None  # xG visitante
    homeScore: Optional[float] = None  # xG local
    totalScore: Optional[float] = None
    betLine: Optional[float] = None
    betSide: Optional[Literal["OVER", "UNDER"]] = None
    ouProb: Optional[int] = Field(default=None, ge=0, le=100)


class MatchPrediction(BaseModel):
    sport: str
    homeTeam: str
    homeTeamFullName: str
    homeTeamLogo: str
    awayTeam: str
    awayTeamFullName: str
    awayTeamLogo: str
    predictedWinner: str
    confidence: int = Field(ge=0, le=100)
    riskLevel: Literal["low", "medium", "high"]
    gameDate: str
    status: str
    notes: str
    arguments: PredictionArguments
    soccerLeague: Optional[str] = None  # slug del panel (ej. "premier", "ligamx") -- ver safesports-panel/types/prediction.ts
    soccerData: Optional[SoccerData] = None  # contrato soccer-v1: marcador {home, away}

    # Solo presentes al exportar historial ya jugado (export_history_to_panel_format).
    actualWinner: Optional[str] = None
    isCorrect: Optional[bool] = None


def soccer_predicted_score(scoreline: Union[str, dict, None] = None,
                           predicted_result: str = "EMPATE") -> dict:
    """Enteros home/away para soccerData.predictedScore (contrato soccer-v1)."""
    parsed = _parse_scoreline(scoreline)
    if parsed is not None:
        return parsed
    if predicted_result == "LOCAL":
        return {"home": 2, "away": 1}
    if predicted_result == "VISITANTE":
        return {"home": 1, "away": 2}
    return {"home": 1, "away": 1}


def _parse_scoreline(scoreline: Union[str, dict, None]) -> Optional[dict]:
    if scoreline is None:
        return None
    if isinstance(scoreline, dict):
        if "home" in scoreline and "away" in scoreline:
            return {"home": int(scoreline["home"]), "away": int(scoreline["away"])}
        return None
    text = str(scoreline).strip().replace("–", "-").replace(":", "-")
    parts = text.split("-")
    if len(parts) != 2:
        return None
    try:
        return {"home": int(parts[0].strip()), "away": int(parts[1].strip())}
    except ValueError:
        return None
