"""Contrato de exportación de predicciones hacia safesports-panel.

Ver .claude/CLAUDE.md para el formato de payload documentado y
docs/plan_5_ligas_ligamx.md (Paso 1) para el motivo de este módulo:
toda predicción exportada debe pasar por MatchPrediction antes de
escribirse o postearse, para fallar ruidosamente si falta un campo
en vez de mandar un payload roto en silencio.
"""
from typing import List, Literal, Optional

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

    # Solo presentes al exportar historial ya jugado (export_history_to_panel_format).
    actualWinner: Optional[str] = None
    isCorrect: Optional[bool] = None
