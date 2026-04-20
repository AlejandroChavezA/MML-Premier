"""
Models Layer
============
Modelos de ML: features → predictions

Clases:
- WinnerPredictor: Predice 1X2 (LOCAL/EMPATE/VISITANTE)
- GoalsPredictor: Predice Over/Under
"""

from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"