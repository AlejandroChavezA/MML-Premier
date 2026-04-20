"""
MML-Premier v2
=============
Sistema de predicción de Premier League reorganizado por capas.

Estructura:
- data/:     Limpieza y gestión de datos crudos
- features/: Ingeniería de features
- models:    Modelos de ML (winner + goals)
- evaluation/: Análisis y métricas
- ui:        Interfaz de usuario
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"