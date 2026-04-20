"""
Features Layer
============
Ingeniería de features: cleaned data → features

Clases:
- FeatureEngineer: Crea features para modelos
- LeagueCompetitiveness: Métricas de competitividad de liga
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CLEANED_DIR = DATA_DIR / "cleaned"