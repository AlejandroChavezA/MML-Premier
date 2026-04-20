"""
Data Layer
=========
Manejo de datos: raw → cleaned

Clases:
- DataCleaner: Limpieza de datos
- DataCollector: Recolección de datos (API/scraping)
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CLEANED_DIR = DATA_DIR / "cleaned"