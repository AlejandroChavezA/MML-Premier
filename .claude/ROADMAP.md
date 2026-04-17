# Roadmap MML-Premier - Plan de Desarrollo

## Objetivo
Sistema de predicción de partidos de fútbol con múltiples mercados (ganador, goles, corners, etc.) para Premier League inicialmente, escalando después a otras ligas.

---

## Estado Actual

### ✅ Completado
- Predicción de ganador (1X2)
- Integración con safesports-panel
- Menú interactivo
- Análisis de competitividad de liga

### 🔄 En Progreso
- Nada actualmente

### 📋 Pendiente
- Features adicionales (goles, corners, etc.)
- Escalamiento a otras ligas

---

## Fase 1: Perfilar Premier League

### 1.1 Predicción Ganador (1X2) ✅
- [x] Modelo Random Forest / Logistic Regression
- [x] Probabilidades para Local/Draw/Visitante
- [x] Confianza del modelo
- [x] Análisis de competitividad de liga

### 1.2 Sobre Competitividad

**¿Mejora el accuracy?**

La competitividad **NO mejora directamente el accuracy del modelo base**. Lo que hace es:

1. **Mejor calibración de probabilidades** - Las probabilidades ahora reflejan mejor la realidad de una liga competitiva
2. **Advertencias al usuario** - El sistema avisa cuando hay más riesgo de upsets
3. **Toma de decisiones** - El usuario puede ajustar sus apuestas según el contexto

**Ejemplo práctico:**
- Sin competitividad: "Arsenal 75% de ganar"
- Con competitividad (MODERADAMENTE COMPETITIVA): "Arsenal 75% de ganar - ⚠️ Esperar más empates y posibles upsets"

### 1.3 Over/Under Goles ✅
- [x] Crear modelo para over/under 2.5
- [x] Integrar en menú (opción 4)
- [ ] Crear modelo para over/under 1.5
- [ ] Crear modelo para over/under 3.5

**Métricas del modelo:**
- Test Accuracy: 58.2%
- CV: 56.1% ± 2.8%
- Distribución: 58.8% Over, 41.2% Under

### 1.4 Corners
- [ ] Crear modelo para corners over/under
- [ ] Predicción de corners totales
- [ ] Hándicap asiático de corners
- [ ] Integrar en menú

### 1.5 Score Exacto
- [ ] Crear modelo para score exacto
- [ ] Top 3 scores más probables
- [ ] Integrar en menú

### 1.6 Ambos Equipos Marcan (BTTS)
- [ ] Crear modelo Sí/No
- [ ] Integrar en menú

### 1.7 Doble Oportunidad
- [ ] Crear modelo 1X, X2, 12
- [ ] Integrar en menú

### 1.8 Medición de Accuracy
- [ ] Comparar predicciones vs resultados reales
- [ ] Dashboard de métricas (accuracy por mercado)
- [ ] Historial de predicciones

---

## Fase 2: Escalamiento a Otras Ligas

### 2.1 LaLiga (España)
- [ ] Obtener datos históricos
- [ ] Limpiar datos con formato actual
- [ ] Calcular competitividad de la liga
- [ ] Entrenar modelos
- [ ] Crear menú específico

### 2.2 Serie A (Italia)
- [ ] Obtener datos históricos
- [ ] Limpiar datos con formato actual
- [ ] Calcular competitividad de la liga
- [ ] Entrenar modelos
- [ ] Crear menú específico

### 2.3 Bundesliga (Alemania)
- [ ] Obtener datos históricos
- [ ] Limpiar datos con formato actual
- [ ] Calcular competitividad de la liga
- [ ] Entrenar modelos
- [ ] Crear menú específico

### 2.4 Ligue 1 (Francia)
- [ ] Obtener datos históricos
- [ ] Limpiar datos con formato actual
- [ ] Calcular competitividad de la liga
- [ ] Entrenar modelos
- [ ] Crear menú específico

---

## Fase 3: Scraping y APIs

### 3.1 Scraping Google ✅
- [x] Script scraper.py creado
- [ ] Actualizar selectores cuando fallen
- [ ] Manejo de CAPTCHA

### 3.2 Football-data.org API
- [ ] Integrar API para fixtures en vivo
- [ ] Actualización automática de datos

---

## Métricas de Éxito

| Mercado | Target Accuracy |
|---------|-----------------|
| Ganador (1X2) | > 55% |
| Over/Under 2.5 | > 55% |
| BTTS | > 55% |
| Score Exacto | > 15% |
| Corners | > 55% |

---

## Análisis de Competitividad - Premier League

### Resultado Actual
```
Temporada 2023: CV=0.373, Score=0.186
Temporada 2024: CV=0.346, Score=0.173
Temporada 2025: CV=0.310, Score=0.155

Global: MODERADAMENTE COMPETITIVA (0.38)
```

### Implicaciones
- Esperar más empates de lo normal
- Reducir confianza en favoritos claros
- Considerar más "sorpresas"
- Usar umbrales conservadores

---

## Estructura de Datos por Mercado

### Ganador (1X2)
```python
{
    "home_win_prob": 0.45,
    "draw_prob": 0.30,
    "away_win_prob": 0.25,
    "prediction": "HOME",
    "confidence": 0.72,
    "competitiveness": 0.38,
    "competitiveness_level": "MODERADAMENTE COMPETITIVA"
}
```

### Over/Under Goles
```python
{
    "market": "over_2.5",
    "over_prob": 0.58,
    "under_prob": 0.42,
    "prediction": "OVER",
    "confidence": 0.65
}
```

### Corners
```python
{
    "market": "corners_over_9.5",
    "over_prob": 0.55,
    "under_prob": 0.45,
    "predicted_total": 10.2,
    "prediction": "OVER",
    "confidence": 0.60
}
```

### Score Exacto
```python
{
    "market": "exact_score",
    "top_scores": [
        {"score": "2-1", "prob": 0.15},
        {"score": "1-1", "prob": 0.12},
        {"score": "2-0", "prob": 0.10}
    ]
}
```

### BTTS
```python
{
    "market": "btts",
    "yes_prob": 0.55,
    "no_prob": 0.45,
    "prediction": "YES",
    "confidence": 0.58
}
```

---

## Comandos de Desarrollo

```bash
# Entrenar modelos
python main.py --train

# Ver menú
python main.py

# Probar competitividad
python test_competitiveness.py

# Probar modelo específico
python -c "from src.prediction_models import MatchPredictor; ..."
```

---

## Notas
- Priorizar accuracy sobre cantidad de features
- Un modelo bien probado es más fácil de replicar
- Medir siempre vs resultados reales
- La competitividad ajusta probabilidades, no el accuracy base
- Cada liga tiene su propia competitividad - calcular por separado
