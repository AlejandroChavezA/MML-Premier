# Plan: Manejo de equipos recién ascendidos (Promoted Teams) y temporada dinámica

## Contexto

El sistema predice la Premier League usando datos de `football-data.org/v4` (token free). Tras cargar la
temporada 2026 (PL 2026-27, próxima), aparecieron equipos **sin historial PL 2023-2025** que rompían el
pipeline (`KeyError: 'unbeaten_streak'`) y cuyas features quedaban en cero.

Además, como 2026 era la temporada máxima cargada pero con `played_games = 0` para todos, las predicciones de
2025 (temporada real completada) habrían usado la tabla 2026 vacía, degradándose.

### Hallazgos verificados (solo lectura)
- `data/historical/*.csv` (2019-2024) están **SIMULADOS** (`download_historical_data.py` → "create realistic
  simulated data"). Solo 2023-2025 son reales (API).
- Equipos con **CERO historia PL** en el archivo 2026: **Coventry City FC** y **Hull City AFC** (exactamente 2).
  - Leeds United FC y Sunderland AFC están en el archivo 2025 (PL 2025-26, ya completada), no en 2026.
- La API key **SÍ tiene Championship (ELC, id 2016) desde 2023**:
  - `season=2023` (Championship 2023-24): ✅ 24 equipos
  - `season=2024` (Championship 2024-25): ✅ 24 equipos
  - `season=2025` (Championship 2025-26): ✅ 24 equipos  ← contiene a Coventry (1°, 95 pts) y Hull (6°, 73 pts)
  - `season=2022` (Championship 2021-22): ❌ restringido (no se usa)

## Objetivo
1. Eliminar la historia simulada.
2. Sembrar los equipos sin historia PL con su **forma real de Championship de la temporada de ascenso**,
   escalada al nivel PL mediante un factor calibrado con equipos ascendidos reales.
3. Que `current_season` apunte a la última temporada CON partidos finalizados (hoy 2025), para no contaminar
   predicciones de 2025 con la tabla 2026 vacía.

## Plan de ejecución

### 1. Borrar datos simulados
- Eliminar `data/historical/*.csv`.
- Eliminar `download_historical_data.py` (su única salida es historia falsa; el pipeline real no lo referencia).

### 2. Cache local de Championship — `src/build_championship_seeds.py` (nuevo)
- Hacer GET a `ELC standings?season=2023`, `?season=2024`, `?season=2025`, tabla `type=TOTAL`.
- Guardar en `data/championship_seeds.json` (una sola vez). Así las predicciones no golpean la API ni sufren
  rate-limit (free ≈ 10 req/min).
- Campos por equipo/temporada: `position`, `points`, `won`, `draw`, `lost`, `goalsFor`, `goalsAgainst`.

### 3. Factor de escala ascendido→PL (calibrado, real)
Usar los 5 equipos ascendidos reales cuyo Championship y PL están disponibles:
- Cohorte PL 2024-25 ← ELC 2023: Ipswich Town FC, Leicester City FC, Southampton FC.
- Cohorte PL 2025-26 ← ELC 2024: Leeds United FC, Sunderland AFC.
- (Burnley FC se excluye: tiene historia PL previa en nuestra ventana; ELC 2022 está restringido.)

Para cada uno calcular razones PL / Championship:
- `k_points = mean( PL_pts_primer_año / ELC_pts_ascenso )`
- `k_winrate`, `k_gf_pg`, `k_ga_pg` análogos (usando `standings_*` reales y `championship_seeds.json`).

Aplicar a Coventry/Hull (sus stats de ELC 2025):
- `seed_PL = stats_ELC2025 * factores_k`
- Posición PL aproximada = promedio de posición de ascendidos (del baseline n=6, paso 4).

### 4. Baseline genérico de ascendidos (n=6, real) — fallback
- `src/build_promoted_baseline.py` (nuevo): promedia la 1ª temporada PL real de los 6 equipos
  (Burnley FC, Luton Town FC, Sheffield United FC, Ipswich Town FC, Leicester City FC, Southampton FC) →
  guardar en `data/promoted_baseline.json`.
- Se usa solo si un equipo no tiene ni PL ni Championship disponible.

### 5. Fallback en `src/feature_engineering.py`
- En `_get_default_form()` / `calculate_team_form`, para equipos sin historia PL:
  - usar su **seed de Championship escalado** si existe (Coventry/Hull);
  - si no, el **baseline genérico n=6**.
- Mantener defaults (`unbeaten_streak`, etc.).
- Afecta a Coventry/Hull en predicciones 2026 (previo a que jueguen).

### 6. `current_season` = última temporada CON `FINISHED`
- En `season_utils` / `feature_engineering`: `current_season` = máxima temporada con partidos `FINISHED`.
- Hoy: 2025 (PL 2025-26, 380 finalizados) es current; 2026 (0 finalizados) queda como upcoming.
- Evita que 2025 use standings 2026 vacíos. **No se borra data 2026.**
- Autoresolución: al jugarse partidos de 2026, la historia de Coventry/Hull se acumula sola dentro de
  `matches_2026` y el seed deja de usarse (tras ~3-4 jornadas).

### 7. Validación
- `build_championship_seeds.py` e `build_promoted_baseline.py` imprimen factores k y seeds de Coventry/Hull;
  deben reflejar nivel PL realista (no los 95/73 crudos de Championship).
- `predict_week_matches(1, 2026)` → 10/10 sin error, Coventry/Hull con seed escalado.
- `predict_week_matches(37, 2025)` sigue con resultados reales.

## Fuera de alcance
- Traer Championship de temporadas anteriores a 2023 (ELC 2022 restringido).
- Fuentes externas de Championship / factores de escala más sofisticados.

## Archivos involucrados
- `data/historical/*.csv` — BORRAR
- `download_historical_data.py` — BORRAR
- `src/build_championship_seeds.py` — NUEVO
- `src/build_promoted_baseline.py` — NUEVO
- `data/championship_seeds.json` — NUEVO (cache)
- `data/promoted_baseline.json` — NUEVO (cache)
- `src/feature_engineering.py` — MODIFICAR fallback
- `src/season_utils.py` — MODIFICAR `current_season`
- `src/prediction_models.py`, `main.py` — verificar integración
