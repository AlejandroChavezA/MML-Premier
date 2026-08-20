# Plan: 5 ligas + Liga MX (fase liga) + ensamble de modelos

## Objetivo
Unificar el sistema para:
- Reconciliar 3 modelos en una sola prediccion final.
- Expandir desde Premier a 5 ligas europeas.
- Incluir **Liga MX solo en fase liga** (Apertura/Clausura regular season), sin liguilla/playoffs.

## Estado actual verificado
- El repo de Premier tiene pipeline v1/v2, menu interactivo y exportacion a panel.
- El repo `MML-Mundial` tiene el trio reutilizable de modelos:
  - XGBoost outcome (`train_final_v3.py`)
  - Poisson GLM O/U (`train_goals_glm.py`)
  - Dixon-Coles scoreline (`train_dixon_coles.py`)
- API-Football en `.env.local` **si** expone Liga MX (`league_id=262`) y tiene histórico libre hasta 2024.
- FBref fue probado como fuente alternativa para Liga MX, pero desde este entorno responde con challenge de Cloudflare (403) incluso con Playwright, así que **no es fuente viable por ahora** sin un scraper más pesado/fragil o un proxy estable.
- Encontramos un dataset público usable en GitHub (`yaacob117/Liga_MX_prediction_matches/matches.csv`) con Liga MX 2021-2024, xG, alineaciones y más. Ya quedó descargado y filtrado: `2320` partidos de fase regular (`data/ligamx/fbref_matches_regular.csv`). Ese dataset sí sirve como **fuente suplementaria** para fase liga.
- `Transfermarkt` sí responde desde aquí y expone calendario/standings de Liga MX 25/26 y 26/27; se usará como **fuente viva** para completar lo más reciente que falta.
- Liga MX **no** entra con liguilla en este repo.

## Restricciones
- No hardcodear 38 jornadas ni nombres de liga en el core.
- Menús deben quedar como capa delgada.
- Todo export debe validar esquema antes de salir.
- ELO por liga aislada.

## Paso 0: Dataset Liga MX (fase liga)
### Objetivo
Dejar la data lista para trabajar solo la fase regular (Apertura/Clausura) y no playoffs.

### Pasitos
1. Descargar y cachear dos fuentes:
   - API-Football `league_id=262` (fixtures/standings 2022-2024).
   - Dataset GitHub `yaacob117/Liga_MX_prediction_matches/matches.csv` (2021-2024, incluye xG y contexto).
   - `Transfermarkt` como fuente viva para temporadas 25/26 y 26/27.
2. Guardar por temporada en `data/ligamx/`.
3. Filtrar solo rondas regulares (`Apertura-*` y `Clausura-*`) **excluyendo** `Play-offs`, `Quarter-finals`, `Semi-finals` y `Final`.
4. Documentar que la liguilla queda fuera de este repo y no se exporta en el dataset.
5. Verificar que las temporadas libres disponibles al menos cubran 2022-2024, que el dataset GitHub aporte 2021-2024, y que Transfermarkt complete 25/26 + 26/27.

### Entregable
- `data/ligamx/ligamx_2022.json`
- `data/ligamx/ligamx_2023.json`
- `data/ligamx/ligamx_2024.json`
- `data/ligamx/ligamx_2022.csv`
- `data/ligamx/ligamx_2023.csv`
- `data/ligamx/ligamx_2024.csv`
- `data/ligamx/fbref_matches_raw.csv`
- `data/ligamx/fbref_matches_regular.csv`

## Paso 1: Contrato de exportacion
### Objetivo
Definir un schema unico de salida y evitar exports silenciosamente rotos.

### Pasitos
1. Crear `schemas.py` con `MatchPrediction` (pydantic).
2. Incluir campos fijos para dashboard y app.
3. Validar toda exportacion antes de escribir o postear.
4. Falla ruidosa si falta un campo o el tipo no cuadra.
5. Mantener versionado agregando campos al final.

### Verificacion
- Export Premier actual pasa el schema sin cambios.

## Paso 2: Refactor del menu a funciones + CLI
### Objetivo
Separar UI de logica de negocio.

### Pasitos
1. Identificar las 9 opciones actuales del menu.
2. Extraer cada opcion a una funcion explicita.
3. Convertir `menu_interface` en capa delgada.
4. Exponer las mismas capacidades por CLI (`run.py`).
5. Mantener comportamiento identico.

### Verificacion
- Cada opcion del menu sigue funcionando igual.

## Paso 3: Ensamble de modelos (Premier primero)
### Objetivo
Reconciliar outcome + O/U + scoreline en una sola prediccion.

### Pasitos
1. Reusar/adaptar los 3 modelos del repo `MML-Mundial`.
2. Crear un modulo generico `core/ensemble.py`.
3. Forzar consistencia entre outcome y marcador elegido.
4. Mantener el modulo agnostico de liga.
5. Probar una jornada completa de Premier.

### Verificacion
- Ninguna prediccion final contradice el outcome.

## Paso 4: Multi-liga config-driven
### Objetivo
Soportar Premier, LaLiga, Bundesliga, Ligue 1, Serie A y luego Liga MX sin duplicar core.

### Pasitos
1. Crear config por liga en `config/leagues/{league}.yaml`.
2. Declarar jornadas, temporada actual, `lower_league`, `winter_break`, features extra.
3. Sacar hardcode de jornadas y nombres de liga del core.
4. Parametrizar fallback de ascendidos por liga.
5. Entrenar modelos por liga, no pooled.
6. Guardar artefactos en `models/{league}/{model_type}/v{n}.pkl`.
7. Implementar cache local por liga para no romper rate limits.
8. Ajustar `rest_days` para ligas con pausa de invierno.
9. Mantener ELO por liga aislada.

### Verificacion
- `predict_week(league, week, season)` devuelve predicciones validas y pasa schema en las ligas soportadas.

## Paso 5: Integrar Liga MX (solo fase liga)
### Objetivo
Agregar Liga MX al mismo framework, pero solo regular season.

### Pasitos
1. Agregar `config/leagues/liga_mx.yaml`.
2. Definir `league_id=262` y el mapeo de temporadas disponibles.
3. Tratar Apertura/Clausura como temporadas/lotes regulares.
4. Ignorar la liguilla por completo en este repo.
5. Construir features y modelos con el mismo core que las otras ligas.

### Verificacion
- Liga MX regular season corre con el mismo `predict_week`.

## Riesgos ya conocidos
1. `rate limit` de API-Football => cache obligatorio.
2. Ligas inferiores europeas pueden no estar disponibles en free => usar baseline genérico si falta la lower league.
3. Liga MX necesita tratar Apertura/Clausura como fase liga y dejar playoffs fuera.

## Orden de trabajo
1. Paso 0
2. Paso 1
3. Paso 2
4. Paso 3
5. Paso 4
6. Paso 5

## Pendientes / cosas por ver despues
1. **Huecos reales en Transfermarkt (2025/2026) -- RESUELTO**: `matches_2025_cleaned.csv`
   le faltaban 4 partidos (jornadas 4, 6, 9 y 12 del Apertura, 8 partidos en vez
   de 9 cada una) y `matches_2026_cleaned.csv` le faltaban 3 (jornada 4: -2,
   jornada 8: -1). Causa real: `PAGES` en `src/build_ligamx_transfermarkt_dataset.py`
   nunca tuvo una entrada para Apertura 2025 (solo Clausura 2025 y Apertura 2026);
   el hueco de 2026 ya se habia corregido en una sesion anterior re-scrapeando esa
   pagina. Se agrego la pagina de Apertura 2025 (`saison_id/2025`, mismo patron que
   Apertura 2026), se corrigio un bug de parseo de asistencia (`_club_rows_from_page`
   asumia que "Attendance" siempre era numerico; Transfermarkt a veces pone "x") y
   se re-corrio el scraper + `build_ligamx_cleaned_dataset.py`. `matches_2025_cleaned.csv`
   ahora tiene 306/306 partidos (9 por jornada, ambas fases) y `matches_2026_cleaned.csv`
   sigue completo.
2. **Liga de Expansion MX / Ascenso MX (openfootball) -- RESUELTO**: se
   escribio el parser del formato de texto (`src/build_ligamx_openfootball_dataset.py`)
   y se resolvio la duda de integracion con una division en dos:
   - Primera division (`*_mx1.txt`) entra como **4ta fuente** de
     `build_ligamx_cleaned_dataset.py` (`SOURCE_PRIORITY`, ultima prioridad).
     Aporta temporadas 2010-2018 que antes no existian (`matches_2010..2018_cleaned.csv`,
     306 partidos c/u) y cruza contra fbref/api_football en 2020-2023
     (`n_sources` sube a 2-3 donde antes era 1). La 2019 queda con 324
     partidos / 261 jugados porque el Clausura 2020 se cancelo por covid a
     mitad de la jornada 10 (63 partidos quedan en `status=CANCELLED`, no se
     inventan resultados).
   - Segunda division (`*_mx2ascenso.txt` / `*_mx2expansion.txt`) se limpia
     **por separado**, sin mezclar con los resultados de Liga MX (es otra
     liga, otro pool de equipos): `segunda_matches_{season}_cleaned.csv`,
     2010-2024. Pensado para features de continuidad de equipos recien
     ascendidos (Atlante tiene 289 partidos de segunda entre 2010 y 2024
     repartidos en varias temporadas).
   - Bug encontrado y arreglado de paso: `_season_of`/`_phase_of` en
     `build_ligamx_cleaned_dataset.py` cortaban temporada/fase en julio: el
     Apertura a veces arranca la ultima semana de junio (ej. Apertura
     2023-24 empezo el 30-jun-2023), y esos partidos de jornada 1 quedaban
     mal clasificados como Clausura de la temporada anterior. El corte se
     movio a junio (Clausura de Liga MX nunca pasa de mayo, ni en fase liga
     ni en liguilla, asi que no hay riesgo de solapamiento). Esto ya afectaba
     a `matches_2022_cleaned.csv` con fbref antes de tocar openfootball.
3. **`core/panel_export.py` manda argumentos genericos (Liga MX y cualquier liga
   del ensamble nuevo)**: se arreglo el bug de `homeTeamFullName`/`awayTeamFullName`
   (mandaba el codigo de 3 letras en vez del nombre completo -- mismo bug que tenia
   `PredictionMenu.get_team_full_name` en `src/menu_interface.py`, ya corregido ahi
   tambien) y se conectaron argumentos reales (`forWinner`/`forLoser` armados con
   `points_diff`/`form_diff`/`h2h`/etc. via `menu_actions.build_match_arguments`,
   nuevo en `src/menu_actions.py`) para el pipeline legacy de Premier
   (`transform_to_panel_format` y `export_history_to_panel_format`). Pero
   `ensemble_predictions_to_panel_format` (el que usa Liga MX vía `run.py`) sigue
   mandando solo `"Confianza del ensamble: X%"` -- las features crudas que
   `OutcomeXGB.predict_proba` calcula (via `feature_engineer.create_match_features`)
   no salen de ahi, se pierden en `core/ensemble.py::reconcile()` antes de llegar a
   `core/predict_week.py`. Falta: pasar `raw_features` a traves de esas 3 capas
   (outcome_xgb -> reconcile -> predict_week -> panel_export) igual que ya se hizo
   en el lado legacy, y reusar `build_match_arguments` (ya tolera las claves de
   ambos feature engineers, legacy y src_v2).
