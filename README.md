<div align="center">

# ⚽ MML-Premier

**Sistema de predicción de partidos de fútbol basado en Machine Learning**

Premier League · Liga MX (en expansión) · Integración con [safesports-panel](https://safesports-panel.vercel.app)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![status](https://img.shields.io/badge/status-en%20desarrollo-yellow)

</div>

---

## 📖 Contenido

- [¿Qué hace este proyecto?](#-qué-hace-este-proyecto)
- [Arquitectura](#-arquitectura)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Pipeline de datos](#-pipeline-de-datos)
- [Modelos](#-modelos)
- [Integración con el dashboard](#-integración-con-el-dashboard)
- [Expansión multi-liga](#-expansión-multi-liga-en-progreso)
- [Variables de entorno](#-variables-de-entorno)
- [Notas y limitaciones](#-notas-y-limitaciones)

---

## 🎯 ¿Qué hace este proyecto?

`MML-Premier` recolecta datos históricos de partidos, entrena modelos de Machine Learning
(clasificación de resultado, goles, over/under) y expone un **menú interactivo por consola**
para generar predicciones jornada a jornada. Las predicciones pueden exportarse directamente
al panel de administración [safesports-panel](https://safesports-panel.vercel.app).

Actualmente soporta **Premier League** como liga principal, con una expansión en curso hacia
**Liga MX** y otras 4 ligas europeas (ver [Expansión multi-liga](#-expansión-multi-liga-en-progreso)).

## 🏗 Arquitectura

```
[football-data.org API] ─▶ [data/raw] ─▶ [limpieza] ─▶ [data/cleaned]
                                                             │
                                                             ▼
                                          [feature engineering] ─▶ [modelos sklearn]
                                                             │
                                                             ▼
                                          [menú interactivo] ─▶ [safesports-panel]
```

El repo mantiene **dos implementaciones en paralelo**:

| Pipeline | Entrypoint | Estado |
|---|---|---|
| `src/` | `python main.py` | Activo, es el que usa `main.py` por defecto |
| `src_v2/` | `python main.py --v2` | Reorganización en capas (`data/`, `features/`, `models/`, `evaluation/`, `ui/`), en progreso |

## 🚀 Instalación

```bash
# 1. Clonar el repo
git clone https://github.com/AlejandroChavezA/MML-Premier.git
cd MML-Premier

# 2. Crear y activar entorno virtual
python -m venv premier-league-env
source premier-league-env/bin/activate      # Windows: premier-league-env\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env.local
# edita .env.local con tus credenciales (ver sección Variables de entorno)
```

## 🕹 Uso

```bash
python main.py            # Menú interactivo (actualiza datos automáticamente, caché de 6h)
python main.py --train    # Reentrena los modelos con los datos limpios actuales
python main.py --jornada  # Predictor detallado por jornada (incluye export al dashboard)
python main.py --v2       # Ejecuta el pipeline reorganizado en src_v2/
python main.py --help     # Ver todas las opciones
```

### Menú principal

```
1.  Predicción de jornada completa (Ganador + O/U)
2.  Predicción por jornada (detalles)      ← incluye envío al dashboard
3.  Predicción partido por partido (Ganador + O/U)
4.  Estadísticas de equipos
5.  Ver tabla de posiciones actual
6.  Cambiar modelo de predicción
7.  Rendimiento de modelos
8.  Limpiar caché O/U
9.  Salir
10. Exportar historial al panel
```

### Scripts sueltos

```bash
python test_competitiveness.py     # smoke test manual del módulo de competitividad
python scripts/inspect_cleaned.py  # preview con rich/duckdb de data/ligamx/cleaned/*.csv
```

> No hay suite de tests formal (sin `pytest`, sin carpeta `tests/`).

## 📂 Estructura del repositorio

```
MML-Premier/
├── main.py                    # Entrypoint del menú interactivo
├── update_premier_data.py     # Descarga datos desde football-data.org
├── src/                       # Pipeline activo (v1)
│   ├── menu_interface.py      # Menú + envío al dashboard
│   ├── data_cleaning.py       # Normalización de datos crudos
│   ├── feature_engineering.py # Features para los modelos
│   ├── prediction_models.py   # Entrenamiento/carga de modelos sklearn
│   ├── advanced_*.py          # Features y modelos avanzados
│   ├── season_utils.py        # Lógica de "temporada actual"
│   ├── competitiveness.py     # Ajuste de competitividad
│   └── build_ligamx_*.py      # Scripts de ingestión de Liga MX
├── src_v2/                    # Reorganización en capas (en progreso)
│   ├── data/ features/ models/ evaluation/ ui/
│   └── predict.py train.py
├── models/                    # Modelos entrenados (.pkl) — sin versionado, se sobreescriben
├── data/
│   ├── raw/ cleaned/          # Datos de Premier League
│   └── ligamx/                # Datos de Liga MX (fase liga, sin liguilla)
├── scripts/                   # Utilidades de inspección de datos
├── docs/                      # Planes y documentación de expansión
└── .claude/CLAUDE.md          # Detalle de la integración con safesports-panel
```

## 🔄 Pipeline de datos

1. **Recolección** — `update_premier_data.py` consulta la API de
   [football-data.org](https://www.football-data.org/) y guarda crudo en `data/raw/`.
2. **Limpieza** — `src/data_cleaning.py` normaliza a
   `data/cleaned/matches_{season}_cleaned.csv` / `standings_{season}_cleaned.csv`.
3. **Features** — `src/feature_engineering.py` y `src/advanced_feature_engineering.py`
   generan las variables de entrada de los modelos.
4. **Modelos** — `src/prediction_models.py` / `src/advanced_prediction_models.py`
   entrenan y cargan los `.pkl` en `models/` (Random Forest, Gradient Boosting,
   Regresión Logística, predictor de goles, Over/Under).
5. **Menú** — `src/menu_interface.py` genera las predicciones y, opcionalmente,
   las envía a safesports-panel.

`main.py` corre los pasos 1–2 automáticamente en cada ejecución, salvo que
`.update_cache.json` indique una actualización en las últimas 6 horas.

> **Nota sobre temporadas**: la temporada "actual" cambia en **julio** (no en enero),
> siguiendo el calendario de la Premier League (`src/season_utils.py`). Para evaluar o
> predecir contra datos ya jugados, usar `get_latest_finished_season()` en vez de
> `get_current_season()`.

## 🤖 Modelos

| Modelo | Tarea |
|---|---|
| Random Forest / Gradient Boosting / Regresión Logística | Predicción de ganador (1X2) |
| Goals Predictor | Predicción de goles |
| Random Forest / Gradient Boosting | Over/Under |

Los artefactos (`models/*.pkl`, `*_scaler.pkl`, `*_columns.pkl`, `*_features.pkl`) **no están
versionados** — cada reentrenamiento (`python main.py --train`) los sobreescribe in place.

## 📤 Integración con el dashboard

Las predicciones pueden exportarse a [safesports-panel](https://safesports-panel.vercel.app)
vía `/api/predictions/import`, autenticando con API key o credenciales de admin.
El detalle completo (formato de payload, endpoints, variables requeridas) está en
[`.claude/CLAUDE.md`](.claude/CLAUDE.md) y [`API_PREDICCIONES.md`](API_PREDICCIONES.md).

## 🌎 Expansión multi-liga (en progreso)

Trabajo en curso descrito en [`docs/plan_5_ligas_ligamx.md`](docs/plan_5_ligas_ligamx.md):

- Núcleo *config-driven* para Premier + 4 ligas europeas + Liga MX.
- Ensamble de 3 modelos (XGBoost outcome, Poisson GLM O/U, Dixon-Coles scoreline),
  portado desde el repo hermano `MML-Mundial`.
- **Liga MX** se limita a **fase regular** (Apertura/Clausura) — la liguilla/playoffs
  queda explícitamente excluida en todos los scripts de construcción de datos.
- Tres fuentes alimentan `data/ligamx/`:
  - `src/build_liga_mx_dataset.py` — API-Football (`league_id=262`)
  - `src/build_ligamx_fbref_dataset.py` — mirror público en GitHub (FBref bloquea Cloudflare)
  - `src/build_ligamx_transfermarkt_dataset.py` — scraper en vivo de Transfermarkt

Ninguno de estos pasos modifica todavía el core de Premier; son scripts independientes
y re-ejecutables que cachean su salida como CSV/JSON.

## 🔐 Variables de entorno

Se cargan desde `.env` / `.env.local` en la raíz del proyecto (ver
[`.env.example`](.env.example)).

| Variable | Descripción |
|---|---|
| `SAFESPORTS_PANEL_URL` | URL del panel (ej. `http://localhost:3000`) |
| `SAFESPORTS_PANEL_EMAIL` / `SAFESPORTS_PANEL_PASSWORD` | Credenciales de admin para generar API key |
| `SAFESPORTS_USER_API_KEY` | API key directa (alternativa a email/password) |
| `IMPORT_API_SECRET` | Secret compartido para importación masiva |
| `API_FOOTBALL_API_KEY` | Necesaria para los scripts de ingestión de Liga MX |

## ⚠️ Notas y limitaciones

- El `X-Auth-Token` de football-data.org está **hardcodeado** en `update_premier_data.py`
  y `src/data_collection.py` (no se lee de variables de entorno) — tenerlo en cuenta antes
  de tocar esos archivos.
- No hay versionado de modelos: reentrenar sobreescribe los `.pkl` existentes.
- El scraping (Google, Transfermarkt) puede fallar; conviene manejar errores al modificarlo.
