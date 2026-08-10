# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See also `.claude/CLAUDE.md` for the safesports-panel export integration (endpoints, payload format, required env vars) — not repeated here.

## Setup & Commands

```bash
python -m venv premier-league-env
source premier-league-env/bin/activate
pip install -r requirements.txt

python main.py            # interactive menu (auto-updates data on startup, 6h cache)
python main.py --train    # retrain models on current cleaned data
python main.py --jornada  # detailed matchday predictor (includes dashboard export prompt)
python main.py --v2       # run the src_v2/ reorganized pipeline instead of src/
python main.py --help

python test_competitiveness.py   # standalone script exercising the competitiveness module, not pytest
python scripts/inspect_cleaned.py  # rich/duckdb preview of data/ligamx/cleaned/*.csv
```

There is no formal test suite (no pytest config, no `tests/` dir) — `test_competitiveness.py` is a manual smoke script run directly.

## Architecture

### Two parallel implementations
- **`src/`** — the actively used pipeline, driven by `main.py` → `src/menu_interface.py` (`PredictionMenu`, ~1500 lines). Handles training, jornada/individual/team-stats menu modes, competitiveness adjustment, and export to safesports-panel.
- **`src_v2/`** — a layered reorganization (`data/`, `features/`, `models/`, `evaluation/`, `ui/` packages under `src_v2/`) reached via `python main.py --v2` or directly via `src_v2/predict.py` / `src_v2/train.py`. Not the default path; treat as the "cleaner" rewrite in progress, not a dead branch.
- Root-level `scraper.py`, `simple_data_update.py`, `train_advanced.py`, `update_premier_data.py` are standalone utility scripts (data refresh / experimentation), separate from both `src/` and `src_v2/`.

### Data flow (Premier League, v1 path)
1. `update_premier_data.py` pulls from football-data.org (`X-Auth-Token` is currently **hardcoded** in `update_premier_data.py` and `src/data_collection.py`, not read from env — be aware when touching those files) into `data/raw/`.
2. `src/data_cleaning.py` normalizes into `data/cleaned/matches_{season}_cleaned.csv` / `standings_{season}_cleaned.csv`.
3. `src/feature_engineering.py` + `src/advanced_feature_engineering.py` build model features from the cleaned data.
4. `src/prediction_models.py` / `src/advanced_prediction_models.py` train and load models; artifacts are plain pickles in `models/` (`winner_predictor.pkl`, `random_forest.pkl`, `gradient_boosting.pkl`, `logistic_regression.pkl`, `goals_predictor.pkl`, `over_under_rf.pkl`/`over_under_gb.pkl`, plus matching `*_scaler.pkl`/`*_columns.pkl`/`*_features.pkl`). There is no model versioning — retraining overwrites these in place.
5. `src/menu_interface.py` renders predictions and optionally POSTs them to safesports-panel (see `.claude/CLAUDE.md`).

`main.py` runs step 1–2 automatically on every launch unless `.update_cache.json` shows a run within the last 6 hours (bypass with the internal `force` path, not a CLI flag); it also snapshots finished-match/matchday counts before/after to detect a stuck update.

### Season handling gotcha
`src/season_utils.py` defines the "current season" as the year-of-start, switching over in **July** (not January) to match the Premier League calendar: month ≥ 7 → current year, else previous year. Because a new season's `matches_{year}_cleaned.csv` exists but is empty of `FINISHED` rows before matches are played, use `get_latest_finished_season()` rather than `get_current_season()` whenever you need the season to actually evaluate/predict against — several call sites intentionally distinguish these two.

### Liga MX / multi-league expansion (in progress, uncommitted)
`docs/plan_5_ligas_ligamx.md` is the working plan — read it before touching Liga MX code. Key points:
- Goal is a config-driven multi-league core (Premier + 4 more European leagues + Liga MX), with a 3-model ensemble (XGBoost outcome, Poisson GLM O/U, Dixon-Coles scoreline) ported from the sibling repo `MML-Mundial`. None of that ensemble/core refactor exists yet in this repo.
- Liga MX is scoped to **regular season only** (Apertura/Clausura) — playoffs/liguilla are explicitly excluded everywhere (`PLAYOFF_MARKERS` filtering in the build scripts).
- Three data sources feed `data/ligamx/`: `src/build_liga_mx_dataset.py` (API-Football, `league_id=262`, needs `API_FOOTBALL_API_KEY` in `.env.local`/`.env`), `src/build_ligamx_fbref_dataset.py` (a public GitHub CSV mirror, FBref itself is Cloudflare-blocked from this environment), and `src/build_ligamx_transfermarkt_dataset.py` (a live Transfermarkt scraper for the current 25/26–26/27 seasons). Each is a standalone, re-runnable script that caches its output as CSV/JSON under `data/ligamx/`.

## Environment variables
Loaded from `.env`/`.env.local` at the project root (see `.env.example`); `src/menu_interface.py:load_env_files()` parses these manually rather than via `python-dotenv`. Panel-related vars are documented in `.claude/CLAUDE.md`; Liga MX work additionally needs `API_FOOTBALL_API_KEY`.
