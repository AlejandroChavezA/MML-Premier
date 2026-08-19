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

# core/ ensemble (uncommitted, config-driven multi-league — see Liga MX section below)
python -m core.train_ensemble --data-dir data/cleaned --models-dir models/premier_league
python -m core.evaluate_ensemble --league premier_league   # gate: ensemble vs legacy sklearn, chronological holdout
python -m core.evaluate_ensemble --league liga_mx          # same gate; liga_mx has no legacy, compares vs naive home-win baseline
python run.py jornada --league liga_mx --phase AP --next --send   # Liga MX predictions (menu_interface.py is Premier-only, no league picker)

# GB+RF ensemble (src_v2, ported from MML-Mundial — see "Related repositories" below)
python -m src_v2.train_winner_ensemble --data-dir data/ligamx/cleaned --models-dir models/liga_mx   # per-league trainer, chronological split
python -m core.evaluate_winner_ensemble --league premier_league   # 3-way backtest: GB+RF vs core/'s XGBoost vs legacy/baseline
```

There is no formal test suite (no pytest config, no `tests/` dir) — `test_competitiveness.py` is a manual smoke script run directly.

## Architecture

### Three generations of pipeline, only one in production
- **`src/`** — the actively used pipeline, driven by `main.py` → `src/menu_interface.py` (`PredictionMenu`, ~1500 lines). Handles training, jornada/individual/team-stats menu modes, competitiveness adjustment, and export to safesports-panel. **Known-unreliable evaluation methodology**: `train_test_split` is random (not chronological) and standings features ignore `match_date` (leak future results into the past) — see the deprecation comments at the top of `src/prediction_models.py` / `src/advanced_prediction_models.py`. Still the production default; don't invest further in fixing it in place.
- **`src_v2/`** — a layered reorganization (`data/`, `features/`, `models/`, `evaluation/`, `ui/` packages under `src_v2/`) reached via `python main.py --v2` or directly via `src_v2/predict.py` / `src_v2/train.py`. Not the default path. Its `features/feature_engineer.py` already fixed the standings leakage with point-in-time features (`get_standings` filters `date < match_date`) — `core/` reuses this feature engineer rather than re-fixing `src/`. **This is where `gradient_boosting_v2` actually comes from**: `src_v2/train.py` trains `src_v2/models/winner_predictor.py`'s `WinnerPredictor` and saves it as `models/winner_predictor.pkl`, which `src/prediction_models.py` then loads directly as the `gradient_boosting_v2` model — a real, working training script that a narrowly-scoped `grep src/*.py` will not find (see "Related repositories" below for how this was missed once already). `WinnerPredictor` is a GB(80%)+RF(20%) weighted ensemble (ported from `MML-Mundial`, not a single GB) with an optional `dates` param for chronological-split training (`core/eval_utils.py`); the per-league trainer is `src_v2/train_winner_ensemble.py`, separate from `src_v2/train.py` (which stays Premier-only and untouched to avoid silently changing the live `models/winner_predictor.pkl`).
- **`core/` + `config/` + `models/{league}/`** — the newest layer (uncommitted), a config-driven multi-league ensemble (XGBoost outcome + Poisson GLM O/U + Dixon-Coles scoreline, ported from the sibling repo `MML-Mundial`). Reuses `src_v2`'s feature engineer, trains/evaluates with a chronological split (`core/eval_utils.py`), and has a promotion gate (`core/evaluate_ensemble.py`) that only recommends switching production over once the ensemble ties or beats whatever legacy baseline exists for that league — see `docs/plan_5_ligas_ligamx.md`. This is the intended long-term replacement for `src/`; new model/eval work should go here, not into `src/`.
- Root-level `scraper.py`, `simple_data_update.py`, `train_advanced.py`, `update_premier_data.py` are standalone utility scripts (data refresh / experimentation), separate from all three of the above.

### Before touching model/evaluation code, read this
Two mistakes cost a full session here already — don't repeat them.

1. **Search all three generations, not just one.** `src/`, `src_v2/`, and `core/` each have their
   own model code with overlapping names. A `grep` scoped to only `src/*.py` for "where does
   `gradient_boosting_v2` get trained" missed `src_v2/models/winner_predictor.py` entirely and
   produced a confidently wrong answer ("no training script exists"). When asked "where does X
   come from" / "is Y a bug", check all three directories before answering, not the one that
   seems most obviously relevant.
2. **Sibling repos may already have the answer — check before rebuilding.** `docs/plan_5_ligas_ligamx.md`
   names `MML-Mundial` (sibling repo, `/Users/sas/Documents/Github/ModelosML/MML-Mundial`) as the
   source the `core/` ensemble was ported from, but a full session of methodology work happened
   here before anyone actually opened that repo. It has a genuinely more complete ensemble
   (`src_v2/models/winner_predictor.py`: GB 80% + RF 20% + LR 0%, vs. the GB-only version that got
   copied into this repo) and tooling that doesn't exist here at all (`experiments.py`:
   `run_feature_ablation()`, `run_hyperparameter_tuning()` → `models/best_config.json`,
   `run_calibration_analysis()` for Brier score; `AUDITORIA.md` as a template for a
   severity-ranked bug audit). Check `MML-Mundial` before reinventing evaluation or ensembling
   logic for this repo.
3. **Docs — including this file and `MML-Mundial`'s `SUMMARY.md`/`AGENTS.md`/`README.md` — are not
   ground truth. Verify claims against the actual code before repeating them.** Confirmed false
   claims found by actually reading the code instead of trusting the docs: this file once said
   "the ensemble/core refactor does not exist yet" while `core/` already had substantial working
   code; `MML-Mundial/SUMMARY.md` claims an "80/20 time-series split" for its `WinnerPredictor`,
   but the actual `train()` method uses plain random `train_test_split` — a separate, disconnected
   experiment (`experiments.py:run_time_series_validation`) does the real chronological split and
   was never wired into training. Treat split/methodology claims in any `.md` as unverified until
   you've read the function that supposedly implements them. `MML-Mundial/README.md` is also
   stale/generic (leftover from an earlier unrelated template) — don't assume a repo's README
   describes its current state.

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
- Goal is a config-driven multi-league core (Premier + 4 more European leagues + Liga MX) — see the `core/` generation above. Liga MX already has trained artifacts in `models/liga_mx/` and a config at `config/leagues/liga_mx.yaml`.
- Liga MX is scoped to **regular season only** (Apertura/Clausura) — playoffs/liguilla are explicitly excluded everywhere (`PLAYOFF_MARKERS` filtering in the build scripts).
- Three data sources feed `data/ligamx/`: `src/build_liga_mx_dataset.py` (API-Football, `league_id=262`, needs `API_FOOTBALL_API_KEY` in `.env.local`/`.env`), `src/build_ligamx_fbref_dataset.py` (a public GitHub CSV mirror, FBref itself is Cloudflare-blocked from this environment), and `src/build_ligamx_transfermarkt_dataset.py` (a live Transfermarkt scraper for the current 25/26–26/27 seasons, `PAGES` list at the top of the file — add a season/phase there before expecting `build_ligamx_cleaned_dataset.py` to pick it up). Each is a standalone, re-runnable script that caches its output as CSV/JSON under `data/ligamx/`.
- Liga MX has **no legacy sklearn baseline** (`src/prediction_models.py` was only ever trained on Premier data) — `LeagueConfig.has_legacy_baseline` (`core/league_config.py`) is `false` for `liga_mx`, and `core/evaluate_ensemble.py` compares against a naive "always home win" baseline instead of crashing/silently returning 0% against a Premier-only legacy.
- Liga MX predictions are CLI-only for now: `python run.py jornada --league liga_mx --phase AP|CL --next|--last-finished|--matchday N [--send]`. The interactive menu (`python main.py`) is still Premier-only ("PREDICTOR PREMIER LEAGUE", no league picker).
- Production prediction history: `data/prediction_history.json` now carries a `league` field (`core/history.py` for the multi-league writer used by `run.py`; `src/menu_interface.py:_save_prediction_to_history` tags Premier entries the same way). Entries from before this field existed are all Premier.
- Model quality, as of this writing: on a chronological holdout, `core/`'s XGBoost ensemble and the ported GB+RF ensemble (`src_v2/train_winner_ensemble.py`) both land around 44-47% for both leagues — barely above (Premier) or exactly tied with (Liga MX) the naive "always home win" baseline, and both lose clearly to the legacy `gradient_boosting_v2` in Premier (66.1% on the same holdout — but note that number is a single production model's track record, not yet reproduced from a documented training recipe with this methodology). Run `python -m core.evaluate_winner_ensemble --league <slug>` for the current numbers before assuming any of the three is production-ready for Liga MX.

## Environment variables
Loaded from `.env`/`.env.local` at the project root (see `.env.example`); `src/menu_interface.py:load_env_files()` parses these manually rather than via `python-dotenv`. Panel-related vars are documented in `.claude/CLAUDE.md`; Liga MX work additionally needs `API_FOOTBALL_API_KEY`.
