# Time-Series Demand Forecasting Service

End-to-end forecasting service for hourly electricity demand (PJM
dataset): feature pipeline, leakage-aware walk-forward validation, an
XGBoost model, and FastAPI serving with Postgres persistence and Redis
caching, containerised with docker-compose.

## Layout

src/dfs/features.py   lag / rolling / calendar features with leakage discipline
src/dfs/validate.py   expanding-window walk-forward harness and naive baselines
src/dfs/train.py      training CLI, writes metrics.json and model.pkl
src/dfs/service.py    FastAPI + Postgres + Redis, with graceful fallbacks
scripts/loadtest.py   load test for the serving layer
tests/test_all.py     includes a test proving the feature layer cannot leak
Dockerfile, docker-compose.yml, .github/workflows/ci.yml

## Setup

1. Download PJME_hourly.csv from Kaggle (robikscube/hourly-energy-consumption)
   into data/.
2. Train and evaluate:
   PYTHONPATH=src python -m dfs.train --csv data/PJME_hourly.csv --out artifacts/
3. Serve the stack: docker compose up --build
4. Load test: python scripts/loadtest.py http://localhost:8000

## Design

Walk-forward validation, not K-fold. Random splits let the model train
on the future and predict the past. Chronological expanding windows mirror
deployment: always predict forward, retrain as data arrives. Metrics are
pooled across all windows rather than averaged per window.

Two baselines. Persistence (y at t-1) is a strong baseline for
autocorrelated series at a one-hour horizon, so beating it is a meaningful
result. Seasonal-naive (same hour last week) is the standard yardstick for
seasonal data. Both are reported rather than the more flattering figure
alone.

Leakage. Rolling statistics computed without shifting first would
include the target in its own feature, a common bug in time-series
pipelines. The feature layer shifts by the forecast horizon before
computing any rolling window, and a test proves this: features built on
the full series are compared against features built on a series truncated
at time t, and the row at t must be identical in both.

Model choice. XGBoost on lag, rolling, and calendar features is a
standard, fast tabular baseline. The model itself is intentionally
unremarkable; the validation methodology is where the rigour is.

Recursive multi-step forecasting. /forecast?h=24 predicts one step,
appends it to the series, and repeats, so errors compound with the
horizon. A direct approach (one model per horizon) trades training cost
for accuracy at longer horizons; recursive was chosen for a single-model
service.

Caching. For a fixed model version, the forecast for a given
(last-observation, horizon) pair is deterministic, making it well suited
to caching. The cache key includes the model version, so a retrain
implicitly invalidates stale entries.

Graceful degradation. The service falls back to an in-process cache
and skips persistence when REDIS_URL/DATABASE_URL are not set, keeping
the same API contract so it is testable without external infrastructure.
docker-compose provides the full stack.
