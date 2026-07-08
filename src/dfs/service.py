"""service.py — Serving layer: FastAPI + Postgres (storage) + Redis (cache).

Architecture:
    client -> FastAPI -> Redis (cache hit? return)
                      -> feature build + model predict
                      -> Postgres (log prediction)  -> Redis (fill cache)

Both infra layers degrade gracefully: without REDIS_URL/DATABASE_URL the
service runs with an in-process LRU cache and no persistence, so the API
is testable anywhere; docker-compose provides the real stack.

Cache design decision to defend: forecasts for a given (timestamp, horizon)
are deterministic for a fixed model, so they're ideal cache material.
Key = model_version:horizon:timestamp, TTL = 1h (a retrain rotates
model_version, implicitly invalidating stale entries).

Endpoints:
    GET  /health              liveness + which layers are live
    GET  /forecast?h=24       forecast next h hours (cached)
    GET  /metrics             walk-forward metrics of the loaded model
    GET  /history?hours=48    recent observed demand from Postgres
"""

from __future__ import annotations

import json
import os
import pickle
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .features import build_features

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

_state: dict = {}


# ---------------------------- cache layer ----------------------------------
class InProcessLRU:
    """Fallback cache when Redis is absent (keeps the API contract identical)."""
    def __init__(self, cap=4096):
        self.cap, self.d = cap, OrderedDict()
    def get(self, k):
        v = self.d.get(k)
        if v is not None: self.d.move_to_end(k)
        return v
    def set(self, k, v, ttl=None):
        self.d[k] = v; self.d.move_to_end(k)
        if len(self.d) > self.cap: self.d.popitem(last=False)


def make_cache():
    url = os.environ.get("REDIS_URL")
    if url:
        import redis
        r = redis.Redis.from_url(url, decode_responses=True)
        r.ping()
        class RedisCache:
            def get(self, k): return r.get(k)
            def set(self, k, v, ttl=3600): r.set(k, v, ex=ttl)
        return RedisCache(), "redis"
    return InProcessLRU(), "in-process"


# --------------------------- storage layer ---------------------------------
def make_db():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None, "none"
    import psycopg
    conn = psycopg.connect(url, autocommit=True)
    conn.execute("SELECT pg_advisory_lock(917238)")
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY,
                requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                target_ts TIMESTAMPTZ NOT NULL,
                horizon_h INT NOT NULL,
                predicted_mw DOUBLE PRECISION NOT NULL,
                model_version TEXT NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                ts TIMESTAMPTZ PRIMARY KEY,
                mw DOUBLE PRECISION NOT NULL
            )""")
    finally:
        conn.execute("SELECT pg_advisory_unlock(917238)")
    return conn, "postgres"


# ----------------------------- app setup -----------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(os.path.join(ARTIFACT_DIR, "model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    hist = pd.read_csv(os.path.join(ARTIFACT_DIR, "recent_history.csv"),
                       index_col=0, parse_dates=True)["mw"]
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}

    cache, cache_kind = make_cache()
    db, db_kind = make_db()
    if db is not None:  # seed observations
        with db.cursor() as cur:
            cur.executemany(
                "INSERT INTO observations (ts, mw) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                [(ts.to_pydatetime(), float(v)) for ts, v in hist.items()])

    _state.update(model=bundle["model"], feats=bundle["feature_names"],
                  version=bundle["csv_sha256_16"], history=hist,
                  metrics=metrics, cache=cache, cache_kind=cache_kind,
                  db=db, db_kind=db_kind)
    yield
    if db is not None: db.close()


app = FastAPI(title="Demand Forecasting Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": _state["version"],
            "cache": _state["cache_kind"], "db": _state["db_kind"]}


@app.get("/metrics")
def metrics():
    return _state["metrics"]


@app.get("/history")
def history(hours: int = Query(48, ge=1, le=24 * 14)):
    h = _state["history"].iloc[-hours:]
    return {"hours": hours,
            "series": [{"ts": ts.isoformat(), "mw": float(v)} for ts, v in h.items()]}


@app.get("/forecast")
def forecast(h: int = Query(24, ge=1, le=168)):
    t0 = time.perf_counter_ns()
    hist: pd.Series = _state["history"]
    last_ts = hist.index[-1]
    key = f"{_state['version']}:h{h}:{last_ts.isoformat()}"

    cached = _state["cache"].get(key)
    if cached is not None:
        body = json.loads(cached)
        body["cache"] = "hit"
        body["latency_us"] = (time.perf_counter_ns() - t0) / 1000
        return body

    # Recursive multi-step forecast: predict t+1, append, repeat.
    # (Honest limitation to know: errors compound with h; a direct
    # per-horizon model is the standard alternative — discussed in README.)
    series = hist.copy()
    preds = []
    for _ in range(h):
        nxt = series.index[-1] + pd.Timedelta(hours=1)
        series.loc[nxt] = np.nan
        X = build_features(series, horizon=1).iloc[[-1]][_state["feats"]]
        if X.isna().any().any():
            raise HTTPException(500, "insufficient history to build features")
        yhat = float(_state["model"].predict(X)[0])
        series.iloc[-1] = yhat
        preds.append({"ts": nxt.isoformat(), "predicted_mw": round(yhat, 1)})

    body = {"horizon_hours": h, "model_version": _state["version"],
            "forecast": preds, "cache": "miss"}
    _state["cache"].set(key, json.dumps(body), ttl=3600)

    if _state["db"] is not None:
        with _state["db"].cursor() as cur:
            cur.executemany(
                "INSERT INTO predictions (target_ts, horizon_h, predicted_mw, model_version)"
                " VALUES (%s,%s,%s,%s)",
                [(p["ts"], h, p["predicted_mw"], _state["version"]) for p in preds])

    body["latency_us"] = (time.perf_counter_ns() - t0) / 1000
    return body
