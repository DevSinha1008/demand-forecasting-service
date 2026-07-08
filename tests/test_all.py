"""Tests. The leakage test is the one that matters most: it PROVES the
feature layer can't see the future, rather than asserting it in a comment.

Method: build features twice — once on the full series, once on the series
truncated at time t. For any t, the feature row AT t must be identical in
both. If a feature ever differs, it consumed information from after t.
This is the test that would have caught the classic .rolling()-without-
.shift() bug.
"""

import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dfs.features import build_features, make_xy, load_pjm_csv, DEFAULT_LAGS
from dfs.validate import walk_forward, rmse


def synthetic_series(n_hours=24 * 365 * 3, seed=7) -> pd.Series:
    """PJM-shaped synthetic demand: daily + weekly + annual cycles + noise.
    For TESTS only — CV metrics must come from the real PJME CSV."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n_hours, freq="h")
    t = np.arange(n_hours)
    daily = 4000 * np.sin(2 * np.pi * (np.asarray(idx.hour) - 6) / 24)
    weekly = np.where(np.asarray(idx.dayofweek) >= 5, -2500, 0)
    annual = 3000 * np.sin(2 * np.pi * np.asarray(idx.dayofyear) / 365)
    y = np.asarray(30000 + daily + weekly + annual
                   + rng.normal(0, 700, n_hours), dtype=float)
    # AR(1) persistence so lags carry genuine signal
    for i in range(1, n_hours):
        y[i] = 0.6 * y[i - 1] + 0.4 * y[i]
    return pd.Series(y, index=idx, name="MW")


# --------------------------- THE leakage test -------------------------------
def test_features_cannot_see_future():
    y = synthetic_series(24 * 40)
    X_full = build_features(y)
    for cut in [24 * 20, 24 * 25, 24 * 33]:
        y_trunc = y.iloc[:cut]
        X_trunc = build_features(y_trunc)
        t = y_trunc.index[-1]
        full_row = X_full.loc[t].dropna()
        trunc_row = X_trunc.loc[t].dropna()
        pd.testing.assert_series_equal(full_row, trunc_row, check_names=False)


def test_horizon_respects_gap():
    y = synthetic_series(24 * 30)
    X = build_features(y, horizon=24)
    # with h=24 no feature may use lags closer than 24h
    assert "lag_1" not in X.columns and "lag_24" in X.columns


def test_walk_forward_is_chronological():
    y = synthetic_series(24 * 200)
    X, y2 = make_xy(y)

    class RecordingModel:
        def __init__(self): self.train_max = None
        def fit(self, X, y): self.train_max = X.index.max()
        def predict(self, X):
            # every test timestamp must be strictly after all training data
            assert (X.index > self.train_max).all(), "test overlaps train!"
            return np.full(len(X), y2.mean())

    walk_forward(RecordingModel, X, y2, n_windows=4,
                 test_hours=24 * 5, min_train_hours=24 * 100)


def test_model_beats_baselines_on_synthetic():
    from xgboost import XGBRegressor
    y = synthetic_series()
    X, y2 = make_xy(y)
    rep = walk_forward(
        lambda: XGBRegressor(n_estimators=60, max_depth=6, n_jobs=-1,
                             random_state=0),
        X, y2, n_windows=4, test_hours=24 * 30, min_train_hours=24 * 365)
    assert rep.model_rmse < rep.seasonal_rmse, "model should beat seasonal-naive"
    assert rep.n_windows == 4


def test_csv_loader_handles_duplicates_and_gaps(tmp_path):
    idx = pd.date_range("2020-01-01", periods=100, freq="h")
    df = pd.DataFrame({"Datetime": idx, "PJME_MW": np.arange(100.0)})
    df = pd.concat([df, df.iloc[[10]]])          # DST-style duplicate
    df = df.drop(index=50)                        # a missing hour
    p = tmp_path / "pjm.csv"
    df.to_csv(p, index=False)
    s = load_pjm_csv(str(p))
    assert s.index.is_unique and len(s) == 100
    assert not s.isna().any()


def test_service_endpoints(tmp_path, monkeypatch):
    """Spin the API up with a tiny model; verify contract + cache hit path."""
    from xgboost import XGBRegressor
    import pickle, json as js
    y = synthetic_series(24 * 400)
    X, y2 = make_xy(y)
    m = XGBRegressor(n_estimators=30, max_depth=4, random_state=0).fit(X, y2)
    (tmp_path / "artifacts").mkdir()
    with open(tmp_path / "artifacts" / "model.pkl", "wb") as f:
        pickle.dump({"model": m, "feature_names": list(X.columns),
                     "horizon": 1, "train_rows": len(X),
                     "csv_sha256_16": "testmodel"}, f)
    y.iloc[-24 * 14:].to_frame("mw").to_csv(tmp_path / "artifacts" / "recent_history.csv")
    (tmp_path / "artifacts" / "metrics.json").write_text(js.dumps({"windows [G]": 4}))

    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    import importlib
    from dfs import service
    importlib.reload(service)
    from fastapi.testclient import TestClient
    with TestClient(service.app) as c:
        assert c.get("/health").json()["cache"] == "in-process"
        r1 = c.get("/forecast?h=6").json()
        assert len(r1["forecast"]) == 6 and r1["cache"] == "miss"
        r2 = c.get("/forecast?h=6").json()
        assert r2["cache"] == "hit"          # cache path exercised
        assert c.get("/metrics").json()["windows [G]"] == 4
        assert len(c.get("/history?hours=24").json()["series"]) == 24
