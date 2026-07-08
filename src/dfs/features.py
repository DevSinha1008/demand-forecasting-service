"""features.py — Feature engineering for hourly demand forecasting.

The leakage discipline (the thing this project exists to demonstrate):
every feature for predicting hour t may use ONLY information available
strictly before t. Concretely:

  * lag features shift the series by >= 1 hour (lag_1 = value at t-1);
  * rolling statistics are computed over a window ENDING at t-1, which is
    why we .shift(1) BEFORE .rolling() — rolling then shifting would still
    be fine, but rolling WITHOUT shifting would include y_t itself in its
    own feature: textbook leakage, and the single most common bug in
    time-series ML;
  * calendar features (hour, day-of-week, month, …) are known in advance,
    so they're legitimately usable for any horizon.

The horizon parameter generalises this: to forecast t+h using data through
t, all lags shift by at least h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lags chosen for physical meaning, not kitchen-sink:
#   1,2,3   — immediate persistence (demand is strongly autocorrelated)
#   24      — same hour yesterday (daily cycle)
#   168     — same hour last week (weekly cycle)
DEFAULT_LAGS = (1, 2, 3, 24, 168)
DEFAULT_ROLLS = (24, 168)  # rolling mean/std over last day, last week


def load_pjm_csv(path: str, value_col: str | None = None) -> pd.Series:
    """Load a PJM hourly CSV (Datetime + one MW column) into a clean Series.

    Handles the real dataset's quirks: DST duplicates (kept as mean),
    missing hours (reindexed and interpolated — only ~a handful in PJME).
    """
    df = pd.read_csv(path)
    dt_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    if value_col is None:
        value_col = [c for c in df.columns if c != dt_col][0]
    df[dt_col] = pd.to_datetime(df[dt_col])
    s = (df.set_index(dt_col)[value_col]
           .sort_index()
           .groupby(level=0).mean())           # collapse DST duplicates
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    n_missing = len(full_idx) - len(s)
    s = s.reindex(full_idx).interpolate(limit=3)
    s.name = value_col
    s.attrs["missing_hours_filled"] = int(n_missing)
    return s


def build_features(y: pd.Series,
                   lags: tuple[int, ...] = DEFAULT_LAGS,
                   rolls: tuple[int, ...] = DEFAULT_ROLLS,
                   horizon: int = 1) -> pd.DataFrame:
    """Return X aligned with target y_t; all features use info from <= t-horizon."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1 (h=0 would leak the target)")
    X = pd.DataFrame(index=y.index)

    # --- lagged values ---
    for lag in lags:
        eff = max(lag, horizon)          # never look closer than the horizon
        X[f"lag_{eff}"] = y.shift(eff)

    # --- rolling stats over a window ending at t-horizon ---
    base = y.shift(horizon)              # shift FIRST: excludes y_t (and closer)
    for w in rolls:
        X[f"roll_mean_{w}"] = base.rolling(w, min_periods=w).mean()
        X[f"roll_std_{w}"] = base.rolling(w, min_periods=w).std()

    # --- calendar/seasonal encodings (known in advance, no leakage) ---
    idx = X.index
    X["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    X["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    X["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    X["is_weekend"] = (idx.dayofweek >= 5).astype(np.int8)

    # de-duplicate columns (e.g. horizon collapsing two lags onto one name)
    X = X.loc[:, ~X.columns.duplicated()]
    return X


def make_xy(y: pd.Series, horizon: int = 1, **kw) -> tuple[pd.DataFrame, pd.Series]:
    """Features + target with warmup NaNs dropped, indices aligned."""
    X = build_features(y, horizon=horizon, **kw)
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]
