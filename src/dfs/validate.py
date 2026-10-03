"""validate.py — Walk-forward validation against naive baselines.

Why not random K-fold: shuffling hourly data trains on Tuesday 3pm to
predict Tuesday 2pm — the model sees the future. Scores look great and
mean nothing (lookahead bias). Time series must be split chronologically.

Scheme: EXPANDING-WINDOW walk-forward.

    |-------- train --------|-- test --|
    |---------- train ----------|-- test --|
    |------------- train -----------|-- test --|

Each window trains on everything before the test block, predicts the
block, then the block joins the training set for the next window. This
mirrors deployment: you always predict the future from the past, and you
retrain as data arrives. 

Baselines:
  * persistence   — predict y_{t-h}: "demand now = demand an hour ago".
                    Brutally strong at h=1 for autocorrelated series.
  * seasonal-naive — predict y_{t-168}: same hour last week. The honest
                    yardstick for seasonal data and the one [E]% is
                    measured against (beating persistence at h=1 is hard;
                    beating seasonal-naive is the meaningful claim).
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
    
@dataclass
class WindowResult:
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    n_test: int
    model_rmse: float
    model_mae: float
    persistence_rmse: float
    seasonal_rmse: float

@dataclass
class WalkForwardReport:
    windows: list[WindowResult] = field(default_factory=list)

    # pooled = metric over all test predictions concatenated (not mean of
    # window metrics, which would overweight small windows)
    model_rmse: float = 0.0
    model_mae: float = 0.0
    persistence_rmse: float = 0.0
    seasonal_rmse: float = 0.0

    @property
    def n_windows(self) -> int:                    
        return len(self.windows)

    @property
    def rmse_improvement_vs_seasonal(self) -> float: 
        return 100.0 * (self.seasonal_rmse - self.model_rmse) / self.seasonal_rmse

    @property
    def rmse_improvement_vs_persistence(self) -> float:
        return 100.0 * (self.persistence_rmse - self.model_rmse) / self.persistence_rmse

    def summary(self) -> dict:
        return {
            "windows [G]": self.n_windows,
            "model_rmse_mw": round(self.model_rmse, 1),
            "model_mae_mw [F]": round(self.model_mae, 1),
            "seasonal_naive_rmse_mw": round(self.seasonal_rmse, 1),
            "persistence_rmse_mw": round(self.persistence_rmse, 1),
            "rmse_improvement_vs_seasonal_pct [E]":
                round(self.rmse_improvement_vs_seasonal, 1),
            "rmse_improvement_vs_persistence_pct":
                round(self.rmse_improvement_vs_persistence, 1),
        }


def walk_forward(model_factory,
                 X: pd.DataFrame,
                 y: pd.Series,
                 horizon: int = 1,
                 n_windows: int = 8,
                 test_hours: int = 24 * 90,
                 min_train_hours: int = 24 * 365 * 2) -> WalkForwardReport:
    """Expanding-window walk-forward evaluation.

    model_factory: zero-arg callable returning a FRESH unfitted model per
    window (reusing a fitted model across windows would leak state).
    """
    n = len(y)
    need = min_train_hours + n_windows * test_hours
    if n < need:
        # shrink test blocks to fit small datasets (keeps tests fast);
        # real PJME easily satisfies the default sizes
        test_hours = max(24, (n - min_train_hours) // n_windows)
        if test_hours < 24:
            raise ValueError(f"dataset too small: {n} rows")

    rep = WalkForwardReport()
    ytr_all, ypr_all, pers_all, seas_all = [], [], [], []

    for w in range(n_windows):
        test_start = n - (n_windows - w) * test_hours
        test_end = test_start + test_hours
        X_tr, y_tr = X.iloc[:test_start], y.iloc[:test_start]
        X_te, y_te = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

        model = model_factory()
        model.fit(X_tr, y_tr)
        pred = np.asarray(model.predict(X_te), dtype=float)

        # baselines from the ORIGINAL series, aligned to test timestamps
        pers = y.shift(horizon).iloc[test_start:test_end].to_numpy()
        seas = y.shift(168).iloc[test_start:test_end].to_numpy()

        yt = y_te.to_numpy()
        rep.windows.append(WindowResult(
            train_end=y.index[test_start - 1], test_end=y.index[test_end - 1],
            n_test=len(yt),
            model_rmse=rmse(yt, pred), model_mae=mae(yt, pred),
            persistence_rmse=rmse(yt, pers), seasonal_rmse=rmse(yt, seas)))

        ytr_all.append(yt); ypr_all.append(pred)
        pers_all.append(pers); seas_all.append(seas)

    yt = np.concatenate(ytr_all); yp = np.concatenate(ypr_all)
    rep.model_rmse = rmse(yt, yp)
    rep.model_mae = mae(yt, yp)
    rep.persistence_rmse = rmse(yt, np.concatenate(pers_all))
    rep.seasonal_rmse = rmse(yt, np.concatenate(seas_all))
    return rep
