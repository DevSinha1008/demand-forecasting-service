"""train.py — Reproducible training run.

Usage:
    python -m dfs.train --csv data/PJME_hourly.csv --out artifacts/

Steps: load -> features -> walk-forward evaluation (produces [E]/[F]/[G])
-> fit final model on all data -> save model + metrics.json + a features
manifest, so a served prediction is traceable to the exact training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import sys

from xgboost import XGBRegressor

from .features import make_xy, load_pjm_csv
from .validate import walk_forward


def model_factory():
    # Deliberately unglamorous model; sophistication lives in the
    # validation methodology, not the estimator. Fixed seed => reproducible.
    return XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", random_state=42, n_jobs=-1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="PJM hourly CSV (e.g. PJME_hourly.csv)")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--windows", type=int, default=8)
    a = ap.parse_args(argv)

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    y = load_pjm_csv(a.csv)
    X, y_al = make_xy(y, horizon=a.horizon)
    print(f"loaded {len(y):,} hours ({y.index.min()} .. {y.index.max()}), "
          f"{X.shape[1]} features, {len(X):,} usable rows", file=sys.stderr)

    rep = walk_forward(model_factory, X, y_al,
                       horizon=a.horizon, n_windows=a.windows)
    metrics = rep.summary()
    print(json.dumps(metrics, indent=2))

    final = model_factory()
    final.fit(X, y_al)

    csv_hash = hashlib.sha256(pathlib.Path(a.csv).read_bytes()).hexdigest()[:16]
    with open(out / "model.pkl", "wb") as f:
        pickle.dump({"model": final,
                     "feature_names": list(X.columns),
                     "horizon": a.horizon,
                     "train_rows": len(X),
                     "csv_sha256_16": csv_hash}, f)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # last 168h of the series: the service needs recent history to build lags
    y.iloc[-24 * 14:].to_frame("mw").to_csv(out / "recent_history.csv")
    print(f"artifacts written to {out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
