#!/usr/bin/env python3
"""Train Scavenge Surrogate Model (V1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from larrak2.surrogate.features import get_scavenge_schema_v1
from larrak2.surrogate.models import EnsembleRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/surrogate_v1/scavenge/train.npz")
    parser.add_argument("--outdir", type=str, default="models/surrogate_v1")
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    if not data_path.exists():
        print(f"Error: {data_path} does not exist.")
        return

    with np.load(data_path) as data:
        X = data["X"]
        y = data["y"]

    print(f"Training on {len(X)} samples...")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()

    schema = get_scavenge_schema_v1()
    base_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    ensemble = EnsembleRegressor(
        base_estimator=base_model,
        n_estimators=10,
        schema_hash=schema._hash,
        feature_names=schema.feature_names,
    )
    ensemble.fit(X, y)

    y_pred, y_std = ensemble.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"Training R2: {r2:.4f}")

    out_path = outdir / "model_scavenge.pkl"
    ensemble.save(out_path)

    print(f"Saved ensemble to {out_path} (Hash: {schema._hash})")


if __name__ == "__main__":
    main()
