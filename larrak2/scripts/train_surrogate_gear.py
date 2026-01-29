#!/usr/bin/env python3
"""Train Gear Surrogate Model.

Embeds strict schema hash into the artifact.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from larrak2.surrogate.features import get_gear_schema_v1
from larrak2.surrogate.models import EnsembleRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/surrogate_v1/gear/train.npz")
    parser.add_argument("--outdir", type=str, default="models/surrogate_v1")
    args = parser.parse_args()
    
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    if not data_path.exists():
        print(f"Error: {data_path} does not exist. Run gen_surrogate_data_gear.py first.")
        return
        
    with np.load(data_path) as data:
        X = data["X"]
        y = data["y"]
        
    print(f"Training on {len(X)} samples...")
    
    # Train
    # Train
    print("Initializing EnsembleRegressor...")
    base = GradientBoostingRegressor(n_estimators=100, random_state=42)
    schema = get_gear_schema_v1()
    
    model = EnsembleRegressor(
        base_estimator=base,
        n_estimators=10, 
        schema_hash=schema._hash,
        feature_names=schema.feature_names
    )
    
    model.fit(X, y)
    
    # Evaluate (Mean Prediction)
    y_pred, y_std = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"Training R2 (Mean): {r2:.4f}")
    print(f"Mean Uncertainty (Std): {np.mean(y_std):.4f}")
    
    # Save
    out_path = outdir / "model_gear.pkl"
    model.save(out_path)
    
    print(f"Saved Ensemble artifact to {out_path} (Hash: {schema._hash})")


if __name__ == "__main__":
    main()
