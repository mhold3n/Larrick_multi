#!/usr/bin/env python3
"""Train residual surrogate models.

Surrogates predict:
- Delta Efficiency (Target - Baseline)
- Delta Loss (Target - Baseline)

Method: Gradient Boosting Regressor (sklearn)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


def train_models(X: np.ndarray, y: np.ndarray) -> tuple[BaseEstimator, dict]:
    """Train regressors.
    
    Args:
        X: Features (N x D)
        y: Targets (N x 2) -> [delta_eff, delta_loss]
        
    Returns:
        (model, metrics)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use MultiOutputRegressor wrapping GradientBoosting
    # Or separate models. MultiOutput is convenient.
    
    # GBR is good for non-linear residuals
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model = MultiOutputRegressor(est)
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    
    metrics = {
        "mse_eff": mse[0],
        "mse_loss": mse[1],
        "r2_eff": r2[0],
        "r2_loss": r2[1]
    }
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Surrogate Models")
    parser.add_argument("--data", type=str, default="data/surrogate_v1/training_data.npz", help="Input data")
    parser.add_argument("--outdir", type=str, default="models/surrogate_v1", help="Output directory")
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    with np.load(data_path) as data:
        X = data["X"]
        y_resid = data["y_resid"]
        
    print(f"Training on {len(X)} samples...")
    model, metrics = train_models(X, y_resid)
    
    print("Metrics (Test Set):")
    print(f"  Efficiency Residual R2: {metrics['r2_eff']:.4f}, MSE: {metrics['mse_eff']:.2e}")
    print(f"  Loss Residual R2:       {metrics['r2_loss']:.4f}, MSE: {metrics['mse_loss']:.2e}")
    
    # Save
    model_path = outdir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
