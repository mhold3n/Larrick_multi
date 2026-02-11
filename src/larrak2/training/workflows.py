"""Training Workflows for various surrogate models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# OpenFOAM Imports
try:
    from larrak2.surrogate.openfoam_nn import (
        DEFAULT_FEATURE_KEYS,
        DEFAULT_TARGET_KEYS,
        load_dataset_json,
        load_dataset_npz,
        save_artifact,
        train_openfoam_surrogate,
    )
except ImportError:
    pass

# Gear NN Imports
try:
    from larrak2.surrogate.gear_loss_net import GearLossNetwork
    from larrak2.training.basic import train_model as train_proch_basic
except ImportError:
    pass

# Sklearn Imports
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputRegressor

    from larrak2.surrogate.features import get_gear_schema_v1, get_scavenge_schema_v1
    from larrak2.surrogate.models import EnsembleRegressor
except ImportError:
    pass


def train_openfoam_workflow(args: argparse.Namespace):
    """Workflow for OpenFOAM NN training."""
    print("Starting OpenFOAM NN Training Workflow...")
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    suf = data_path.suffix.lower()
    if suf == ".json":
        X, Y = load_dataset_json(
            data_path, feature_keys=DEFAULT_FEATURE_KEYS, target_keys=DEFAULT_TARGET_KEYS
        )
    elif suf == ".npz":
        X, Y = load_dataset_npz(data_path)
    elif suf == ".parquet":
        df = pd.read_parquet(data_path)
        X = df[list(DEFAULT_FEATURE_KEYS)].to_numpy(dtype=np.float64)
        Y = df[list(DEFAULT_TARGET_KEYS)].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported format: {suf}")

    # 2. Train
    hidden_layers = tuple(int(s) for s in args.hidden.split(",")) if args.hidden else (64, 64)

    artifact = train_openfoam_surrogate(
        X,
        Y,
        feature_keys=DEFAULT_FEATURE_KEYS,
        target_keys=DEFAULT_TARGET_KEYS,
        hidden_layers=hidden_layers,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=0.2,
    )

    # 3. Save
    out_path = outdir / args.name
    save_artifact(artifact, out_path)
    print(f"Saved to {out_path}")


def train_gear_nn_workflow(args: argparse.Namespace):
    """Workflow for Gear Loss NN training."""
    print("Starting Gear NN Training Workflow...")
    df = pd.read_parquet(args.data)

    # Columns logic (from Phase 2)
    if "c1" not in df.columns and "req_amp" in df.columns:
        print("Deriving pitch coefficients from req_amp...")
        df["c0"] = 0.0
        df["c1"] = df["req_amp"] * df["base_radius"]
        df["c2"] = 0.0
        df["c3"] = 0.0
        df["c4"] = 0.0

    X_cols = ["rpm", "torque", "base_radius", "c0", "c1", "c2", "c3", "c4"]
    rename_map = {"dual_W_mesh": "W_mesh", "dual_W_bearing": "W_bearing"}
    df.rename(columns=rename_map, inplace=True)

    available_y = [c for c in ["W_mesh", "W_bearing", "W_churning"] if c in df.columns]
    if not available_y:
        raise ValueError("No target columns found (W_mesh, W_bearing, W_churning)")

    X = df[X_cols].values.astype(np.float32)
    y = df[available_y].values.astype(np.float32)

    model = GearLossNetwork(
        input_dim=len(X_cols),
        hidden_dim=int(args.hidden) if args.hidden and "," not in args.hidden else 64,
        output_dim=len(available_y),
    )

    train_proch_basic(model=model, X=X, y=y, output_dir=args.outdir, epochs=args.epochs, lr=args.lr)
    print("Gear NN training complete.")


def train_scavenge_gbr_workflow(args: argparse.Namespace):
    """Workflow for Scavenge GBR training."""
    print("Starting Scavenge GBR Training Workflow...")
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(data_path) as data:
        X = data["X"]
        y = data["y"]

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
    print(f"Saved to {out_path}")


def train_gear_gbr_workflow(args: argparse.Namespace):
    """Workflow for Gear GBR training."""
    print("Starting Gear GBR Training Workflow...")
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(data_path) as data:
        X = data["X"]
        y = data["y"]

    base_est = GradientBoostingRegressor(n_estimators=100, random_state=42)
    # Wrap in MultiOutputRegressor to handle [mesh, bearing, churning]
    if y.ndim > 1 and y.shape[1] > 1:
        base = MultiOutputRegressor(base_est)
    else:
        base = base_est

    schema = get_gear_schema_v1()

    model = EnsembleRegressor(
        base_estimator=base,
        n_estimators=10,
        schema_hash=schema._hash,
        feature_names=schema.feature_names,
    )

    model.fit(X, y)

    y_pred, y_std = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"Training R2 (Mean): {r2:.4f}")

    out_path = outdir / "model_gear.pkl"
    model.save(out_path)
    print(f"Saved to {out_path}")


def train_residual_workflow(args: argparse.Namespace):
    """Workflow for Residual Surrogate training."""
    print("Starting Residual Surrogate Training Workflow...")
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(data_path) as data:
        X = data["X"]
        y = data["y_resid"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    est = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model = MultiOutputRegressor(est)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")

    print(f"Efficiency Residual R2: {r2[0]:.4f}")
    print(f"Loss Residual R2:       {r2[1]:.4f}")

    import pickle

    with open(outdir / "model_residual.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved to {outdir / 'model_residual.pkl'}")
