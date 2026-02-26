"""CLI and utilities for training HiFi surrogate ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from larrak2.surrogate.hifi.models import (
    FlowCoefficientSurrogate,
    StructuralSurrogate,
    ThermalSurrogate,
)
from larrak2.training.hifi_schema import TrainingDataset


def train_ensemble(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = True,
) -> dict[str, float]:
    """Train each ensemble member on bootstrap data."""
    criterion = nn.MSELoss()

    for idx, member in enumerate(model.models):
        n = len(X_train)
        if n == 0:
            break

        boot_idx = np.random.choice(n, n, replace=True)
        X_boot = torch.tensor(X_train[boot_idx], dtype=torch.float32)
        y_boot = torch.tensor(y_train[boot_idx], dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_boot, y_boot), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(member.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best = float("inf")
        patience = 0

        for _epoch in range(epochs):
            member.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = member(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            member.eval()
            with torch.no_grad():
                val_loss = criterion(
                    member(torch.tensor(X_val, dtype=torch.float32)),
                    torch.tensor(y_val, dtype=torch.float32),
                ).item()
            scheduler.step(val_loss)

            if val_loss < best:
                best = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 20:
                    break

        if verbose:
            print(f"  Member {idx + 1}/{len(model.models)} val_loss={best:.6f}")

    model.eval()
    with torch.no_grad():
        mean, std = model(torch.tensor(X_val, dtype=torch.float32))
        final_loss = criterion(mean, torch.tensor(y_val, dtype=torch.float32)).item()

    return {"final_val_loss": float(final_loss), "mean_uncertainty": float(std.mean().item())}


def train_all_surrogates(
    data_path: str,
    output_dir: str,
    *,
    epochs: int = 100,
    n_models: int = 5,
) -> dict[str, dict[str, float]]:
    """Train thermal/structural/flow surrogate ensembles."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = (
        TrainingDataset.from_parquet(data_path)
        if str(data_path).endswith(".parquet")
        else TrainingDataset.from_json(data_path)
    )

    train_ds, val_ds = dataset.split(train_frac=0.8)
    results: dict[str, dict[str, float]] = {}

    tasks = [
        (
            "thermal",
            ThermalSurrogate(n_models=n_models),
            train_ds.get_thermal_data,
            val_ds.get_thermal_data,
            out / "thermal_surrogate.pt",
        ),
        (
            "structural",
            StructuralSurrogate(n_models=n_models),
            train_ds.get_structural_data,
            val_ds.get_structural_data,
            out / "structural_surrogate.pt",
        ),
        (
            "flow",
            FlowCoefficientSurrogate(n_models=n_models),
            train_ds.get_flow_data,
            val_ds.get_flow_data,
            out / "flow_surrogate.pt",
        ),
    ]

    for name, model, train_fn, val_fn, target in tasks:
        X_train, y_train = train_fn()
        X_val, y_val = val_fn()

        if len(X_train) == 0 or len(X_val) == 0:
            results[name] = {"final_val_loss": float("nan"), "mean_uncertainty": float("nan")}
            continue

        hist = train_ensemble(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            verbose=True,
        )
        model.save(str(target))
        results[name] = hist

    dataset.norm_params.save(out / "normalization.json")
    (out / "training_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train HiFi surrogate ensembles")
    parser.add_argument("--data", required=True, help="Path to DOE results (.json/.parquet)")
    parser.add_argument("--output", default="models/hifi", help="Output model directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--n-models", type=int, default=5, help="Ensemble members")
    args = parser.parse_args(argv)

    results = train_all_surrogates(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        n_models=args.n_models,
    )

    print("Training complete")
    for name, metrics in results.items():
        print(f"{name}: val_loss={metrics.get('final_val_loss')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
