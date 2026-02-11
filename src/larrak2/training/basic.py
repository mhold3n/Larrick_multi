"""Basic Training Utilities (PyTorch)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str | Path,
    epochs: int = 1000,
    lr: float = 1e-3,
    val_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train a PyTorch model with standard scaling and validation.

    Args:
        model: PyTorch model to train.
        X: Input features (numpy array).
        y: Target values (numpy array).
        output_dir: Directory to save artifacts.
        epochs: Number of training epochs.
        lr: Learning rate.
        val_size: Fraction of data for validation.
        seed: Random seed.

    Returns:
        Dictionary containing training history and paths to saved artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on input shape: {X.shape}, output shape: {y.shape}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=seed)

    # Normalize Inputs
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    # Normalize Outputs
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print("Training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Simple Full Batch
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_t)
                val_loss = criterion(y_val_pred, y_val_t)

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss.item())

            print(f"Epoch {epoch}: Train {loss.item():.6f}, Val {val_loss.item():.6f}")

            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Save Scalers
    joblib.dump(scaler_X, output_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, output_dir / "scaler_y.pkl")

    return {
        "best_val_loss": best_loss,
        "model_path": str(output_dir / "best_model.pt"),
        "scaler_X_path": str(output_dir / "scaler_X.pkl"),
        "scaler_y_path": str(output_dir / "scaler_y.pkl"),
    }
