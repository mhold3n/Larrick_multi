"""OpenFOAM-derived neural surrogate (PyTorch).

This module provides:
- a small MLP surrogate for breathing/richness state prediction
- dataset loading utilities for training (JSON / NPZ)
- stable artifact save/load with normalization parameters

The surrogate is trained on OpenFOAM-generated (or OpenFOAM-derived) data and is
intended to be used inside the Pareto evaluation loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = e
else:  # pragma: no cover
    _TORCH_IMPORT_ERROR = None


FeatureKey = Literal[
    "rpm",
    "torque",
    "lambda_af",
    "bore_mm",
    "stroke_mm",
    "intake_port_area_m2",
    "exhaust_port_area_m2",
    "p_manifold_Pa",
    "p_back_Pa",
    "overlap_deg",
    "intake_open_deg",
    "intake_close_deg",
    "exhaust_open_deg",
    "exhaust_close_deg",
]


TargetKey = Literal[
    "m_air_trapped",
    "scavenging_efficiency",
    "residual_fraction",
    "trapped_o2_mass",
]


DEFAULT_FEATURE_KEYS: tuple[FeatureKey, ...] = (
    "rpm",
    "torque",
    "lambda_af",
    "bore_mm",
    "stroke_mm",
    "intake_port_area_m2",
    "exhaust_port_area_m2",
    "p_manifold_Pa",
    "p_back_Pa",
    "overlap_deg",
    "intake_open_deg",
    "intake_close_deg",
    "exhaust_open_deg",
    "exhaust_close_deg",
)

DEFAULT_TARGET_KEYS: tuple[TargetKey, ...] = (
    "m_air_trapped",
    "scavenging_efficiency",
    "residual_fraction",
    "trapped_o2_mass",
)


@dataclass(frozen=True)
class Normalization:
    """Z-score normalization (mean/std) for features and targets."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def normalize_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / np.where(self.x_std > 0, self.x_std, 1.0)

    def denormalize_y(self, Y: np.ndarray) -> np.ndarray:
        return Y * np.where(self.y_std > 0, self.y_std, 1.0) + self.y_mean

    @staticmethod
    def fit(X: np.ndarray, Y: np.ndarray) -> "Normalization":
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        y_mean = np.mean(Y, axis=0)
        y_std = np.std(Y, axis=0)
        return Normalization(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


@dataclass(frozen=True)
class OpenFoamSurrogateArtifact:
    feature_keys: tuple[str, ...]
    target_keys: tuple[str, ...]
    hidden_layers: tuple[int, ...]
    normalization: Normalization
    state_dict: dict[str, Any]


class OpenFoamSurrogate:
    """Runtime wrapper for OpenFOAM NN surrogate."""

    def __init__(self, artifact: OpenFoamSurrogateArtifact):
        require_torch()
        assert torch is not None  # for type checkers

        self.artifact = artifact
        self.model = MLP(
            in_dim=len(artifact.feature_keys),
            hidden=artifact.hidden_layers,
            out_dim=len(artifact.target_keys),
        )
        self.model.load_state_dict(artifact.state_dict)
        self.model.eval()

    def predict_one(self, features: dict[str, float]) -> dict[str, float]:
        """Predict targets for a single feature dict."""

        require_torch()
        assert torch is not None  # for type checkers

        x = np.array([float(features[k]) for k in self.artifact.feature_keys], dtype=np.float64)
        x_n = self.artifact.normalization.normalize_x(x[None, :])

        with torch.no_grad():
            y_n = self.model(torch.tensor(x_n, dtype=torch.float32)).cpu().numpy()

        y = self.artifact.normalization.denormalize_y(y_n)[0]
        return {k: float(v) for k, v in zip(self.artifact.target_keys, y)}


# Singleton (lazy)
_OPENFOAM_SURROGATE: OpenFoamSurrogate | None = None


def get_openfoam_surrogate(path: str | Path) -> OpenFoamSurrogate:
    """Load and cache an OpenFOAM surrogate from `path`."""

    global _OPENFOAM_SURROGATE
    if _OPENFOAM_SURROGATE is None:
        artifact = load_artifact(path)
        _OPENFOAM_SURROGATE = OpenFoamSurrogate(artifact)
    return _OPENFOAM_SURROGATE


def require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the OpenFOAM NN surrogate. "
            "Install with the optional extra (recommended): `pip install -e \".[openfoam_nn]\"`."
        ) from _TORCH_IMPORT_ERROR


def load_dataset_json(
    path: str | Path,
    *,
    feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS,
    target_keys: tuple[str, ...] = DEFAULT_TARGET_KEYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset from a JSON list of dict records."""

    records = json.loads(Path(path).read_text())
    if not isinstance(records, list):
        raise ValueError("JSON dataset must be a list of records")

    X = np.zeros((len(records), len(feature_keys)), dtype=np.float64)
    Y = np.zeros((len(records), len(target_keys)), dtype=np.float64)

    for i, r in enumerate(records):
        for j, k in enumerate(feature_keys):
            if k not in r:
                raise KeyError(f"Missing feature '{k}' in record {i}")
            X[i, j] = float(r[k])
        for j, k in enumerate(target_keys):
            if k not in r:
                raise KeyError(f"Missing target '{k}' in record {i}")
            Y[i, j] = float(r[k])

    return X, Y


def load_dataset_jsonl(
    path: str | Path,
    *,
    feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS,
    target_keys: tuple[str, ...] = DEFAULT_TARGET_KEYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset from JSONL (one record per line)."""

    xs: list[list[float]] = []
    ys: list[list[float]] = []
    p = Path(path)
    for i, line in enumerate(p.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        row_x: list[float] = []
        row_y: list[float] = []
        for k in feature_keys:
            if k not in r:
                raise KeyError(f"Missing feature '{k}' in record {i}")
            row_x.append(float(r[k]))
        for k in target_keys:
            if k not in r:
                raise KeyError(f"Missing target '{k}' in record {i}")
            row_y.append(float(r[k]))
        xs.append(row_x)
        ys.append(row_y)

    X = np.asarray(xs, dtype=np.float64)
    Y = np.asarray(ys, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("JSONL dataset must produce 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X/Y row mismatch: {X.shape[0]} vs {Y.shape[0]}")
    return X, Y


def load_dataset_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset from an NPZ with arrays X and Y."""

    with np.load(Path(path)) as data:
        if "X" not in data.files or "Y" not in data.files:
            raise ValueError("NPZ dataset must contain arrays 'X' and 'Y'")
        X = np.asarray(data["X"], dtype=np.float64)
        Y = np.asarray(data["Y"], dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X/Y row mismatch: {X.shape[0]} vs {Y.shape[0]}")
    return X, Y


def train_openfoam_surrogate(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS,
    target_keys: tuple[str, ...] = DEFAULT_TARGET_KEYS,
    hidden_layers: tuple[int, ...] = (64, 64),
    seed: int = 42,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_frac: float = 0.2,
) -> OpenFoamSurrogateArtifact:
    """Train an MLP surrogate and return a serializable artifact."""

    require_torch()
    assert torch is not None and nn is not None  # for type checkers

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to train a neural surrogate")

    # Shuffle and split
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    norm = Normalization.fit(X_tr, Y_tr)
    X_tr_n = norm.normalize_x(X_tr)
    X_val_n = norm.normalize_x(X_val)
    Y_tr_n = (Y_tr - norm.y_mean) / np.where(norm.y_std > 0, norm.y_std, 1.0)
    Y_val_n = (Y_val - norm.y_mean) / np.where(norm.y_std > 0, norm.y_std, 1.0)

    # Torch tensors
    X_tr_t = torch.tensor(X_tr_n, dtype=torch.float32)
    Y_tr_t = torch.tensor(Y_tr_n, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val_n, dtype=torch.float32)

    model = MLP(in_dim=X.shape[1], hidden=hidden_layers, out_dim=Y.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    patience = 50
    patience_left = patience

    for _ in range(epochs):
        model.train()
        optim.zero_grad()
        pred = model(X_tr_t)
        loss = loss_fn(pred, Y_tr_t)
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(loss_fn(val_pred, Y_val_t).item())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = model.state_dict()

    return OpenFoamSurrogateArtifact(
        feature_keys=tuple(feature_keys),
        target_keys=tuple(target_keys),
        hidden_layers=tuple(hidden_layers),
        normalization=norm,
        state_dict=best_state,
    )


def save_artifact(artifact: OpenFoamSurrogateArtifact, path: str | Path) -> None:
    """Save artifact to disk as a torch checkpoint."""

    require_torch()
    assert torch is not None  # for type checkers

    payload = {
        "feature_keys": list(artifact.feature_keys),
        "target_keys": list(artifact.target_keys),
        "hidden_layers": list(artifact.hidden_layers),
        "normalization": {
            "x_mean": artifact.normalization.x_mean.tolist(),
            "x_std": artifact.normalization.x_std.tolist(),
            "y_mean": artifact.normalization.y_mean.tolist(),
            "y_std": artifact.normalization.y_std.tolist(),
        },
        "state_dict": artifact.state_dict,
    }
    torch.save(payload, str(path))


def load_artifact(path: str | Path) -> OpenFoamSurrogateArtifact:
    """Load artifact from disk."""

    require_torch()
    assert torch is not None  # for type checkers

    payload = torch.load(str(path), map_location="cpu")
    norm = Normalization(
        x_mean=np.asarray(payload["normalization"]["x_mean"], dtype=np.float64),
        x_std=np.asarray(payload["normalization"]["x_std"], dtype=np.float64),
        y_mean=np.asarray(payload["normalization"]["y_mean"], dtype=np.float64),
        y_std=np.asarray(payload["normalization"]["y_std"], dtype=np.float64),
    )
    return OpenFoamSurrogateArtifact(
        feature_keys=tuple(payload["feature_keys"]),
        target_keys=tuple(payload["target_keys"]),
        hidden_layers=tuple(int(x) for x in payload["hidden_layers"]),
        normalization=norm,
        state_dict=payload["state_dict"],
    )

