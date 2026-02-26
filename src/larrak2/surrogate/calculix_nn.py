"""CalculiX-derived neural surrogate (PyTorch).

This module provides:
- a compact MLP surrogate for stress prediction
- dataset loading utilities for training (JSON / JSONL / NPZ)
- stable artifact save/load with normalization parameters
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
    "base_radius_mm",
    "face_width_mm",
    "module_mm",
    "pressure_angle_deg",
    "helix_angle_deg",
    "profile_shift",
]

TargetKey = Literal["max_stress"]


DEFAULT_FEATURE_KEYS: tuple[FeatureKey, ...] = (
    "rpm",
    "torque",
    "base_radius_mm",
    "face_width_mm",
    "module_mm",
    "pressure_angle_deg",
    "helix_angle_deg",
    "profile_shift",
)

DEFAULT_TARGET_KEYS: tuple[TargetKey, ...] = ("max_stress",)


@dataclass(frozen=True)
class Normalization:
    """Z-score normalization (mean/std) for features and targets."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def normalize_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / np.where(self.x_std > 0, self.x_std, 1.0)

    def normalize_y(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.y_mean) / np.where(self.y_std > 0, self.y_std, 1.0)

    def denormalize_y(self, Y: np.ndarray) -> np.ndarray:
        return Y * np.where(self.y_std > 0, self.y_std, 1.0) + self.y_mean

    @staticmethod
    def fit(X: np.ndarray, Y: np.ndarray) -> Normalization:
        return Normalization(
            x_mean=np.mean(X, axis=0),
            x_std=np.std(X, axis=0),
            y_mean=np.mean(Y, axis=0),
            y_std=np.std(Y, axis=0),
        )


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
class CalculixSurrogateArtifact:
    feature_keys: tuple[str, ...]
    target_keys: tuple[str, ...]
    hidden_layers: tuple[int, ...]
    normalization: Normalization
    state_dict: dict[str, Any]


class CalculixSurrogate:
    """Runtime wrapper for CalculiX NN surrogate."""

    def __init__(self, artifact: CalculixSurrogateArtifact):
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


_CALCULIX_SURROGATE: CalculixSurrogate | None = None


def get_calculix_surrogate(path: str | Path) -> CalculixSurrogate:
    """Load and cache a CalculiX surrogate from `path`."""
    global _CALCULIX_SURROGATE
    if _CALCULIX_SURROGATE is None:
        artifact = load_artifact(path)
        _CALCULIX_SURROGATE = CalculixSurrogate(artifact)
    return _CALCULIX_SURROGATE


def require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the CalculiX NN surrogate. "
            'Install with the optional extra: `pip install -e ".[dev]"`.'
        ) from _TORCH_IMPORT_ERROR


def load_dataset_json(
    path: str | Path,
    *,
    feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS,
    target_keys: tuple[str, ...] = DEFAULT_TARGET_KEYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset from a JSON list of records."""
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list):
        raise ValueError("JSON dataset must be a list of records")

    X = np.zeros((len(records), len(feature_keys)), dtype=np.float64)
    Y = np.zeros((len(records), len(target_keys)), dtype=np.float64)
    for i, rec in enumerate(records):
        for j, k in enumerate(feature_keys):
            if k not in rec:
                raise KeyError(f"Missing feature '{k}' in record {i}")
            X[i, j] = float(rec[k])
        for j, k in enumerate(target_keys):
            if k not in rec:
                raise KeyError(f"Missing target '{k}' in record {i}")
            Y[i, j] = float(rec[k])
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
    for i, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        row_x: list[float] = []
        row_y: list[float] = []
        for k in feature_keys:
            if k not in rec:
                raise KeyError(f"Missing feature '{k}' in record {i}")
            row_x.append(float(rec[k]))
        for k in target_keys:
            if k not in rec:
                raise KeyError(f"Missing target '{k}' in record {i}")
            row_y.append(float(rec[k]))
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
    """Load dataset from NPZ with arrays X and Y."""
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


def train_calculix_surrogate(
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
) -> CalculixSurrogateArtifact:
    """Train an MLP surrogate and return a serializable artifact."""
    require_torch()
    assert torch is not None and nn is not None  # for type checkers

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X/Y row mismatch: {X.shape[0]} vs {Y.shape[0]}")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples to train a neural surrogate")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    X = X[idx]
    Y = Y[idx]

    n_val = max(1, int(round(X.shape[0] * val_frac)))
    n_val = min(n_val, X.shape[0] - 1)
    X_train, X_val = X[:-n_val], X[-n_val:]
    Y_train, Y_val = Y[:-n_val], Y[-n_val:]

    norm = Normalization.fit(X_train, Y_train)
    Xn_train = norm.normalize_x(X_train)
    Yn_train = norm.normalize_y(Y_train)
    Xn_val = norm.normalize_x(X_val)
    Yn_val = norm.normalize_y(Y_val)

    torch.manual_seed(seed)
    model = MLP(in_dim=X.shape[1], hidden=hidden_layers, out_dim=Y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    xt = torch.tensor(Xn_train, dtype=torch.float32)
    yt = torch.tensor(Yn_train, dtype=torch.float32)
    xv = torch.tensor(Xn_val, dtype=torch.float32)
    yv = torch.tensor(Yn_val, dtype=torch.float32)

    for _ in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad()
        pred = model(xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _ = loss_fn(model(xv), yv)

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return CalculixSurrogateArtifact(
        feature_keys=tuple(feature_keys),
        target_keys=tuple(target_keys),
        hidden_layers=tuple(int(h) for h in hidden_layers),
        normalization=norm,
        state_dict=state,
    )


def save_artifact(artifact: CalculixSurrogateArtifact, path: str | Path) -> None:
    """Serialize artifact to disk."""
    require_torch()
    assert torch is not None  # for type checkers

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_keys": list(artifact.feature_keys),
        "target_keys": list(artifact.target_keys),
        "hidden_layers": list(artifact.hidden_layers),
        "normalization": {
            "x_mean": artifact.normalization.x_mean,
            "x_std": artifact.normalization.x_std,
            "y_mean": artifact.normalization.y_mean,
            "y_std": artifact.normalization.y_std,
        },
        "state_dict": artifact.state_dict,
    }
    torch.save(payload, p)


def load_artifact(path: str | Path) -> CalculixSurrogateArtifact:
    """Load serialized artifact from disk."""
    require_torch()
    assert torch is not None  # for type checkers

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch versions
        payload = torch.load(Path(path), map_location="cpu")
    norm = payload["normalization"]
    return CalculixSurrogateArtifact(
        feature_keys=tuple(payload["feature_keys"]),
        target_keys=tuple(payload["target_keys"]),
        hidden_layers=tuple(int(h) for h in payload["hidden_layers"]),
        normalization=Normalization(
            x_mean=np.asarray(norm["x_mean"], dtype=np.float64),
            x_std=np.asarray(norm["x_std"], dtype=np.float64),
            y_mean=np.asarray(norm["y_mean"], dtype=np.float64),
            y_std=np.asarray(norm["y_std"], dtype=np.float64),
        ),
        state_dict=payload["state_dict"],
    )
