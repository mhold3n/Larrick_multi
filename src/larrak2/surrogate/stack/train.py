"""Training and export helpers for the global stack surrogate model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .artifact import DenseLayer, StackSurrogateArtifact

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - runtime dependency guard
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_ERR = exc
else:
    _TORCH_ERR = None


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover - dependency guard
        raise ImportError(
            "PyTorch is required for stack surrogate training. Install project dependencies."
        ) from _TORCH_ERR


@dataclass(frozen=True)
class Normalization:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    @staticmethod
    def fit(X: np.ndarray, Y: np.ndarray) -> Normalization:
        return Normalization(
            x_mean=np.mean(X, axis=0),
            x_std=np.std(X, axis=0),
            y_mean=np.mean(Y, axis=0),
            y_std=np.std(Y, axis=0),
        )

    def normalize_x(self, X: np.ndarray) -> np.ndarray:
        scale = np.where(np.abs(self.x_std) > 0.0, self.x_std, 1.0)
        return (X - self.x_mean) / scale

    def normalize_y(self, Y: np.ndarray) -> np.ndarray:
        scale = np.where(np.abs(self.y_std) > 0.0, self.y_std, 1.0)
        return (Y - self.y_mean) / scale


class StackMLP(nn.Module):
    """Simple MLP used for stack surrogate training."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_layers: tuple[int, ...],
        output_dim: int,
        activation: str = "relu",
        leaky_relu_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if activation not in {"relu", "leaky_relu"}:
            raise ValueError(f"Unsupported activation '{activation}'")
        self.activation = activation
        self.leaky_relu_slope = float(leaky_relu_slope)

        modules: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_layers:
            modules.append(nn.Linear(prev, int(h)))
            if activation == "relu":
                modules.append(nn.ReLU())
            else:
                modules.append(nn.LeakyReLU(negative_slope=float(leaky_relu_slope)))
            prev = int(h)
        modules.append(nn.Linear(prev, int(output_dim)))
        self.net = nn.Sequential(*modules)

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


def export_torch_mlp_artifact(
    *,
    model: StackMLP,
    normalization: Normalization,
    feature_names: tuple[str, ...],
    objective_names: tuple[str, ...],
    constraint_names: tuple[str, ...],
    fidelity: int,
) -> StackSurrogateArtifact:
    """Export trained torch model to portable stack artifact."""
    _require_torch()
    layers: list[DenseLayer] = []
    for module in model.net:
        if isinstance(module, nn.Linear):
            w = module.weight.detach().cpu().numpy().astype(np.float64, copy=True)
            b = module.bias.detach().cpu().numpy().astype(np.float64, copy=True)
            layers.append(DenseLayer(weight=w, bias=b))

    hidden_layers = tuple(layer.weight.shape[0] for layer in layers[:-1])
    return StackSurrogateArtifact(
        feature_names=feature_names,
        objective_names=objective_names,
        constraint_names=constraint_names,
        hidden_layers=hidden_layers,
        activation=model.activation,
        leaky_relu_slope=float(model.leaky_relu_slope),
        fidelity=int(fidelity),
        x_mean=np.asarray(normalization.x_mean, dtype=np.float64),
        x_std=np.asarray(normalization.x_std, dtype=np.float64),
        y_mean=np.asarray(normalization.y_mean, dtype=np.float64),
        y_std=np.asarray(normalization.y_std, dtype=np.float64),
        layers=tuple(layers),
    )


def train_stack_surrogate(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    objective_names: tuple[str, ...],
    constraint_names: tuple[str, ...],
    fidelity: int,
    hidden_layers: tuple[int, ...] = (128, 128),
    activation: str = "relu",
    leaky_relu_slope: float = 0.01,
    seed: int = 42,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[StackSurrogateArtifact, dict[str, Any]]:
    """Train stack MLP and export artifact with summary metrics."""
    _require_torch()
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X/Y row mismatch: {X.shape[0]} vs {Y.shape[0]}")
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 samples for stack surrogate training")

    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(X.shape[0])
    n_val = max(1, int(round(float(val_frac) * X.shape[0])))
    n_test = max(1, int(round(float(test_frac) * X.shape[0])))
    max_holdout = max(1, X.shape[0] - 1)
    if n_val + n_test > max_holdout:
        total_holdout = max(1, min(max_holdout, n_val + n_test))
        n_val = max(
            1, int(round(total_holdout * float(val_frac) / max(float(val_frac + test_frac), 1e-12)))
        )
        n_test = max(1, total_holdout - n_val)
    n_val = min(n_val, X.shape[0] - 1)
    n_test = min(n_test, max(0, X.shape[0] - n_val - 1))
    if n_test <= 0:
        n_test = 1
        n_val = max(1, n_val - 1)

    X = X[idx]
    Y = Y[idx]
    train_end = X.shape[0] - n_val - n_test
    X_train = X[:train_end]
    Y_train = Y[:train_end]
    X_val = X[train_end : train_end + n_val]
    Y_val = Y[train_end : train_end + n_val]
    X_test = X[train_end + n_val :]
    Y_test = Y[train_end + n_val :]

    norm = Normalization.fit(X_train, Y_train)
    Xn_train = norm.normalize_x(X_train)
    Yn_train = norm.normalize_y(Y_train)
    Xn_val = norm.normalize_x(X_val)
    Yn_val = norm.normalize_y(Y_val)
    Xn_test = norm.normalize_x(X_test)
    Yn_test = norm.normalize_y(Y_test)

    torch.manual_seed(int(seed))
    model = StackMLP(
        input_dim=X.shape[1],
        hidden_layers=tuple(int(v) for v in hidden_layers),
        output_dim=Y.shape[1],
        activation=activation,
        leaky_relu_slope=float(leaky_relu_slope),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()

    xt = torch.tensor(Xn_train, dtype=torch.float32)
    yt = torch.tensor(Yn_train, dtype=torch.float32)
    xv = torch.tensor(Xn_val, dtype=torch.float32)
    yv = torch.tensor(Yn_val, dtype=torch.float32)
    xte = torch.tensor(Xn_test, dtype=torch.float32)
    yte = torch.tensor(Yn_test, dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    patience = 40
    patience_left = patience

    for _ in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad()
        pred = model(xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv).item())
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    artifact = export_torch_mlp_artifact(
        model=model,
        normalization=norm,
        feature_names=feature_names,
        objective_names=objective_names,
        constraint_names=constraint_names,
        fidelity=int(fidelity),
    )

    model.eval()
    with torch.no_grad():
        train_loss = float(loss_fn(model(xt), yt).item())
        val_loss = float(loss_fn(model(xv), yv).item())
        test_loss = float(loss_fn(model(xte), yte).item())

    metrics = {
        "n_samples": int(X.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "train_mse_norm": train_loss,
        "val_mse_norm": val_loss,
        "test_mse_norm": test_loss,
        "hidden_layers": list(hidden_layers),
        "activation": activation,
    }
    return artifact, metrics
