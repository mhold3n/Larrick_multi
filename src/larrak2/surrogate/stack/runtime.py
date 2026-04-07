"""Runtime inference helpers for stack surrogate artifacts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from larrak_runtime.core.encoding import resolve_index_for_encoding

from .artifact import StackSurrogateArtifact, load_stack_artifact

_X_NAME_RE = re.compile(r"^x[_-]?(\d+)$")


def parse_feature_index(name: str) -> int | None:
    """Parse x-feature names like x_000, x12, x-42."""
    m = _X_NAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def default_feature_names(n_total: int) -> tuple[str, ...]:
    """Canonical feature schema for stack models."""
    return tuple([f"x_{i:03d}" for i in range(int(n_total))] + ["rpm", "torque"])


def feature_vector_from_inputs(
    feature_names: tuple[str, ...],
    x_full: np.ndarray,
    *,
    rpm: float,
    torque: float,
    encoding_version: str | None = None,
) -> np.ndarray:
    """Build feature vector in schema order from full design + operating point."""
    x = np.asarray(x_full, dtype=np.float64).reshape(-1)
    out = np.zeros(len(feature_names), dtype=np.float64)
    for i, name in enumerate(feature_names):
        if name == "rpm":
            out[i] = float(rpm)
            continue
        if name == "torque":
            out[i] = float(torque)
            continue
        idx = parse_feature_index(name)
        if idx is None:
            raise ValueError(
                f"Unsupported feature '{name}'. Only x_###, rpm, torque are supported."
            )
        idx = resolve_index_for_encoding(idx, encoding_version)
        if idx < 0 or idx >= x.size:
            raise ValueError(f"Feature '{name}' index {idx} out of range for vector size {x.size}")
        out[i] = float(x[idx])
    return out


class StackSurrogateRuntime:
    """Numeric inference runtime over exported dense layers."""

    def __init__(self, artifact: StackSurrogateArtifact):
        self.artifact = artifact
        self._x_scale = np.where(np.abs(artifact.x_std) > 0.0, artifact.x_std, 1.0)
        self._y_scale = np.where(np.abs(artifact.y_std) > 0.0, artifact.y_std, 1.0)
        self.n_obj = len(artifact.objective_names)
        self.n_constr = len(artifact.constraint_names)

    def predict_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict objective and constraint vectors from feature row."""
        x = np.asarray(features, dtype=np.float64).reshape(-1)
        if x.size != len(self.artifact.feature_names):
            raise ValueError(
                f"Feature length mismatch: expected {len(self.artifact.feature_names)}, got {x.size}"
            )
        h = (x - self.artifact.x_mean) / self._x_scale
        for i, layer in enumerate(self.artifact.layers):
            h = layer.weight @ h + layer.bias
            if i < len(self.artifact.layers) - 1:
                if self.artifact.activation == "relu":
                    h = np.maximum(h, 0.0)
                elif self.artifact.activation == "leaky_relu":
                    a = float(self.artifact.leaky_relu_slope)
                    h = np.where(h >= 0.0, h, a * h)
                else:
                    raise ValueError(f"Unsupported activation '{self.artifact.activation}'")
        y = h * self._y_scale + self.artifact.y_mean
        return y[: self.n_obj], y[self.n_obj :]

    def predict_design(
        self, x_full: np.ndarray, *, rpm: float, torque: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict from full design vector + operating point."""
        feats = feature_vector_from_inputs(
            self.artifact.feature_names,
            x_full,
            rpm=float(rpm),
            torque=float(torque),
            encoding_version=getattr(self.artifact, "encoding_version", None),
        )
        return self.predict_features(feats)


_RUNTIME_CACHE: dict[Path, StackSurrogateRuntime] = {}


def load_stack_runtime(
    path: str | Path, *, validation_mode: str = "strict"
) -> StackSurrogateRuntime:
    """Load and cache runtime by absolute path."""
    p = Path(path).resolve()
    if p not in _RUNTIME_CACHE:
        _RUNTIME_CACHE[p] = StackSurrogateRuntime(
            load_stack_artifact(p, validation_mode=str(validation_mode))
        )
    return _RUNTIME_CACHE[p]
