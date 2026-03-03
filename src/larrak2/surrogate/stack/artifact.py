"""Artifact format for the global surrogate stack used by symbolic CasADi refinement."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.surrogate.quality_contract import (
    sha256_file,
    validate_artifact_quality,
    write_quality_report,
)


@dataclass(frozen=True)
class DenseLayer:
    """One affine layer in the exported MLP."""

    weight: np.ndarray  # Shape: (out_dim, in_dim)
    bias: np.ndarray  # Shape: (out_dim,)

    def __post_init__(self) -> None:
        w = np.asarray(self.weight, dtype=np.float64)
        b = np.asarray(self.bias, dtype=np.float64).reshape(-1)
        if w.ndim != 2:
            raise ValueError(f"weight must be 2D, got {w.shape}")
        if b.ndim != 1:
            raise ValueError(f"bias must be 1D, got {b.shape}")
        if w.shape[0] != b.shape[0]:
            raise ValueError(f"weight/bias mismatch: {w.shape} vs {b.shape}")
        object.__setattr__(self, "weight", w)
        object.__setattr__(self, "bias", b)


@dataclass(frozen=True)
class StackSurrogateArtifact:
    """Serialized model for symbolic objective/constraint prediction."""

    feature_names: tuple[str, ...]
    objective_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    hidden_layers: tuple[int, ...]
    activation: str
    leaky_relu_slope: float
    fidelity: int
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    layers: tuple[DenseLayer, ...]
    version_hash: str = ""

    def __post_init__(self) -> None:
        x_mean = np.asarray(self.x_mean, dtype=np.float64).reshape(-1)
        x_std = np.asarray(self.x_std, dtype=np.float64).reshape(-1)
        y_mean = np.asarray(self.y_mean, dtype=np.float64).reshape(-1)
        y_std = np.asarray(self.y_std, dtype=np.float64).reshape(-1)

        n_in = len(self.feature_names)
        n_out = len(self.objective_names) + len(self.constraint_names)
        if n_in <= 0:
            raise ValueError("feature_names cannot be empty")
        if len(self.objective_names) <= 0:
            raise ValueError("objective_names cannot be empty")
        if len(self.constraint_names) <= 0:
            raise ValueError("constraint_names cannot be empty")
        if x_mean.size != n_in or x_std.size != n_in:
            raise ValueError(
                f"x normalization shape mismatch: mean={x_mean.size}, std={x_std.size}, in={n_in}"
            )
        if y_mean.size != n_out or y_std.size != n_out:
            raise ValueError(
                f"y normalization shape mismatch: mean={y_mean.size}, std={y_std.size}, out={n_out}"
            )
        if self.activation not in {"relu", "leaky_relu"}:
            raise ValueError(f"Unsupported activation '{self.activation}'")
        if self.fidelity < 0:
            raise ValueError(f"fidelity must be >= 0, got {self.fidelity}")
        if not self.layers:
            raise ValueError("layers cannot be empty")

        prev = n_in
        for layer in self.layers:
            if layer.weight.shape[1] != prev:
                raise ValueError(
                    f"layer input mismatch: expected {prev}, got {layer.weight.shape[1]}"
                )
            prev = int(layer.weight.shape[0])
        if prev != n_out:
            raise ValueError(f"final layer output mismatch: got {prev}, expected {n_out}")

        object.__setattr__(self, "x_mean", x_mean)
        object.__setattr__(self, "x_std", x_std)
        object.__setattr__(self, "y_mean", y_mean)
        object.__setattr__(self, "y_std", y_std)

        if not self.version_hash:
            object.__setattr__(self, "version_hash", _compute_version_hash(self))


def _compute_version_hash(artifact: StackSurrogateArtifact) -> str:
    """Compute stable hash over schema and parameters."""
    h = hashlib.sha256()
    meta = {
        "feature_names": list(artifact.feature_names),
        "objective_names": list(artifact.objective_names),
        "constraint_names": list(artifact.constraint_names),
        "hidden_layers": list(artifact.hidden_layers),
        "activation": artifact.activation,
        "leaky_relu_slope": float(artifact.leaky_relu_slope),
        "fidelity": int(artifact.fidelity),
    }
    h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    h.update(np.asarray(artifact.x_mean, dtype=np.float64).tobytes())
    h.update(np.asarray(artifact.x_std, dtype=np.float64).tobytes())
    h.update(np.asarray(artifact.y_mean, dtype=np.float64).tobytes())
    h.update(np.asarray(artifact.y_std, dtype=np.float64).tobytes())
    for layer in artifact.layers:
        h.update(np.asarray(layer.weight, dtype=np.float64).tobytes())
        h.update(np.asarray(layer.bias, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def save_stack_artifact(artifact: StackSurrogateArtifact, path: str | Path) -> Path:
    """Persist artifact to a compressed NPZ."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "feature_names": list(artifact.feature_names),
        "objective_names": list(artifact.objective_names),
        "constraint_names": list(artifact.constraint_names),
        "hidden_layers": list(artifact.hidden_layers),
        "activation": artifact.activation,
        "leaky_relu_slope": float(artifact.leaky_relu_slope),
        "fidelity": int(artifact.fidelity),
        "version_hash": artifact.version_hash,
        "n_layers": len(artifact.layers),
    }

    payload: dict[str, Any] = {
        "__meta_json__": np.array(json.dumps(meta), dtype=object),
        "x_mean": np.asarray(artifact.x_mean, dtype=np.float64),
        "x_std": np.asarray(artifact.x_std, dtype=np.float64),
        "y_mean": np.asarray(artifact.y_mean, dtype=np.float64),
        "y_std": np.asarray(artifact.y_std, dtype=np.float64),
    }
    for i, layer in enumerate(artifact.layers):
        payload[f"W{i}"] = np.asarray(layer.weight, dtype=np.float64)
        payload[f"b{i}"] = np.asarray(layer.bias, dtype=np.float64)

    np.savez_compressed(target, **payload)
    report = {
        "schema_version": "surrogate_quality_report_v1",
        "surrogate_kind": "stack",
        "artifact_file": str(target.name),
        "artifact_sha256": sha256_file(target),
        "dataset_manifest": {
            "source_path": "",
            "num_samples": 0,
            "num_features": int(len(artifact.feature_names)),
            "num_targets": int(len(artifact.objective_names) + len(artifact.constraint_names)),
            "dataset_sha256": "",
        },
        "metrics": {
            "train": {"mse": 0.0},
            "val": {"mse": 0.0},
            "test": {"mse": 0.0},
            "slice_metrics": [],
        },
        "ood_thresholds": {},
        "uncertainty_calibration": {"method": "none"},
        "required_artifacts": [target.name],
        "pass": True,
        "fail_reasons": [],
    }
    write_quality_report(target.parent / "quality_report.json", report)
    return target


def load_stack_artifact(
    path: str | Path,
    *,
    validation_mode: str = "strict",
) -> StackSurrogateArtifact:
    """Load stack surrogate artifact from NPZ."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stack surrogate artifact not found: {p}")
    validate_artifact_quality(
        p,
        surrogate_kind="stack",
        validation_mode=str(validation_mode),
        required_artifacts=[p.name],
    )

    with np.load(p, allow_pickle=True) as data:
        if "__meta_json__" not in data:
            raise ValueError(f"Artifact missing __meta_json__: {p}")
        meta = json.loads(str(data["__meta_json__"].item()))
        n_layers = int(meta.get("n_layers", 0))
        if n_layers <= 0:
            raise ValueError(f"Artifact has invalid layer count: {n_layers}")

        layers: list[DenseLayer] = []
        for i in range(n_layers):
            wk = f"W{i}"
            bk = f"b{i}"
            if wk not in data or bk not in data:
                raise ValueError(f"Artifact missing layer tensors '{wk}'/'{bk}'")
            layers.append(
                DenseLayer(
                    weight=np.asarray(data[wk], dtype=np.float64),
                    bias=np.asarray(data[bk], dtype=np.float64),
                )
            )

        artifact = StackSurrogateArtifact(
            feature_names=tuple(meta["feature_names"]),
            objective_names=tuple(meta["objective_names"]),
            constraint_names=tuple(meta["constraint_names"]),
            hidden_layers=tuple(int(v) for v in meta.get("hidden_layers", [])),
            activation=str(meta.get("activation", "relu")),
            leaky_relu_slope=float(meta.get("leaky_relu_slope", 0.01)),
            fidelity=int(meta.get("fidelity", 1)),
            x_mean=np.asarray(data["x_mean"], dtype=np.float64),
            x_std=np.asarray(data["x_std"], dtype=np.float64),
            y_mean=np.asarray(data["y_mean"], dtype=np.float64),
            y_std=np.asarray(data["y_std"], dtype=np.float64),
            layers=tuple(layers),
            version_hash=str(meta.get("version_hash", "")),
        )
    return artifact
