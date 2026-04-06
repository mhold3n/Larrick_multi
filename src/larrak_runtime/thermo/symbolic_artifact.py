"""Artifact format and training utilities for thermo symbolic surrogates."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.encoding import ENCODING_VERSION, LEGACY_ENCODING_VERSION
from ..surrogate.quality_contract import (
    load_quality_report,
    regression_metrics,
    sha256_file,
    thermo_symbolic_balanced_profile,
    thermo_symbolic_quality_fail_reasons,
    validate_artifact_quality,
    validate_thermo_symbolic_quality,
    write_quality_report,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThermoSymbolicArtifact:
    """Affine surrogate artifact used to overlay thermo terms in symbolic NLP."""

    feature_names: tuple[str, ...]
    objective_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    fidelity: int
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    weight: np.ndarray  # shape: (n_out, n_in)
    bias: np.ndarray  # shape: (n_out,)
    encoding_version: str = ENCODING_VERSION
    version_hash: str = ""

    def __post_init__(self) -> None:
        n_in = len(self.feature_names)
        n_obj = len(self.objective_names)
        n_con = len(self.constraint_names)
        n_out = n_obj + n_con
        if n_in <= 0:
            raise ValueError("feature_names cannot be empty")
        if n_out <= 0:
            raise ValueError("objective_names + constraint_names cannot both be empty")
        if int(self.fidelity) < 0:
            raise ValueError("fidelity must be >= 0")

        x_mean = np.asarray(self.x_mean, dtype=np.float64).reshape(-1)
        x_std = np.asarray(self.x_std, dtype=np.float64).reshape(-1)
        y_mean = np.asarray(self.y_mean, dtype=np.float64).reshape(-1)
        y_std = np.asarray(self.y_std, dtype=np.float64).reshape(-1)
        weight = np.asarray(self.weight, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64).reshape(-1)

        if x_mean.size != n_in or x_std.size != n_in:
            raise ValueError("x normalization shape mismatch")
        if y_mean.size != n_out or y_std.size != n_out:
            raise ValueError("y normalization shape mismatch")
        if weight.shape != (n_out, n_in):
            raise ValueError(f"weight shape mismatch: expected {(n_out, n_in)}, got {weight.shape}")
        if bias.size != n_out:
            raise ValueError("bias shape mismatch")

        object.__setattr__(self, "x_mean", x_mean)
        object.__setattr__(self, "x_std", x_std)
        object.__setattr__(self, "y_mean", y_mean)
        object.__setattr__(self, "y_std", y_std)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)
        if not self.version_hash:
            object.__setattr__(self, "version_hash", _compute_version_hash(self))


def _compute_version_hash(artifact: ThermoSymbolicArtifact) -> str:
    h = hashlib.sha256()
    meta = {
        "feature_names": list(artifact.feature_names),
        "objective_names": list(artifact.objective_names),
        "constraint_names": list(artifact.constraint_names),
        "fidelity": int(artifact.fidelity),
    }
    h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    h.update(np.asarray(artifact.x_mean, dtype=np.float64).tobytes())
    h.update(np.asarray(artifact.x_std, dtype=np.float64).tobytes())
    h.update(np.asarray(artifact.y_mean, dtype=np.float64).tobytes())
    h.update(np.asarray(artifact.y_std, dtype=np.float64).tobytes())
    h.update(np.asarray(artifact.weight, dtype=np.float64).tobytes())
    h.update(np.asarray(artifact.bias, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _split_indices(
    n: int,
    *,
    seed: int,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    n_val = max(1, int(round(float(val_frac) * n)))
    n_val = min(n_val, max(1, n - 2)) if n >= 3 else min(n_val, max(0, n - 1))
    n_test = n_val if n >= 5 else max(1, n - n_val - 1)
    n_test = min(n_test, max(0, n - n_val - 1))
    n_train = max(1, n - n_val - n_test)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _predict_affine(
    *,
    weight: np.ndarray,
    bias: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    x_scale = np.where(np.abs(x_std) > 0.0, x_std, 1.0)
    y_scale = np.where(np.abs(y_std) > 0.0, y_std, 1.0)
    Xn = (np.asarray(X, dtype=np.float64) - x_mean.reshape(1, -1)) / x_scale.reshape(1, -1)
    Yn = Xn @ weight.T + bias.reshape(1, -1)
    return Yn * y_scale.reshape(1, -1) + y_mean.reshape(1, -1)


def _target_scales_from_train(y_train: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] == 0:
        return np.ones(y.shape[1] if y.ndim == 2 else 0, dtype=np.float64)
    p05 = np.quantile(y, 0.05, axis=0)
    p95 = np.quantile(y, 0.95, axis=0)
    return np.maximum(p95 - p05, 1e-9).astype(np.float64)


def _per_target_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: tuple[str, ...],
    target_scale: np.ndarray,
) -> list[dict[str, float | str]]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    out: list[dict[str, float | str]] = []
    for j, name in enumerate(target_names):
        if yt.ndim != 2 or yp.ndim != 2 or yt.shape[0] == 0:
            rmse = float("nan")
            mae = float("nan")
            r2 = float("nan")
        else:
            m = regression_metrics(yt[:, j], yp[:, j])
            rmse = float(m["rmse"])
            mae = float(m["mae"])
            r2 = float(m["r2"])
        scale = float(target_scale[j]) if j < target_scale.size else float("nan")
        nrmse = (
            float(rmse / scale)
            if np.isfinite(rmse) and np.isfinite(scale) and scale > 0.0
            else float("nan")
        )
        out.append(
            {
                "name": str(name),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "nrmse": nrmse,
                "target_scale": scale,
            }
        )
    return out


def train_thermo_symbolic_affine(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    objective_names: tuple[str, ...],
    constraint_names: tuple[str, ...],
    fidelity: int,
    seed: int = 42,
    val_frac: float = 0.2,
) -> tuple[ThermoSymbolicArtifact, dict[str, Any]]:
    """Fit a normalized affine surrogate and return artifact + metrics."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X/Y row mismatch")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length mismatch")
    if Y.shape[1] != len(objective_names) + len(constraint_names):
        raise ValueError("target schema mismatch")
    if X.shape[0] < 5:
        raise ValueError("Need at least 5 samples for train/val/test splits")

    train_idx, val_idx, test_idx = _split_indices(
        X.shape[0], seed=int(seed), val_frac=float(val_frac)
    )
    X_tr = X[train_idx]
    Y_tr = Y[train_idx]

    x_mean = np.mean(X_tr, axis=0)
    x_std = np.std(X_tr, axis=0)
    y_mean = np.mean(Y_tr, axis=0)
    y_std = np.std(Y_tr, axis=0)
    x_scale = np.where(np.abs(x_std) > 0.0, x_std, 1.0)
    y_scale = np.where(np.abs(y_std) > 0.0, y_std, 1.0)

    Xn = (X_tr - x_mean.reshape(1, -1)) / x_scale.reshape(1, -1)
    Yn = (Y_tr - y_mean.reshape(1, -1)) / y_scale.reshape(1, -1)
    Xn_aug = np.hstack([Xn, np.ones((Xn.shape[0], 1), dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(Xn_aug, Yn, rcond=None)
    weight = np.asarray(coef[:-1, :].T, dtype=np.float64)
    bias = np.asarray(coef[-1, :], dtype=np.float64)

    artifact = ThermoSymbolicArtifact(
        feature_names=tuple(feature_names),
        objective_names=tuple(objective_names),
        constraint_names=tuple(constraint_names),
        fidelity=int(fidelity),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        weight=weight,
        bias=bias,
    )

    Y_pred = _predict_affine(
        weight=artifact.weight,
        bias=artifact.bias,
        x_mean=artifact.x_mean,
        x_std=artifact.x_std,
        y_mean=artifact.y_mean,
        y_std=artifact.y_std,
        X=X,
    )

    target_names = tuple(str(v) for v in objective_names) + tuple(str(v) for v in constraint_names)
    target_scale = _target_scales_from_train(Y_tr)

    def _split_metrics(idx: np.ndarray) -> dict[str, Any]:
        if idx.size == 0:
            return {
                "rmse": float("nan"),
                "mae": float("nan"),
                "r2": float("nan"),
                "per_target": _per_target_metrics(
                    y_true=np.zeros((0, Y.shape[1]), dtype=np.float64),
                    y_pred=np.zeros((0, Y.shape[1]), dtype=np.float64),
                    target_names=target_names,
                    target_scale=target_scale,
                ),
            }
        agg = regression_metrics(Y[idx], Y_pred[idx])
        return {
            "rmse": float(agg["rmse"]),
            "mae": float(agg["mae"]),
            "r2": float(agg["r2"]),
            "per_target": _per_target_metrics(
                y_true=Y[idx],
                y_pred=Y_pred[idx],
                target_names=target_names,
                target_scale=target_scale,
            ),
        }

    train_m = _split_metrics(train_idx)
    val_m = _split_metrics(val_idx)
    test_m = _split_metrics(test_idx)
    metrics = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_targets": int(Y.shape[1]),
        "train": train_m,
        "val": val_m,
        "test": test_m,
        "normalization_method": "p95_p05_range",
        "target_names": list(target_names),
        "target_scale": {
            str(target_names[i]): float(target_scale[i]) for i in range(len(target_names))
        },
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }
    return artifact, metrics


def save_thermo_symbolic_artifact(
    artifact: ThermoSymbolicArtifact,
    path: str | Path,
    *,
    quality_report: dict[str, Any] | None = None,
) -> Path:
    """Persist thermo symbolic artifact and optional quality report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "feature_names": list(artifact.feature_names),
        "objective_names": list(artifact.objective_names),
        "constraint_names": list(artifact.constraint_names),
        "fidelity": int(artifact.fidelity),
        "encoding_version": str(artifact.encoding_version),
        "version_hash": str(artifact.version_hash),
        "model_family": "affine_v1",
    }
    np.savez_compressed(
        target,
        __meta_json__=np.array(json.dumps(meta), dtype=object),
        x_mean=np.asarray(artifact.x_mean, dtype=np.float64),
        x_std=np.asarray(artifact.x_std, dtype=np.float64),
        y_mean=np.asarray(artifact.y_mean, dtype=np.float64),
        y_std=np.asarray(artifact.y_std, dtype=np.float64),
        weight=np.asarray(artifact.weight, dtype=np.float64),
        bias=np.asarray(artifact.bias, dtype=np.float64),
    )

    target_names = tuple(str(v) for v in artifact.objective_names) + tuple(
        str(v) for v in artifact.constraint_names
    )
    default_per_target = [
        {
            "name": str(name),
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 1.0,
            "nrmse": 0.0,
            "target_scale": 1.0,
        }
        for name in target_names
    ]
    report = dict(quality_report or {})
    if not report:
        report = {
            "schema_version": "surrogate_quality_report_v1",
            "surrogate_kind": "thermo_symbolic",
            "artifact_file": target.name,
            "artifact_sha256": "",
            "dataset_manifest": {
                "source_path": "",
                "num_samples": 0,
                "num_features": int(len(artifact.feature_names)),
                "num_targets": int(len(artifact.objective_names) + len(artifact.constraint_names)),
                "dataset_sha256": "",
            },
            "metrics": {
                "train": {
                    "rmse": 0.0,
                    "mae": 0.0,
                    "r2": 1.0,
                    "per_target": list(default_per_target),
                },
                "val": {
                    "rmse": 0.0,
                    "mae": 0.0,
                    "r2": 1.0,
                    "per_target": list(default_per_target),
                },
                "test": {
                    "rmse": 0.0,
                    "mae": 0.0,
                    "r2": 1.0,
                    "per_target": list(default_per_target),
                },
                "slice_metrics": [],
            },
            "quality_profile": thermo_symbolic_balanced_profile(),
            "ood_thresholds": {},
            "uncertainty_calibration": {
                "method": "deterministic_affine",
                "status": "not_applicable",
            },
            "required_artifacts": [target.name],
            "pass": True,
            "fail_reasons": [],
        }
    report["schema_version"] = str(report.get("schema_version", "surrogate_quality_report_v1"))
    report["surrogate_kind"] = "thermo_symbolic"
    report["artifact_file"] = target.name
    report["quality_profile"] = dict(
        thermo_symbolic_balanced_profile() | dict(report.get("quality_profile", {}) or {})
    )
    report["required_artifacts"] = [target.name]
    report["artifact_sha256"] = sha256_file(target)
    reasons = thermo_symbolic_quality_fail_reasons(report)
    report["pass"] = bool(len(reasons) == 0)
    report["fail_reasons"] = reasons
    write_quality_report(target.parent / "quality_report.json", report)
    return target


def _raise_or_warn(mode: str, message: str) -> None:
    m = str(mode).strip().lower()
    if m == "strict":
        raise ValueError(message)
    if m == "warn":
        LOGGER.warning(message)


def load_thermo_symbolic_artifact(
    path: str | Path,
    *,
    validation_mode: str = "strict",
) -> ThermoSymbolicArtifact:
    """Load thermo symbolic artifact from NPZ and validate quality contract."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Thermo symbolic artifact not found: {p}")
    report = validate_artifact_quality(
        p,
        surrogate_kind="thermo_symbolic",
        validation_mode=str(validation_mode),
        required_artifacts=[p.name],
    )
    with np.load(p, allow_pickle=True) as data:
        if "__meta_json__" not in data:
            raise ValueError(f"Artifact missing __meta_json__: {p}")
        meta = json.loads(str(data["__meta_json__"].item()))
        artifact = ThermoSymbolicArtifact(
            feature_names=tuple(meta["feature_names"]),
            objective_names=tuple(meta["objective_names"]),
            constraint_names=tuple(meta["constraint_names"]),
            fidelity=int(meta.get("fidelity", 1)),
            encoding_version=str(meta.get("encoding_version", LEGACY_ENCODING_VERSION)),
            x_mean=np.asarray(data["x_mean"], dtype=np.float64),
            x_std=np.asarray(data["x_std"], dtype=np.float64),
            y_mean=np.asarray(data["y_mean"], dtype=np.float64),
            y_std=np.asarray(data["y_std"], dtype=np.float64),
            weight=np.asarray(data["weight"], dtype=np.float64),
            bias=np.asarray(data["bias"], dtype=np.float64),
            version_hash=str(meta.get("version_hash", "")),
        )
    declared_hash = str(artifact.version_hash).strip()
    computed_hash = _compute_version_hash(artifact)
    if declared_hash and declared_hash != computed_hash:
        _raise_or_warn(
            validation_mode,
            "Thermo symbolic artifact version hash mismatch: "
            f"declared={declared_hash}, computed={computed_hash}, path={p}",
        )
    if not declared_hash:
        artifact = ThermoSymbolicArtifact(
            feature_names=artifact.feature_names,
            objective_names=artifact.objective_names,
            constraint_names=artifact.constraint_names,
            fidelity=artifact.fidelity,
            encoding_version=artifact.encoding_version,
            x_mean=artifact.x_mean,
            x_std=artifact.x_std,
            y_mean=artifact.y_mean,
            y_std=artifact.y_std,
            weight=artifact.weight,
            bias=artifact.bias,
            version_hash=computed_hash,
        )
    if str(validation_mode).strip().lower() != "off":
        qpath = p.parent / "quality_report.json"
        if report is None and qpath.exists():
            report = load_quality_report(qpath)
        if isinstance(report, dict):
            validate_thermo_symbolic_quality(report, validation_mode=str(validation_mode))
    return artifact
