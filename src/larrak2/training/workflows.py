"""Training Workflows for various surrogate models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.core.archive_io import load_archive
from larrak2.core.artifact_paths import (
    DEFAULT_STACK_SURROGATE_DIR,
    assert_not_legacy_models_path,
    assert_not_legacy_models_write,
)
from larrak2.core.constraints import get_constraint_names
from larrak2.core.encoding import N_TOTAL, mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext
from larrak2.surrogate.quality_contract import (
    dataset_manifest_for_file,
    regression_metrics,
    sha256_file,
    write_quality_report,
)
from larrak2.surrogate.stack import (
    default_feature_names,
    save_stack_artifact,
    train_stack_surrogate,
)

# OpenFOAM Imports
try:
    from larrak2.surrogate.openfoam_nn import (
        DEFAULT_FEATURE_KEYS,
        DEFAULT_TARGET_KEYS,
        load_dataset_json,
        load_dataset_jsonl,
        load_dataset_npz,
        save_artifact,
        train_openfoam_surrogate,
    )
except ImportError:
    pass


def _split_indices(
    n: int,
    *,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    n_train = max(1, int(round(float(train_frac) * n)))
    n_train = min(n_train, max(1, n - 2)) if n >= 3 else min(n_train, n)
    n_val = max(1, int(round(float(val_frac) * n))) if n >= 3 else max(0, n - n_train)
    n_val = min(n_val, max(0, n - n_train - 1)) if n >= 3 else n_val
    n_test = max(0, n - n_train - n_val)
    if n_test == 0 and n >= 3:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train = max(1, n_train - 1)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val : n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def _stack_quality_report(
    *,
    artifact_path: Path,
    source_meta: dict[str, Any],
    feature_names: tuple[str, ...],
    objective_names: tuple[str, ...],
    constraint_names: tuple[str, ...],
    metrics: dict[str, Any],
) -> None:
    val_mse = float(metrics.get("val_mse_norm", float("nan")))
    train_mse = float(metrics.get("train_mse_norm", float("nan")))
    passed = bool(np.isfinite(val_mse) and np.isfinite(train_mse))
    report = {
        "schema_version": "surrogate_quality_report_v1",
        "surrogate_kind": "stack",
        "artifact_file": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "dataset_manifest": {
            "source": source_meta,
            "num_samples": int(metrics.get("n_samples", 0)),
            "num_features": int(len(feature_names)),
            "num_targets": int(len(objective_names) + len(constraint_names)),
        },
        "metrics": {
            "train": {"mse_norm": train_mse},
            "val": {"mse_norm": val_mse},
            "test": {"mse_norm": val_mse},
            "slice_metrics": [],
        },
        "ood_thresholds": {"max_val_mse_norm": float(max(0.5, 2.0 * train_mse if np.isfinite(train_mse) else 1.0))},
        "uncertainty_calibration": {"method": "deterministic_mlp", "status": "not_applicable"},
        "required_artifacts": [artifact_path.name],
        "pass": passed,
        "fail_reasons": [] if passed else ["non-finite training/validation metrics"],
    }
    write_quality_report(artifact_path.parent / "quality_report.json", report)


def _predict_openfoam_batch(artifact: Any, X: np.ndarray) -> np.ndarray:
    from larrak2.surrogate.openfoam_nn import OpenFoamSurrogate

    model = OpenFoamSurrogate(artifact)
    preds = np.zeros((X.shape[0], len(artifact.target_keys)), dtype=np.float64)
    for i in range(X.shape[0]):
        row = {k: float(X[i, j]) for j, k in enumerate(artifact.feature_keys)}
        out = model.predict_one(row)
        preds[i, :] = [float(out[k]) for k in artifact.target_keys]
    return preds


def _predict_calculix_batch(artifact: Any, X: np.ndarray) -> np.ndarray:
    from larrak2.surrogate.calculix_nn import CalculixSurrogate

    model = CalculixSurrogate(artifact)
    preds = np.zeros((X.shape[0], len(artifact.target_keys)), dtype=np.float64)
    for i in range(X.shape[0]):
        row = {k: float(X[i, j]) for j, k in enumerate(artifact.feature_keys)}
        out = model.predict_one(row)
        preds[i, :] = [float(out[k]) for k in artifact.target_keys]
    return preds


def _metrics_by_split(
    Y: np.ndarray,
    Y_pred: np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    def _take(idx: np.ndarray) -> dict[str, float]:
        if idx.size == 0:
            return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
        return regression_metrics(Y[idx], Y_pred[idx])

    return {
        "train": _take(train_idx),
        "val": _take(val_idx),
        "test": _take(test_idx),
    }

# CalculiX Imports
try:
    from larrak2.surrogate.calculix_nn import (
        DEFAULT_FEATURE_KEYS as DEFAULT_CCX_FEATURE_KEYS,
        DEFAULT_TARGET_KEYS as DEFAULT_CCX_TARGET_KEYS,
        load_dataset_json as load_ccx_dataset_json,
        load_dataset_jsonl as load_ccx_dataset_jsonl,
        load_dataset_npz as load_ccx_dataset_npz,
        save_artifact as save_ccx_artifact,
        train_calculix_surrogate,
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


def _resolve_artifact_outdir(raw: str | Path, *, purpose: str) -> Path:
    outdir = assert_not_legacy_models_write(raw, purpose=purpose)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _parse_hidden(raw: str | None) -> tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        return (128, 128)
    vals = tuple(int(tok.strip()) for tok in text.split(",") if tok.strip())
    return vals or (128, 128)


def _infer_objective_names(
    *,
    fidelity: int,
    rpm: float,
    torque: float,
    n_obj: int,
) -> tuple[str, ...]:
    probe = evaluate_candidate(
        mid_bounds_candidate(),
        EvalContext(rpm=float(rpm), torque=float(torque), fidelity=int(fidelity), seed=0),
    )
    names = [str(n) for n in probe.diag.get("objectives", {}).get("names", [])]
    if not names:
        names = [f"objective_{i}" for i in range(n_obj)]
    if len(names) != n_obj:
        names = names[:n_obj] + [f"objective_{i}" for i in range(len(names), n_obj)]
    return tuple(names)


def _load_stack_training_dataset(
    *,
    dataset_path: str,
    pareto_dir: str,
    fidelity: int,
    rpm: float,
    torque: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    if str(dataset_path).strip():
        npz_path = assert_not_legacy_models_path(dataset_path, purpose="stack training dataset")
        if not npz_path.exists():
            raise FileNotFoundError(f"Stack training dataset not found: {npz_path}")

        with np.load(npz_path, allow_pickle=True) as data:
            if "X" not in data or "Y" not in data:
                raise ValueError("Stack dataset NPZ must contain 'X' and 'Y' arrays")
            X = np.asarray(data["X"], dtype=np.float64)
            Y = np.asarray(data["Y"], dtype=np.float64)
            feature_names = (
                tuple(str(v) for v in data["feature_names"].tolist())
                if "feature_names" in data
                else default_feature_names(N_TOTAL)
            )
            objective_names = (
                tuple(str(v) for v in data["objective_names"].tolist())
                if "objective_names" in data
                else _infer_objective_names(
                    fidelity=int(fidelity),
                    rpm=float(rpm),
                    torque=float(torque),
                    n_obj=max(1, Y.shape[1] - len(get_constraint_names(int(fidelity)))),
                )
            )
            constraint_names = (
                tuple(str(v) for v in data["constraint_names"].tolist())
                if "constraint_names" in data
                else tuple(get_constraint_names(int(fidelity)))
            )

        metadata = {
            "source": "npz",
            "path": str(npz_path),
        }
        return X, Y, feature_names, objective_names, constraint_names, metadata

    if str(pareto_dir).strip():
        archive_dir = assert_not_legacy_models_path(pareto_dir, purpose="stack training archive")
        X_design, F, G, summary = load_archive(archive_dir)
        rpm_value = float(summary.get("rpm", rpm))
        torque_value = float(summary.get("torque", torque))
        resolved_fidelity = int(summary.get("fidelity", fidelity))
        if resolved_fidelity != int(fidelity):
            raise ValueError(
                f"Archive fidelity ({resolved_fidelity}) does not match requested fidelity ({fidelity})"
            )

        feature_names = default_feature_names(N_TOTAL)
        X = np.hstack(
            [
                np.asarray(X_design, dtype=np.float64),
                np.full((X_design.shape[0], 1), rpm_value, dtype=np.float64),
                np.full((X_design.shape[0], 1), torque_value, dtype=np.float64),
            ]
        )
        Y = np.hstack([np.asarray(F, dtype=np.float64), np.asarray(G, dtype=np.float64)])
        objective_names = tuple(summary.get("objective_names", []))
        if not objective_names:
            objective_names = _infer_objective_names(
                fidelity=int(fidelity),
                rpm=rpm_value,
                torque=torque_value,
                n_obj=F.shape[1],
            )
        constraint_names = tuple(summary.get("constraint_names", []))
        if not constraint_names:
            constraint_names = tuple(get_constraint_names(int(fidelity)))

        metadata = {
            "source": "pareto_archive",
            "path": str(archive_dir),
            "rpm": rpm_value,
            "torque": torque_value,
            "fidelity": int(fidelity),
            "n_pareto": int(X_design.shape[0]),
        }
        return X, Y, feature_names, objective_names, constraint_names, metadata

    raise ValueError("Provide either --dataset or --pareto-dir for stack surrogate training data")


def train_stack_surrogate_workflow(args: argparse.Namespace) -> dict[str, Any]:
    """Workflow for global stack surrogate training and artifact export."""
    print("Starting stack surrogate training workflow...")
    outdir = _resolve_artifact_outdir(
        str(args.outdir or DEFAULT_STACK_SURROGATE_DIR),
        purpose="Stack surrogate artifact output",
    )
    artifact_name = str(getattr(args, "name", "stack_f1_surrogate.npz")).strip() or "stack_f1_surrogate.npz"
    out_path = outdir / artifact_name

    X, Y, feature_names, objective_names, constraint_names, source_meta = _load_stack_training_dataset(
        dataset_path=str(getattr(args, "dataset", "")),
        pareto_dir=str(getattr(args, "pareto_dir", "")),
        fidelity=int(args.fidelity),
        rpm=float(args.rpm),
        torque=float(args.torque),
    )

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Stack training arrays must be 2D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X/Y row mismatch: {X.shape[0]} vs {Y.shape[0]}")
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Feature schema mismatch: X has {X.shape[1]} cols, names={len(feature_names)}")
    if Y.shape[1] != len(objective_names) + len(constraint_names):
        raise ValueError(
            "Target schema mismatch: "
            f"Y has {Y.shape[1]} cols, expected {len(objective_names) + len(constraint_names)} "
            f"(n_obj={len(objective_names)}, n_constr={len(constraint_names)})"
        )

    artifact, metrics = train_stack_surrogate(
        X,
        Y,
        feature_names=feature_names,
        objective_names=objective_names,
        constraint_names=constraint_names,
        fidelity=int(args.fidelity),
        hidden_layers=_parse_hidden(getattr(args, "hidden", None)),
        activation=str(getattr(args, "activation", "relu")),
        leaky_relu_slope=float(getattr(args, "leaky_relu_slope", 0.01)),
        seed=int(args.seed),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_frac=float(args.val_frac),
    )
    save_stack_artifact(artifact, out_path)

    summary = {
        "artifact_path": str(out_path),
        "fidelity": int(args.fidelity),
        "feature_names": list(feature_names),
        "objective_names": list(objective_names),
        "constraint_names": list(constraint_names),
        "source": source_meta,
        "metrics": metrics,
        "version_hash": artifact.version_hash,
    }
    summary_path = outdir / "stack_surrogate_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _stack_quality_report(
        artifact_path=out_path,
        source_meta=source_meta,
        feature_names=feature_names,
        objective_names=objective_names,
        constraint_names=constraint_names,
        metrics=metrics,
    )
    print(f"Saved stack surrogate artifact to {out_path}")
    print(f"Training summary: {summary_path}")
    return summary


def train_openfoam_workflow(args: argparse.Namespace):
    """Workflow for OpenFOAM NN training."""
    print("Starting OpenFOAM NN Training Workflow...")
    data_path = Path(args.data)
    outdir = _resolve_artifact_outdir(args.outdir, purpose="OpenFOAM surrogate artifact")

    # 1. Load Data
    suf = data_path.suffix.lower()
    if suf == ".json":
        X, Y = load_dataset_json(
            data_path, feature_keys=DEFAULT_FEATURE_KEYS, target_keys=DEFAULT_TARGET_KEYS
        )
    elif suf == ".jsonl":
        X, Y = load_dataset_jsonl(
            data_path, feature_keys=DEFAULT_FEATURE_KEYS, target_keys=DEFAULT_TARGET_KEYS
        )
    elif suf == ".npz":
        X, Y = load_dataset_npz(data_path)
    elif suf == ".parquet":
        import pandas as pd

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
    train_idx, val_idx, test_idx = _split_indices(X.shape[0], seed=int(args.seed))
    Y_pred = _predict_openfoam_batch(artifact, np.asarray(X, dtype=np.float64))
    split_metrics = _metrics_by_split(
        np.asarray(Y, dtype=np.float64),
        Y_pred,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    val_rmse = float(split_metrics["val"]["rmse"])
    passed = bool(np.isfinite(val_rmse))
    report = {
        "schema_version": "surrogate_quality_report_v1",
        "surrogate_kind": "openfoam",
        "artifact_file": out_path.name,
        "artifact_sha256": sha256_file(out_path),
        "dataset_manifest": dataset_manifest_for_file(
            data_path,
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            n_targets=int(Y.shape[1]),
        ),
        "metrics": {
            "train": split_metrics["train"],
            "val": split_metrics["val"],
            "test": split_metrics["test"],
            "slice_metrics": [
                {
                    "name": "rpm_low_band",
                    **regression_metrics(
                        np.asarray(Y)[np.asarray(X)[:, 0] <= np.median(np.asarray(X)[:, 0])],
                        np.asarray(Y_pred)[np.asarray(X)[:, 0] <= np.median(np.asarray(X)[:, 0])],
                    ),
                },
                {
                    "name": "rpm_high_band",
                    **regression_metrics(
                        np.asarray(Y)[np.asarray(X)[:, 0] > np.median(np.asarray(X)[:, 0])],
                        np.asarray(Y_pred)[np.asarray(X)[:, 0] > np.median(np.asarray(X)[:, 0])],
                    ),
                },
            ],
        },
        "ood_thresholds": {"rpm_range": [float(np.min(X[:, 0])), float(np.max(X[:, 0]))]},
        "uncertainty_calibration": {"method": "deterministic_mlp", "status": "not_applicable"},
        "required_artifacts": [out_path.name],
        "pass": passed,
        "fail_reasons": [] if passed else ["non-finite validation RMSE"],
    }
    write_quality_report(out_path.parent / "quality_report.json", report)
    print(f"Saved to {out_path}")


def train_calculix_workflow(args: argparse.Namespace):
    """Workflow for CalculiX NN training."""
    print("Starting CalculiX NN Training Workflow...")
    data_path = Path(args.data)
    outdir = _resolve_artifact_outdir(args.outdir, purpose="CalculiX surrogate artifact")

    suf = data_path.suffix.lower()
    if suf == ".json":
        X, Y = load_ccx_dataset_json(
            data_path,
            feature_keys=DEFAULT_CCX_FEATURE_KEYS,
            target_keys=DEFAULT_CCX_TARGET_KEYS,
        )
    elif suf == ".jsonl":
        X, Y = load_ccx_dataset_jsonl(
            data_path,
            feature_keys=DEFAULT_CCX_FEATURE_KEYS,
            target_keys=DEFAULT_CCX_TARGET_KEYS,
        )
    elif suf == ".npz":
        X, Y = load_ccx_dataset_npz(data_path)
    elif suf == ".parquet":
        import pandas as pd

        df = pd.read_parquet(data_path)
        X = df[list(DEFAULT_CCX_FEATURE_KEYS)].to_numpy(dtype=np.float64)
        Y = df[list(DEFAULT_CCX_TARGET_KEYS)].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported format: {suf}")

    hidden_layers = tuple(int(s) for s in args.hidden.split(",")) if args.hidden else (64, 64)

    artifact = train_calculix_surrogate(
        X,
        Y,
        feature_keys=DEFAULT_CCX_FEATURE_KEYS,
        target_keys=DEFAULT_CCX_TARGET_KEYS,
        hidden_layers=hidden_layers,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=0.2,
    )

    out_path = outdir / args.name
    save_ccx_artifact(artifact, out_path)
    train_idx, val_idx, test_idx = _split_indices(X.shape[0], seed=int(args.seed))
    Y_pred = _predict_calculix_batch(artifact, np.asarray(X, dtype=np.float64))
    split_metrics = _metrics_by_split(
        np.asarray(Y, dtype=np.float64),
        Y_pred,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    val_rmse = float(split_metrics["val"]["rmse"])
    passed = bool(np.isfinite(val_rmse))
    report = {
        "schema_version": "surrogate_quality_report_v1",
        "surrogate_kind": "calculix",
        "artifact_file": out_path.name,
        "artifact_sha256": sha256_file(out_path),
        "dataset_manifest": dataset_manifest_for_file(
            data_path,
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            n_targets=int(Y.shape[1]),
        ),
        "metrics": {
            "train": split_metrics["train"],
            "val": split_metrics["val"],
            "test": split_metrics["test"],
            "slice_metrics": [
                {
                    "name": "torque_low_band",
                    **regression_metrics(
                        np.asarray(Y)[np.asarray(X)[:, 1] <= np.median(np.asarray(X)[:, 1])],
                        np.asarray(Y_pred)[np.asarray(X)[:, 1] <= np.median(np.asarray(X)[:, 1])],
                    ),
                },
                {
                    "name": "torque_high_band",
                    **regression_metrics(
                        np.asarray(Y)[np.asarray(X)[:, 1] > np.median(np.asarray(X)[:, 1])],
                        np.asarray(Y_pred)[np.asarray(X)[:, 1] > np.median(np.asarray(X)[:, 1])],
                    ),
                },
            ],
        },
        "ood_thresholds": {
            "rpm_range": [float(np.min(X[:, 0])), float(np.max(X[:, 0]))],
            "torque_range": [float(np.min(X[:, 1])), float(np.max(X[:, 1]))],
        },
        "uncertainty_calibration": {"method": "deterministic_mlp", "status": "not_applicable"},
        "required_artifacts": [out_path.name],
        "pass": passed,
        "fail_reasons": [] if passed else ["non-finite validation RMSE"],
    }
    write_quality_report(out_path.parent / "quality_report.json", report)
    print(f"Saved to {out_path}")


def train_gear_nn_workflow(args: argparse.Namespace):
    """Workflow for Gear Loss NN training."""
    print("Starting Gear NN Training Workflow...")
    import pandas as pd

    outdir = _resolve_artifact_outdir(args.outdir, purpose="Gear-loss NN artifact")

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

    train_proch_basic(model=model, X=X, y=y, output_dir=str(outdir), epochs=args.epochs, lr=args.lr)
    print("Gear NN training complete.")


def train_scavenge_gbr_workflow(args: argparse.Namespace):
    """Workflow for Scavenge GBR training."""
    print("Starting Scavenge GBR Training Workflow...")
    data_path = Path(args.data)
    outdir = _resolve_artifact_outdir(args.outdir, purpose="Scavenge GBR artifact")

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
    outdir = _resolve_artifact_outdir(args.outdir, purpose="Gear GBR artifact")

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
    outdir = _resolve_artifact_outdir(args.outdir, purpose="Residual surrogate artifact")

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
