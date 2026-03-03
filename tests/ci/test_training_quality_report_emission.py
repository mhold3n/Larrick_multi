"""Training workflow quality-report emission tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from larrak2.surrogate.calculix_nn import DEFAULT_FEATURE_KEYS as CCX_FEATURES
from larrak2.surrogate.calculix_nn import DEFAULT_TARGET_KEYS as CCX_TARGETS
from larrak2.surrogate.openfoam_nn import DEFAULT_FEATURE_KEYS as OF_FEATURES
from larrak2.surrogate.openfoam_nn import DEFAULT_TARGET_KEYS as OF_TARGETS
from larrak2.training.workflows import (
    train_calculix_workflow,
    train_openfoam_workflow,
    train_stack_surrogate_workflow,
)


def test_train_openfoam_emits_quality_report(tmp_path: Path) -> None:
    n = 16
    X = np.random.default_rng(0).normal(size=(n, len(OF_FEATURES))).astype(np.float64)
    Y = np.random.default_rng(1).normal(size=(n, len(OF_TARGETS))).astype(np.float64)
    data_path = tmp_path / "openfoam_data.npz"
    np.savez(data_path, X=X, Y=Y)

    outdir = tmp_path / "openfoam_out"
    args = argparse.Namespace(
        data=str(data_path),
        outdir=str(outdir),
        seed=7,
        epochs=5,
        lr=1e-3,
        hidden="16,16",
        weight_decay=0.0,
        name="openfoam_breathing.pt",
    )
    train_openfoam_workflow(args)

    report = json.loads((outdir / "quality_report.json").read_text(encoding="utf-8"))
    assert report["surrogate_kind"] == "openfoam"
    assert report["metrics"]["train"]
    assert report["metrics"]["val"]
    assert report["metrics"]["test"]


def test_train_calculix_emits_quality_report(tmp_path: Path) -> None:
    n = 16
    X = np.random.default_rng(2).normal(size=(n, len(CCX_FEATURES))).astype(np.float64)
    Y = np.random.default_rng(3).normal(size=(n, len(CCX_TARGETS))).astype(np.float64)
    data_path = tmp_path / "calculix_data.npz"
    np.savez(data_path, X=X, Y=Y)

    outdir = tmp_path / "calculix_out"
    args = argparse.Namespace(
        data=str(data_path),
        outdir=str(outdir),
        seed=9,
        epochs=5,
        lr=1e-3,
        hidden="16,16",
        weight_decay=0.0,
        name="calculix_stress.pt",
    )
    train_calculix_workflow(args)

    report = json.loads((outdir / "quality_report.json").read_text(encoding="utf-8"))
    assert report["surrogate_kind"] == "calculix"
    assert report["metrics"]["train"]
    assert report["metrics"]["val"]
    assert report["metrics"]["test"]


def test_train_stack_emits_quality_report(tmp_path: Path) -> None:
    n = 12
    x_cols = 12
    y_cols = 4
    X = np.random.default_rng(4).normal(size=(n, x_cols)).astype(np.float64)
    Y = np.random.default_rng(5).normal(size=(n, y_cols)).astype(np.float64)
    data_path = tmp_path / "stack_data.npz"
    np.savez(
        data_path,
        X=X,
        Y=Y,
        feature_names=np.array([f"x_{i:03d}" for i in range(x_cols)], dtype=object),
        objective_names=np.array([f"obj{i}" for i in range(3)], dtype=object),
        constraint_names=np.array([f"g{i}" for i in range(1)], dtype=object),
    )

    outdir = tmp_path / "stack_out"
    args = argparse.Namespace(
        outdir=str(outdir),
        name="stack_f1_surrogate.npz",
        dataset=str(data_path),
        pareto_dir="",
        fidelity=1,
        rpm=2500.0,
        torque=150.0,
        hidden="16,16",
        activation="relu",
        leaky_relu_slope=0.01,
        epochs=5,
        lr=1e-3,
        weight_decay=1e-6,
        val_frac=0.2,
        seed=11,
    )
    train_stack_surrogate_workflow(args)

    report = json.loads((outdir / "quality_report.json").read_text(encoding="utf-8"))
    assert report["surrogate_kind"] == "stack"
    assert report["metrics"]["train"]
    assert report["metrics"]["val"]
    assert report["metrics"]["test"]
