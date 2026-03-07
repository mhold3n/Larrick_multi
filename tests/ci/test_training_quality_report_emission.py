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
    train_thermo_symbolic_workflow,
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


def test_train_thermo_symbolic_emits_quality_report(tmp_path: Path) -> None:
    n = 20
    x_cols = 12
    y_cols = 5
    X = np.random.default_rng(6).normal(size=(n, x_cols)).astype(np.float64)
    Y = np.random.default_rng(7).normal(size=(n, y_cols)).astype(np.float64)
    data_path = tmp_path / "thermo_symbolic_data.npz"
    np.savez(
        data_path,
        X=X,
        Y=Y,
        feature_names=np.array([f"x_{i:03d}" for i in range(10)] + ["rpm", "torque"], dtype=object),
        objective_names=np.array(
            ["eta_comb_gap", "eta_exp_gap", "motion_law_penalty"], dtype=object
        ),
        constraint_names=np.array(["thermo_power_balance", "thermo_pressure_limit"], dtype=object),
    )

    outdir = tmp_path / "thermo_symbolic_out"
    args = argparse.Namespace(
        outdir=str(outdir),
        name="thermo_symbolic_f1.npz",
        dataset=str(data_path),
        dataset_out="",
        n_samples=64,
        fidelity=1,
        rpm=2600.0,
        torque=130.0,
        objective_names="eta_comb_gap,eta_exp_gap,motion_law_penalty",
        constraint_names="thermo_power_balance,thermo_pressure_limit",
        val_frac=0.2,
        seed=17,
        thermo_model="two_zone_eq_v1",
        thermo_constants_path="",
        thermo_anchor_manifest="",
        surrogate_validation_mode="strict",
    )
    train_thermo_symbolic_workflow(args)

    report = json.loads((outdir / "quality_report.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (outdir / "thermo_symbolic_training_summary.json").read_text(encoding="utf-8")
    )
    assert report["surrogate_kind"] == "thermo_symbolic"
    assert report["quality_profile"]["normalization_method"] == "p95_p05_range"
    assert report["metrics"]["train"]
    assert report["metrics"]["val"]
    assert report["metrics"]["test"]
    assert report["metrics"]["val"]["per_target"]
    assert report["metrics"]["test"]["per_target"]
    assert Path(summary["artifact_path"]).exists()


def test_train_stack_defaults_follow_fidelity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    n = 12
    x_cols = 12
    y_cols = 4
    data_path = tmp_path / "stack_data.npz"
    np.savez(
        data_path,
        X=np.random.default_rng(8).normal(size=(n, x_cols)).astype(np.float64),
        Y=np.random.default_rng(9).normal(size=(n, y_cols)).astype(np.float64),
        feature_names=np.array([f"x_{i:03d}" for i in range(x_cols)], dtype=object),
        objective_names=np.array([f"obj{i}" for i in range(3)], dtype=object),
        constraint_names=np.array([f"g{i}" for i in range(1)], dtype=object),
    )

    args = argparse.Namespace(
        outdir="",
        name="",
        dataset=str(data_path),
        pareto_dir="",
        fidelity=2,
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
    summary = train_stack_surrogate_workflow(args)
    expected = tmp_path / "outputs/artifacts/surrogates/stack_f2/stack_f2_surrogate.npz"
    assert Path(summary["artifact_path"]).resolve() == expected.resolve()
    assert expected.exists()


def test_train_thermo_defaults_follow_fidelity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    n = 20
    x_cols = 12
    y_cols = 5
    data_path = tmp_path / "thermo_symbolic_data.npz"
    np.savez(
        data_path,
        X=np.random.default_rng(10).normal(size=(n, x_cols)).astype(np.float64),
        Y=np.random.default_rng(11).normal(size=(n, y_cols)).astype(np.float64),
        feature_names=np.array([f"x_{i:03d}" for i in range(10)] + ["rpm", "torque"], dtype=object),
        objective_names=np.array(
            ["eta_comb_gap", "eta_exp_gap", "motion_law_penalty"], dtype=object
        ),
        constraint_names=np.array(["thermo_power_balance", "thermo_pressure_limit"], dtype=object),
    )

    args = argparse.Namespace(
        outdir="",
        name="",
        dataset=str(data_path),
        dataset_out="",
        n_samples=64,
        fidelity=2,
        rpm=2600.0,
        torque=130.0,
        objective_names="eta_comb_gap,eta_exp_gap,motion_law_penalty",
        constraint_names="thermo_power_balance,thermo_pressure_limit",
        val_frac=0.2,
        seed=17,
        thermo_model="two_zone_eq_v1",
        thermo_constants_path="",
        thermo_anchor_manifest="",
        surrogate_validation_mode="strict",
    )
    summary = train_thermo_symbolic_workflow(args)
    expected = tmp_path / "outputs/artifacts/surrogates/thermo_symbolic_f2/thermo_symbolic_f2.npz"
    assert Path(summary["artifact_path"]).resolve() == expected.resolve()
    assert expected.exists()
