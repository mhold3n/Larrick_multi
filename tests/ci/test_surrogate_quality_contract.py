"""Surrogate quality-report contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.surrogate.quality_contract import (
    thermo_symbolic_quality_fail_reasons,
    validate_artifact_quality,
    validate_quality_report_schema,
    validate_thermo_symbolic_quality,
    write_quality_report,
)


def _base_report(kind: str, artifact_file: str) -> dict:
    report = {
        "schema_version": "surrogate_quality_report_v1",
        "surrogate_kind": kind,
        "artifact_file": artifact_file,
        "artifact_sha256": "",
        "dataset_manifest": {
            "source_path": "data.npz",
            "num_samples": 10,
            "num_features": 3,
            "num_targets": 1,
            "dataset_sha256": "",
        },
        "metrics": {
            "train": {
                "rmse": 0.1,
                "mae": 0.08,
                "r2": 0.9,
                "per_target": [
                    {
                        "name": "target0",
                        "rmse": 0.1,
                        "mae": 0.08,
                        "r2": 0.9,
                        "nrmse": 0.1,
                        "target_scale": 1.0,
                    }
                ],
            },
            "val": {
                "rmse": 0.2,
                "mae": 0.12,
                "r2": 0.8,
                "per_target": [
                    {
                        "name": "target0",
                        "rmse": 0.2,
                        "mae": 0.12,
                        "r2": 0.8,
                        "nrmse": 0.2,
                        "target_scale": 1.0,
                    }
                ],
            },
            "test": {
                "rmse": 0.2,
                "mae": 0.12,
                "r2": 0.8,
                "per_target": [
                    {
                        "name": "target0",
                        "rmse": 0.2,
                        "mae": 0.12,
                        "r2": 0.8,
                        "nrmse": 0.2,
                        "target_scale": 1.0,
                    }
                ],
            },
            "slice_metrics": [],
        },
        "ood_thresholds": {},
        "uncertainty_calibration": {},
        "required_artifacts": [artifact_file],
        "pass": True,
        "fail_reasons": [],
    }
    if kind == "thermo_symbolic":
        report["quality_profile"] = {
            "profile": "balanced_v1",
            "normalization_method": "p95_p05_range",
            "val_nrmse_max": 0.2,
            "test_nrmse_max": 0.25,
            "min_r2": 0.4,
        }
    return report


def test_quality_report_schema_rejects_missing_fields() -> None:
    with pytest.raises(ValueError):
        validate_quality_report_schema({"schema_version": "v1"})


def test_validate_artifact_quality_missing_report_fails_strict(tmp_path: Path) -> None:
    artifact = tmp_path / "openfoam_breathing.pt"
    artifact.write_bytes(b"not-a-real-model")
    with pytest.raises(FileNotFoundError):
        validate_artifact_quality(
            artifact,
            surrogate_kind="openfoam",
            validation_mode="strict",
            required_artifacts=[artifact.name],
        )


def test_validate_artifact_quality_missing_report_for_missing_dir_points_to_dir_report(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "hifi"
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_artifact_quality(
            missing_dir,
            surrogate_kind="hifi",
            validation_mode="strict",
            required_artifacts=["thermal_surrogate.pt"],
        )
    assert str(missing_dir / "quality_report.json") in str(excinfo.value)


def test_validate_artifact_quality_passes_with_valid_report(tmp_path: Path) -> None:
    artifact = tmp_path / "stack.npz"
    artifact.write_bytes(b"artifact-bytes")
    report = _base_report("stack", artifact.name)
    write_quality_report(tmp_path / "quality_report.json", report)
    loaded = validate_artifact_quality(
        artifact,
        surrogate_kind="stack",
        validation_mode="strict",
        required_artifacts=[artifact.name],
    )
    assert loaded is not None
    assert loaded["surrogate_kind"] == "stack"


def test_validate_artifact_quality_blocks_failed_report(tmp_path: Path) -> None:
    artifact = tmp_path / "calculix_stress.pt"
    artifact.write_bytes(b"artifact")
    report = _base_report("calculix", artifact.name)
    report["pass"] = False
    report["fail_reasons"] = ["threshold_exceeded"]
    (tmp_path / "quality_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_artifact_quality(
            artifact,
            surrogate_kind="calculix",
            validation_mode="strict",
            required_artifacts=[artifact.name],
        )


def test_validate_artifact_quality_checks_hash_linkage(tmp_path: Path) -> None:
    artifact = tmp_path / "stack_model.npz"
    artifact.write_bytes(b"abc")
    report = _base_report("stack", artifact.name)
    report["artifact_sha256"] = "0" * 64
    write_quality_report(tmp_path / "quality_report.json", report)
    with pytest.raises(ValueError):
        validate_artifact_quality(
            artifact,
            surrogate_kind="stack",
            validation_mode="strict",
            required_artifacts=[artifact.name],
        )


def test_validate_artifact_quality_merges_report_and_explicit_required_artifacts(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "hifi"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "thermal_surrogate.pt").write_bytes(b"t")
    (model_dir / "structural_surrogate.pt").write_bytes(b"s")
    (model_dir / "flow_surrogate.pt").write_bytes(b"f")
    (model_dir / "normalization.json").write_text("{}", encoding="utf-8")
    extra_report_required = model_dir / "contract_bundle.json"
    extra_report_required.write_text("{}", encoding="utf-8")
    extra_abs_required = tmp_path / "external_gate.md"
    extra_abs_required.write_text("# ok\n", encoding="utf-8")

    report = _base_report("hifi", "")
    report["required_artifacts"] = [
        "thermal_surrogate.pt",
        "structural_surrogate.pt",
        "flow_surrogate.pt",
        "normalization.json",
        "contract_bundle.json",
    ]
    write_quality_report(model_dir / "quality_report.json", report)

    loaded = validate_artifact_quality(
        model_dir,
        surrogate_kind="hifi",
        validation_mode="strict",
        required_artifacts=[str(extra_abs_required)],
    )
    assert loaded is not None
    assert loaded["surrogate_kind"] == "hifi"


def test_quality_report_schema_accepts_thermo_symbolic_kind() -> None:
    report = _base_report("thermo_symbolic", "thermo_symbolic_f1.npz")
    validate_quality_report_schema(report)


def test_thermo_symbolic_quality_gate_strict_rejects_threshold_violation() -> None:
    report = _base_report("thermo_symbolic", "thermo_symbolic_f1.npz")
    report["metrics"]["val"]["per_target"][0]["nrmse"] = 0.5
    report["metrics"]["test"]["per_target"][0]["r2"] = 0.1
    reasons = thermo_symbolic_quality_fail_reasons(report)
    assert reasons
    with pytest.raises(ValueError):
        validate_thermo_symbolic_quality(report, validation_mode="strict")


def test_thermo_symbolic_quality_gate_warn_allows_degraded_report() -> None:
    report = _base_report("thermo_symbolic", "thermo_symbolic_f1.npz")
    report["metrics"]["val"]["per_target"][0]["nrmse"] = 0.35
    reasons = validate_thermo_symbolic_quality(report, validation_mode="warn")
    assert reasons
