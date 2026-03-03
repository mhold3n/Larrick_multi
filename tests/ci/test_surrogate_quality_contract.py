"""Surrogate quality-report contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.surrogate.quality_contract import (
    validate_artifact_quality,
    validate_quality_report_schema,
    write_quality_report,
)


def _base_report(kind: str, artifact_file: str) -> dict:
    return {
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
            "train": {"rmse": 0.1},
            "val": {"rmse": 0.2},
            "test": {"rmse": 0.2},
            "slice_metrics": [],
        },
        "ood_thresholds": {},
        "uncertainty_calibration": {},
        "required_artifacts": [artifact_file],
        "pass": True,
        "fail_reasons": [],
    }


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
