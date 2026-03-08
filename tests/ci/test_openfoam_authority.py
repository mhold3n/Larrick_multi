"""Authority staging/promotion tests for OpenFOAM artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.surrogate.openfoam_authority import (
    inspect_truth_anchor_manifest,
    promote_openfoam_artifact,
    validate_staged_openfoam_authority_bundle,
    write_staged_openfoam_authority_bundle,
)
from larrak2.surrogate.quality_contract import (
    openfoam_default_data_provenance,
    openfoam_quality_profile,
    sha256_file,
    write_quality_report,
)


def _write_truth_anchor_manifest(path: Path) -> Path:
    payload = {
        "version": "thermo_anchor_v1",
        "validated_envelope": {
            "rpm_min": 1000.0,
            "rpm_max": 7000.0,
            "torque_min": 40.0,
            "torque_max": 400.0,
        },
        "thresholds": {
            "delta_m_air_rel_max": 0.10,
            "delta_residual_abs_max": 0.05,
            "delta_scavenging_abs_max": 0.08,
        },
        "provenance": {
            "generated_by": "tools/build_thermo_anchor_manifest.py",
            "source_type": "truth_runs",
            "input_files": ["outputs/orchestration/truth_records.jsonl"],
        },
        "anchors": [{"rpm": 2000.0, "torque": 80.0, "source": "truth_runs"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_openfoam_quality_report(
    artifact_path: Path,
    *,
    provenance: dict,
    val_nrmse: float = 0.10,
    test_nrmse: float = 0.12,
    val_r2: float = 0.8,
    test_r2: float = 0.78,
) -> Path:
    report = {
        "schema_version": "surrogate_quality_report_v1",
        "surrogate_kind": "openfoam",
        "artifact_file": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "dataset_manifest": {
            "source_path": "outputs/openfoam_doe/results_train.jsonl",
            "num_samples": 12,
            "num_features": 14,
            "num_targets": 4,
            "dataset_sha256": "abc123",
        },
        "metrics": {
            "train": {
                "rmse": 0.05,
                "mae": 0.04,
                "r2": 0.9,
                "per_target": [
                    {
                        "name": "m_air_trapped",
                        "rmse": 0.05,
                        "mae": 0.04,
                        "r2": 0.9,
                        "nrmse": 0.09,
                        "target_scale": 1.0,
                    }
                ],
            },
            "val": {
                "rmse": 0.07,
                "mae": 0.05,
                "r2": val_r2,
                "per_target": [
                    {
                        "name": "m_air_trapped",
                        "rmse": 0.07,
                        "mae": 0.05,
                        "r2": val_r2,
                        "nrmse": val_nrmse,
                        "target_scale": 1.0,
                    }
                ],
            },
            "test": {
                "rmse": 0.08,
                "mae": 0.06,
                "r2": test_r2,
                "per_target": [
                    {
                        "name": "m_air_trapped",
                        "rmse": 0.08,
                        "mae": 0.06,
                        "r2": test_r2,
                        "nrmse": test_nrmse,
                        "target_scale": 1.0,
                    }
                ],
            },
            "slice_metrics": [
                {"name": "rpm_low_band", "rmse": 0.08, "mae": 0.06, "r2": 0.7},
                {"name": "rpm_high_band", "rmse": 0.09, "mae": 0.07, "r2": 0.72},
            ],
        },
        "quality_profile": openfoam_quality_profile(),
        "ood_thresholds": {"rpm_range": [1000.0, 3000.0]},
        "uncertainty_calibration": {"method": "deterministic_mlp", "status": "not_applicable"},
        "required_artifacts": [artifact_path.name],
        "pass": True,
        "fail_reasons": [],
        "data_provenance": provenance,
    }
    report_path = artifact_path.parent / "quality_report.json"
    write_quality_report(report_path, report)
    return report_path


def test_inspect_truth_anchor_manifest_marks_truth_runs(tmp_path: Path) -> None:
    manifest_path = _write_truth_anchor_manifest(tmp_path / "anchor_manifest.json")
    status = inspect_truth_anchor_manifest(manifest_path)
    assert status["truth_backed"] is True
    assert status["anchor_count"] == 1
    assert status["source_type"] == "truth_runs"


def test_validate_staged_bundle_rejects_synthetic_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact" / "openfoam_breathing.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"synthetic")
    report_path = _write_openfoam_quality_report(
        artifact,
        provenance=openfoam_default_data_provenance(
            source_path="outputs/dress_rehearsal/synthetic_openfoam_training.npz",
            kind="synthetic_rehearsal",
            authoritative_for_strict_f2=False,
        ),
    )
    staged = write_staged_openfoam_authority_bundle(
        artifact_path=artifact,
        quality_report_path=report_path,
        bundle_root=tmp_path / "authority",
        data_path=tmp_path / "synthetic.npz",
        source_meta={"source": "provided"},
        template_path=None,
        split_manifest={
            "seed": 1,
            "train_count": 8,
            "val_count": 2,
            "test_count": 2,
            "train_indices": list(range(8)),
            "val_indices": [8, 9],
            "test_indices": [10, 11],
        },
        run_id="synthetic_bundle",
    )
    report = validate_staged_openfoam_authority_bundle(staged["staged_dir"])
    assert report["promotable"] is False
    assert any("dataset_not_doe_generated" == reason for reason in report["failure_reasons"])
    assert any(
        "provenance:synthetic_artifact_not_allowed_in_strict_f2" == reason
        for reason in report["failure_reasons"]
    )


def test_promote_openfoam_artifact_accepts_truth_backed_doe_bundle(tmp_path: Path) -> None:
    anchor_manifest = _write_truth_anchor_manifest(tmp_path / "anchor_manifest.json")
    template_dir = tmp_path / "openfoam_template"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "system").mkdir(parents=True, exist_ok=True)
    (template_dir / "system" / "controlDict").write_text(
        "application rhoPimpleFoam;\n", encoding="utf-8"
    )
    data_path = tmp_path / "results_train.jsonl"
    data_path.write_text('{"ok": true}\n', encoding="utf-8")
    artifact = tmp_path / "artifact" / "openfoam_breathing.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"real-doe-artifact")
    report_path = _write_openfoam_quality_report(
        artifact,
        provenance=openfoam_default_data_provenance(
            source_path=str(data_path),
            kind="doe_generated",
            authoritative_for_strict_f2=True,
            anchor_manifest_path=str(anchor_manifest),
            anchor_manifest_version="thermo_anchor_v1",
            anchor_count=1,
            truth_source_summary={
                "source_path": str(data_path),
                "anchor_source_type": "truth_runs",
            },
        ),
    )
    staged = write_staged_openfoam_authority_bundle(
        artifact_path=artifact,
        quality_report_path=report_path,
        bundle_root=tmp_path / "authority",
        data_path=data_path,
        source_meta={"source": "doe_generated", "n_total_cases": 12, "n_success_cases": 10},
        template_path=template_dir,
        split_manifest={
            "seed": 3,
            "train_count": 8,
            "val_count": 2,
            "test_count": 2,
            "train_indices": list(range(8)),
            "val_indices": [8, 9],
            "test_indices": [10, 11],
        },
        run_id="real_bundle",
    )

    canonical_dir = tmp_path / "canonical_openfoam"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    (canonical_dir / "openfoam_breathing.pt").write_bytes(b"old-canonical")
    old_report = canonical_dir / "quality_report.json"
    old_report.write_text("{}", encoding="utf-8")

    result = promote_openfoam_artifact(
        staged_dir=staged["staged_dir"],
        canonical_dir=canonical_dir,
        backup_root=tmp_path / "archive",
    )
    assert Path(result["canonical_artifact"]).exists()
    assert Path(result["canonical_quality_report"]).exists()
    assert Path(result["backup_path"]).exists()
    assert Path(result["authority_validation_report"]).exists()


def test_promote_openfoam_artifact_rejects_non_truth_anchor_bundle(tmp_path: Path) -> None:
    anchor_manifest = tmp_path / "anchor_manifest.json"
    anchor_manifest.write_text(
        json.dumps(
            {
                "version": "thermo_anchor_v1",
                "provenance": {"source_type": "canonical_default", "input_files": []},
                "anchors": [{"rpm": 2000.0, "torque": 80.0}],
            }
        ),
        encoding="utf-8",
    )
    template_dir = tmp_path / "openfoam_template"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "system").mkdir(parents=True, exist_ok=True)
    (template_dir / "system" / "controlDict").write_text(
        "application rhoPimpleFoam;\n", encoding="utf-8"
    )
    data_path = tmp_path / "results_train.jsonl"
    data_path.write_text('{"ok": true}\n', encoding="utf-8")
    artifact = tmp_path / "artifact" / "openfoam_breathing.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"real-doe-artifact")
    report_path = _write_openfoam_quality_report(
        artifact,
        provenance=openfoam_default_data_provenance(
            source_path=str(data_path),
            kind="doe_generated",
            authoritative_for_strict_f2=True,
            anchor_manifest_path=str(anchor_manifest),
            anchor_manifest_version="thermo_anchor_v1",
            anchor_count=1,
        ),
    )
    staged = write_staged_openfoam_authority_bundle(
        artifact_path=artifact,
        quality_report_path=report_path,
        bundle_root=tmp_path / "authority",
        data_path=data_path,
        source_meta={"source": "doe_generated", "n_total_cases": 12, "n_success_cases": 10},
        template_path=template_dir,
        split_manifest={
            "seed": 3,
            "train_count": 8,
            "val_count": 2,
            "test_count": 2,
            "train_indices": list(range(8)),
            "val_indices": [8, 9],
            "test_indices": [10, 11],
        },
        run_id="non_truth_bundle",
    )

    with pytest.raises(ValueError, match="anchor:anchor_manifest_not_truth_backed"):
        promote_openfoam_artifact(
            staged_dir=staged["staged_dir"],
            canonical_dir=tmp_path / "canonical",
            backup_root=tmp_path / "archive",
        )
