"""Anchor benchmark gating checks for fidelity-2 thermo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.core.encoding import decode_candidate, mid_bounds_candidate
from larrak2.core.types import BreathingConfig, EvalContext
from larrak2.thermo import two_zone
from larrak2.thermo.constants import DEFAULT_THERMO_ANCHOR_MANIFEST_PATH, load_anchor_manifest
from larrak2.thermo.validation import ThermoValidationError


def _write_anchor_manifest(path: Path) -> Path:
    payload = {
        "version": "test",
        "validated_envelope": {
            "rpm_min": 1000.0,
            "rpm_max": 7000.0,
            "torque_min": 20.0,
            "torque_max": 400.0,
        },
        "thresholds": {
            "delta_m_air_rel_max": 0.10,
            "delta_residual_abs_max": 0.05,
            "delta_scavenging_abs_max": 0.08,
        },
        "anchors": [{"rpm": 2800, "torque": 140}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _authoritative_openfoam_artifact() -> dict[str, object]:
    return {
        "model_path": "outputs/artifacts/surrogates/openfoam_nn/openfoam_breathing.pt",
        "quality_report_path": "outputs/artifacts/surrogates/openfoam_nn/quality_report.json",
        "quality_report_exists": True,
        "artifact_file": "openfoam_breathing.pt",
        "dataset_manifest": {"source_path": "outputs/openfoam_doe/results_train.jsonl"},
        "data_provenance": {
            "kind": "truth_records",
            "authoritative_for_strict_f2": True,
            "anchor_manifest_path": "data/thermo/anchor_manifest_v1.json",
            "anchor_manifest_version": "thermo_anchor_v1",
            "anchor_count": 3,
            "truth_source_summary": {"source_path": "outputs/openfoam_doe/results_train.jsonl"},
        },
        "strict_f2_eligible": True,
        "benchmark_authority": "truth_like_authoritative",
        "gate_failure_reason": "",
    }


def test_anchor_benchmark_gate_passes_for_small_disagreement(monkeypatch, tmp_path: Path) -> None:
    candidate = decode_candidate(mid_bounds_candidate())
    base_ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=1,
        seed=3,
        breathing=BreathingConfig(
            intake_open_deg=-20.0,
            intake_close_deg=70.0,
            exhaust_open_deg=-80.0,
            exhaust_close_deg=20.0,
            valve_timing_mode="override",
        ),
    )
    base = two_zone.evaluate_two_zone_thermo(candidate.thermo, base_ctx)

    pred = {
        "m_air_trapped": float(base.diag["m_air_trapped"]),
        "residual_fraction": float(base.diag["residual_fraction"]),
        "scavenging_efficiency": float(base.diag["scavenging_efficiency"]),
    }

    def _fake_predict(**_: object) -> dict[str, float]:
        return pred

    monkeypatch.setattr(two_zone, "_predict_openfoam_breathing", _fake_predict)
    monkeypatch.setattr(
        two_zone, "_openfoam_artifact_summary", lambda *_: _authoritative_openfoam_artifact()
    )
    monkeypatch.setattr(
        two_zone,
        "validate_benchmark_agreement",
        lambda **_: (True, "pass", []),
    )

    manifest_path = _write_anchor_manifest(tmp_path / "anchors.json")
    f2_ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=2,
        seed=3,
        breathing=base_ctx.breathing,
        thermo_anchor_manifest_path=str(manifest_path),
    )
    res = two_zone.evaluate_two_zone_thermo(candidate.thermo, f2_ctx)

    assert res.diag["in_validated_envelope"] is True
    assert res.diag["thermo_benchmark_status"] == "pass"
    assert res.diag["thermo_hybrid_correction_active"] is True
    assert res.diag["anchor_manifest_version"] == "test"
    assert res.diag["anchor_count"] == 1
    assert res.diag["anchor_path"] == str(manifest_path)
    assert res.diag["openfoam_benchmark_authority"] == "truth_like_authoritative"
    assert res.diag["openfoam_strict_f2_eligible"] is True


def test_anchor_benchmark_gate_fails_for_large_disagreement(monkeypatch, tmp_path: Path) -> None:
    candidate = decode_candidate(mid_bounds_candidate())

    def _fake_predict(**_: object) -> dict[str, float]:
        return {
            "m_air_trapped": 20.0,  # intentionally outside tolerance
            "residual_fraction": 0.85,
            "scavenging_efficiency": 0.1,
        }

    monkeypatch.setattr(two_zone, "_predict_openfoam_breathing", _fake_predict)
    monkeypatch.setattr(
        two_zone, "_openfoam_artifact_summary", lambda *_: _authoritative_openfoam_artifact()
    )

    manifest_path = _write_anchor_manifest(tmp_path / "anchors_fail.json")
    ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=2,
        seed=3,
        breathing=BreathingConfig(
            intake_open_deg=-20.0,
            intake_close_deg=70.0,
            exhaust_open_deg=-80.0,
            exhaust_close_deg=20.0,
            valve_timing_mode="override",
        ),
        thermo_anchor_manifest_path=str(manifest_path),
    )
    with pytest.raises(ThermoValidationError, match="benchmark_status=failed") as excinfo:
        two_zone.evaluate_two_zone_thermo(candidate.thermo, ctx)

    payload = excinfo.value.payload
    assert payload["failure_stage"] == "thermo_validation"
    assert payload["validation"]["benchmark_status"] == "failed"
    assert payload["validation"]["nn_disagreement"]["delta_m_air"] > 0.1
    assert payload["eq_breathing"]["m_air_trapped"] > 0.0
    assert payload["nn_breathing"]["m_air_trapped"] == pytest.approx(20.0)
    assert payload["anchor_manifest"]["path"] == str(manifest_path)


def test_fidelity2_strict_rejects_synthetic_openfoam_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    candidate = decode_candidate(mid_bounds_candidate())
    manifest_path = _write_anchor_manifest(tmp_path / "anchors_strict.json")
    artifact_path = tmp_path / "openfoam_breathing.pt"
    artifact_path.write_bytes(b"synthetic")
    monkeypatch.setattr(two_zone, "_resolve_openfoam_surrogate_path", lambda *_: artifact_path)
    monkeypatch.setattr(
        two_zone,
        "_openfoam_artifact_summary",
        lambda *_: {
            "model_path": str(artifact_path),
            "quality_report_path": str(tmp_path / "quality_report.json"),
            "quality_report_exists": True,
            "artifact_file": artifact_path.name,
            "dataset_manifest": {
                "source_path": "outputs/dress_rehearsal/synthetic_openfoam_training.npz"
            },
            "data_provenance": {
                "kind": "synthetic_rehearsal",
                "authoritative_for_strict_f2": False,
                "anchor_manifest_path": "",
                "anchor_manifest_version": "",
                "anchor_count": 0,
                "truth_source_summary": {
                    "source_path": "outputs/dress_rehearsal/synthetic_openfoam_training.npz"
                },
            },
            "strict_f2_eligible": False,
            "benchmark_authority": "synthetic_non_authoritative",
            "gate_failure_reason": "synthetic_artifact_not_allowed_in_strict_f2",
        },
    )
    ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=2,
        seed=3,
        breathing=BreathingConfig(
            intake_open_deg=-20.0,
            intake_close_deg=70.0,
            exhaust_open_deg=-80.0,
            exhaust_close_deg=20.0,
            valve_timing_mode="override",
        ),
        thermo_anchor_manifest_path=str(manifest_path),
        surrogate_validation_mode="strict",
    )
    with pytest.raises(ThermoValidationError, match="OpenFOAM strict F2 provenance gate failed"):
        two_zone.evaluate_two_zone_thermo(candidate.thermo, ctx)


def test_fidelity2_warn_allows_synthetic_openfoam_provenance(monkeypatch, tmp_path: Path) -> None:
    candidate = decode_candidate(mid_bounds_candidate())
    base_ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=1,
        seed=3,
        breathing=BreathingConfig(
            intake_open_deg=-20.0,
            intake_close_deg=70.0,
            exhaust_open_deg=-80.0,
            exhaust_close_deg=20.0,
            valve_timing_mode="override",
        ),
    )
    base = two_zone.evaluate_two_zone_thermo(candidate.thermo, base_ctx)

    monkeypatch.setattr(
        two_zone,
        "_openfoam_artifact_summary",
        lambda *_: {
            "model_path": "outputs/artifacts/surrogates/openfoam_nn/openfoam_breathing.pt",
            "quality_report_path": "outputs/artifacts/surrogates/openfoam_nn/quality_report.json",
            "quality_report_exists": True,
            "artifact_file": "openfoam_breathing.pt",
            "dataset_manifest": {
                "source_path": "outputs/dress_rehearsal/synthetic_openfoam_training.npz"
            },
            "data_provenance": {
                "kind": "synthetic_rehearsal",
                "authoritative_for_strict_f2": False,
                "anchor_manifest_path": "",
                "anchor_manifest_version": "",
                "anchor_count": 0,
                "truth_source_summary": {
                    "source_path": "outputs/dress_rehearsal/synthetic_openfoam_training.npz"
                },
            },
            "strict_f2_eligible": False,
            "benchmark_authority": "synthetic_non_authoritative",
            "gate_failure_reason": "synthetic_artifact_not_allowed_in_strict_f2",
        },
    )
    monkeypatch.setattr(
        two_zone,
        "_predict_openfoam_breathing",
        lambda **_: {
            "m_air_trapped": float(base.diag["m_air_trapped"]),
            "residual_fraction": float(base.diag["residual_fraction"]),
            "scavenging_efficiency": float(base.diag["scavenging_efficiency"]),
        },
    )
    monkeypatch.setattr(two_zone, "validate_benchmark_agreement", lambda **_: (True, "pass", []))
    manifest_path = _write_anchor_manifest(tmp_path / "anchors_warn.json")
    ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=2,
        seed=3,
        breathing=base_ctx.breathing,
        thermo_anchor_manifest_path=str(manifest_path),
        surrogate_validation_mode="warn",
    )
    res = two_zone.evaluate_two_zone_thermo(candidate.thermo, ctx)
    assert res.diag["openfoam_benchmark_authority"] == "synthetic_non_authoritative"
    assert res.diag["openfoam_strict_f2_eligible"] is False
    assert res.diag["openfoam_gate_failure_reason"] == "synthetic_artifact_not_allowed_in_strict_f2"


def test_fidelity2_requires_nonempty_anchors_in_strict_mode(monkeypatch) -> None:
    candidate = decode_candidate(mid_bounds_candidate())

    def _fake_predict(**_: object) -> dict[str, float]:
        return {
            "m_air_trapped": 0.001,
            "residual_fraction": 0.05,
            "scavenging_efficiency": 0.9,
        }

    monkeypatch.setattr(two_zone, "_predict_openfoam_breathing", _fake_predict)
    monkeypatch.setattr(
        two_zone,
        "load_validation_manifest",
        lambda _: {
            "version": "test",
            "validated_envelope": {
                "rpm_min": 0.0,
                "rpm_max": 1e9,
                "torque_min": 0.0,
                "torque_max": 1e9,
            },
            "thresholds": {},
            "anchors": [],
        },
    )
    ctx = EvalContext(
        rpm=2800.0, torque=140.0, fidelity=2, seed=3, surrogate_validation_mode="strict"
    )
    with pytest.raises(RuntimeError, match="anchor_count=0"):
        two_zone.evaluate_two_zone_thermo(candidate.thermo, ctx)


def test_two_zone_uses_breathing_compression_ratio() -> None:
    candidate = decode_candidate(mid_bounds_candidate())
    ctx = EvalContext(
        rpm=2400.0,
        torque=120.0,
        fidelity=1,
        seed=9,
        breathing=BreathingConfig(compression_ratio=12.5),
    )
    res = two_zone.evaluate_two_zone_thermo(candidate.thermo, ctx)
    V = res.diag["V"]
    ratio = float(max(V) / max(min(V), 1e-12))
    assert ratio == pytest.approx(12.5, rel=0.15)


def test_default_anchor_manifest_is_non_empty_and_valid() -> None:
    manifest = load_anchor_manifest(DEFAULT_THERMO_ANCHOR_MANIFEST_PATH)
    anchors = manifest.get("anchors", [])
    assert isinstance(anchors, list)
    assert len(anchors) > 0
    for rec in anchors:
        assert float(rec["rpm"]) > 0.0
        assert float(rec["torque"]) >= 0.0
