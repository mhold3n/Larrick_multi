"""Anchor benchmark gating checks for fidelity-2 thermo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.core.encoding import decode_candidate, mid_bounds_candidate
from larrak2.core.types import BreathingConfig, EvalContext
from larrak2.thermo import two_zone


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


def test_anchor_benchmark_gate_passes_for_small_disagreement(monkeypatch, tmp_path: Path) -> None:
    candidate = decode_candidate(mid_bounds_candidate())
    base_ctx = EvalContext(rpm=2800.0, torque=140.0, fidelity=1, seed=3)
    base = two_zone.evaluate_two_zone_thermo(candidate.thermo, base_ctx)

    pred = {
        "m_air_trapped": float(base.diag["m_air_trapped"]),
        "residual_fraction": float(base.diag["residual_fraction"]),
        "scavenging_efficiency": float(base.diag["scavenging_efficiency"]),
    }

    def _fake_predict(**_: object) -> dict[str, float]:
        return pred

    monkeypatch.setattr(two_zone, "_predict_openfoam_breathing", _fake_predict)

    manifest_path = _write_anchor_manifest(tmp_path / "anchors.json")
    f2_ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=2,
        seed=3,
        thermo_anchor_manifest_path=str(manifest_path),
    )
    res = two_zone.evaluate_two_zone_thermo(candidate.thermo, f2_ctx)

    assert res.diag["in_validated_envelope"] is True
    assert res.diag["thermo_benchmark_status"] == "pass"
    assert res.diag["thermo_hybrid_correction_active"] is True


def test_anchor_benchmark_gate_fails_for_large_disagreement(monkeypatch, tmp_path: Path) -> None:
    candidate = decode_candidate(mid_bounds_candidate())

    def _fake_predict(**_: object) -> dict[str, float]:
        return {
            "m_air_trapped": 20.0,  # intentionally outside tolerance
            "residual_fraction": 0.85,
            "scavenging_efficiency": 0.1,
        }

    monkeypatch.setattr(two_zone, "_predict_openfoam_breathing", _fake_predict)

    manifest_path = _write_anchor_manifest(tmp_path / "anchors_fail.json")
    ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=2,
        seed=3,
        thermo_anchor_manifest_path=str(manifest_path),
    )
    with pytest.raises(RuntimeError, match="benchmark_status=failed"):
        two_zone.evaluate_two_zone_thermo(candidate.thermo, ctx)


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
    ctx = EvalContext(rpm=2800.0, torque=140.0, fidelity=2, seed=3, surrogate_validation_mode="strict")
    with pytest.raises(RuntimeError, match="non-empty anchor manifest"):
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
