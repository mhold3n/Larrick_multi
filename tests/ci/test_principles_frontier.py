"""Tests for principles-first frontier synthesis utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from larrak2.core.types import EvalContext
from larrak2.pipelines.principles_frontier import (
    OperatingPoint,
    load_principles_profile,
    synthesize_principles_frontier,
)


def test_load_principles_profile_default_is_valid() -> None:
    path, payload = load_principles_profile("iso_litvin_v1")
    assert path.exists()
    assert payload["profile_id"] == "iso_litvin_v1"
    assert isinstance(payload["source_references"], list)
    assert payload["source_references"]


def test_load_principles_profile_rejects_missing_source_references(tmp_path: Path) -> None:
    profile_path = tmp_path / "bad_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_id": "bad_v1",
                "anchor_manifest": "data/thermo/anchor_manifest_v1.json",
                "envelope": {
                    "rpm_min": 1200,
                    "rpm_max": 3200,
                    "torque_min": 80,
                    "torque_max": 220,
                },
                "gate_thresholds": {
                    "coverage_rpm_fraction_min": 0.6,
                    "coverage_torque_fraction_min": 0.6,
                },
                "source_references": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_references"):
        load_principles_profile(str(profile_path))


def test_synthesize_principles_frontier_emits_artifacts_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_points(*, profile_payload, anchor_manifest):  # noqa: ARG001
        return [
            OperatingPoint(rpm=1800.0, torque=120.0, source="a"),
            OperatingPoint(rpm=2200.0, torque=160.0, source="b"),
            OperatingPoint(rpm=2600.0, torque=200.0, source="c"),
        ]

    def _fake_restore(*, x_seed, ctx, root_max_iter, xl, xu):  # noqa: ARG001
        x = np.asarray(x_seed, dtype=np.float64).copy()
        basis = float((ctx.rpm / 1000.0) + (ctx.torque / 200.0))
        F = np.asarray(
            [
                basis,
                10.0 - basis,
                0.5 * basis,
                abs(5.0 - basis),
                0.25 * basis,
                12.0 - basis,
            ],
            dtype=np.float64,
        )
        G = np.zeros(10, dtype=np.float64)
        payload = {
            "root_success": True,
            "hard_violation_score": 0.0,
            "hard_reasons": [],
            "stages": [],
        }
        return x, payload, F, G

    monkeypatch.setattr("larrak2.pipelines.principles_frontier._build_operating_points", _fake_points)
    monkeypatch.setattr("larrak2.pipelines.principles_frontier._restore_candidate", _fake_restore)

    result = synthesize_principles_frontier(
        outdir=tmp_path / "out",
        ctx=EvalContext(rpm=2000.0, torque=150.0, fidelity=0, seed=3),
        profile_name="iso_litvin_v1",
        seed=3,
        seed_count=9,
        min_frontier_size=2,
        root_max_iter=8,
        export_archive_dir=tmp_path / "archive",
    )

    assert result.gate["frontier_gate_pass"] is True
    assert result.store.n_candidates >= 2
    assert Path(result.artifacts["principles_frontier_summary"]).exists()
    assert Path(result.artifacts["principles_frontier_gate"]).exists()
    assert Path(result.artifacts["principles_rootfinding_trace"]).exists()
    assert Path(result.artifacts["principles_export_archive_dir"]).exists()


def test_synthesize_principles_frontier_f0_disallows_placeholder_gate_basis_when_no_hard_feasible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_points(*, profile_payload, anchor_manifest):  # noqa: ARG001
        return [
            OperatingPoint(rpm=1800.0, torque=120.0, source="a"),
            OperatingPoint(rpm=2200.0, torque=160.0, source="b"),
            OperatingPoint(rpm=2600.0, torque=200.0, source="c"),
        ]

    def _fake_restore(*, x_seed, ctx, root_max_iter, xl, xu):  # noqa: ARG001
        x = np.asarray(x_seed, dtype=np.float64).copy()
        basis = float((ctx.rpm / 1000.0) + (ctx.torque / 200.0))
        F = np.asarray(
            [
                basis,
                10.0 - basis,
                0.5 * basis,
                abs(5.0 - basis),
                0.25 * basis,
                12.0 - basis,
            ],
            dtype=np.float64,
        )
        G = np.zeros(10, dtype=np.float64)
        payload = {
            "root_success": False,
            "hard_violation_score": 0.2,
            "hard_reasons": ["system_power_balance"],
            "stages": [],
        }
        return x, payload, F, G

    monkeypatch.setattr("larrak2.pipelines.principles_frontier._build_operating_points", _fake_points)
    monkeypatch.setattr("larrak2.pipelines.principles_frontier._restore_candidate", _fake_restore)

    result = synthesize_principles_frontier(
        outdir=tmp_path / "out_f0",
        ctx=EvalContext(rpm=2000.0, torque=150.0, fidelity=0, seed=3),
        profile_name="iso_litvin_v1",
        seed=3,
        seed_count=9,
        min_frontier_size=2,
        root_max_iter=8,
        export_archive_dir=tmp_path / "archive_f0",
    )

    assert result.store.n_candidates >= 2
    assert result.gate["gate_basis"] == "placeholder_frontier"
    assert result.gate["frontier_gate_pass"] is False
    assert result.gate["placeholder_frontier_disallowed"] is True
    assert result.gate["n_hard_feasible_explore"] == 0


def test_synthesize_principles_frontier_f0_allows_placeholder_with_nonproduction_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_points(*, profile_payload, anchor_manifest):  # noqa: ARG001
        return [
            OperatingPoint(rpm=1800.0, torque=120.0, source="a"),
            OperatingPoint(rpm=2200.0, torque=160.0, source="b"),
            OperatingPoint(rpm=2600.0, torque=200.0, source="c"),
        ]

    def _fake_restore(*, x_seed, ctx, root_max_iter, xl, xu):  # noqa: ARG001
        x = np.asarray(x_seed, dtype=np.float64).copy()
        basis = float((ctx.rpm / 1000.0) + (ctx.torque / 200.0))
        F = np.asarray(
            [
                basis,
                10.0 - basis,
                0.5 * basis,
                abs(5.0 - basis),
                0.25 * basis,
                12.0 - basis,
            ],
            dtype=np.float64,
        )
        G = np.zeros(10, dtype=np.float64)
        payload = {
            "root_success": False,
            "hard_violation_score": 0.2,
            "hard_reasons": ["system_power_balance"],
            "stages": [],
        }
        return x, payload, F, G

    monkeypatch.setattr("larrak2.pipelines.principles_frontier._build_operating_points", _fake_points)
    monkeypatch.setattr("larrak2.pipelines.principles_frontier._restore_candidate", _fake_restore)

    result = synthesize_principles_frontier(
        outdir=tmp_path / "out_f0_override",
        ctx=EvalContext(rpm=2000.0, torque=150.0, fidelity=0, seed=3),
        profile_name="iso_litvin_v1",
        seed=3,
        seed_count=9,
        min_frontier_size=2,
        root_max_iter=8,
        export_archive_dir=tmp_path / "archive_f0_override",
        allow_nonproduction_paths=True,
    )

    assert result.gate["gate_basis"] == "placeholder_frontier"
    assert result.gate["frontier_gate_pass"] is True
    assert result.gate["placeholder_frontier_disallowed"] is False
