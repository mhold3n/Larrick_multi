"""Tests for reduced-order principles region synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from larrak2.core.encoding import N_TOTAL, decode_candidate
from larrak2.core.types import EvalContext
from larrak2.pipelines.principles_core import (
    REALWORLD_NAMES,
    REDUCED_VARIABLE_NAMES,
    PrinciplesReducedVector,
    expand_reduced_vector,
    reduced_mid_bounds,
)
from larrak2.pipelines.principles_evaluator import (
    PrinciplesAlignmentResult,
    PrinciplesProxyResult,
)
from larrak2.pipelines.principles_frontier import (
    OperatingPoint,
    load_principles_profile,
    synthesize_principles_frontier,
)
from larrak2.pipelines.principles_search import PrinciplesDiagnosis


def test_load_principles_profile_v2_is_valid() -> None:
    path, payload = load_principles_profile("iso_litvin_v2")
    assert path.exists()
    assert payload["profile_id"] == "iso_litvin_v2"
    assert payload["reduced_core"]["variable_names"] == list(REDUCED_VARIABLE_NAMES)
    assert payload["weight_vectors"]
    assert payload["source_references"]


def test_load_principles_profile_v1_upgrades_to_v2_fields() -> None:
    path, payload = load_principles_profile("iso_litvin_v1")
    assert path.exists()
    assert payload["profile_id"] == "iso_litvin_v1"
    for key in (
        "blend_weights",
        "canonical_alignment",
        "reduced_core",
        "expansion_policy",
        "normalization_scales",
        "weight_vectors",
    ):
        assert key in payload


def test_load_principles_profile_rejects_missing_source_references(tmp_path: Path) -> None:
    profile_path = tmp_path / "bad_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_id": "bad_v2",
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


def test_expand_reduced_vector_is_deterministic_and_applies_defaults() -> None:
    _, payload = load_principles_profile("iso_litvin_v2")
    reduced = PrinciplesReducedVector.from_array(reduced_mid_bounds())
    x_full, expansion = expand_reduced_vector(reduced, profile_payload=payload, rpm=7000.0)
    candidate = decode_candidate(x_full)

    assert x_full.shape == (N_TOTAL,)
    assert candidate.gear.pitch_coeffs[5] == pytest.approx(0.0)
    assert candidate.gear.pitch_coeffs[6] == pytest.approx(0.0)
    defaults = expansion["realworld_defaults"]
    for name in REALWORLD_NAMES:
        assert name in defaults
    assert defaults["surface_finish_level"] == pytest.approx(0.70)
    assert defaults["lube_mode_level"] == pytest.approx(0.70)
    assert defaults["material_quality_level"] == pytest.approx(0.50)
    assert expansion["deterministic_overrides"]["material_quality_level"] >= 0.50
    assert expansion["pitch_line_velocity_m_s"] > 0.0


def test_synthesize_principles_frontier_emits_region_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_points(*, profile_payload, anchor_manifest):  # noqa: ARG001
        return [
            OperatingPoint(rpm=1800.0, torque=120.0, source="anchor_a"),
            OperatingPoint(rpm=2600.0, torque=200.0, source="anchor_b"),
        ]

    def _fake_search(
        *,
        profile_payload,
        operating_points,
        base_ctx,
        alignment_mode,
        alignment_fidelity,
        alignment_phase,
        stage_max_iter,
        region_min_size,
    ):
        assert alignment_mode == "blend"
        assert alignment_fidelity == 1
        assert alignment_phase == "explore"
        assert stage_max_iter == 8
        assert region_min_size == 2
        assert base_ctx.fidelity == 0
        assert len(operating_points) == 2

        reduced_a = reduced_mid_bounds()
        reduced_b = reduced_mid_bounds().copy()
        reduced_b[0] += 5.0
        x_a, expansion_a = expand_reduced_vector(
            reduced_a, profile_payload=profile_payload, rpm=1800.0
        )
        x_b, expansion_b = expand_reduced_vector(
            reduced_b, profile_payload=profile_payload, rpm=2600.0
        )

        proxy_a = PrinciplesProxyResult(
            F=np.array([1.0, 1.2, 0.9, 0.2, 0.3, 0.1], dtype=np.float64),
            G=np.array([-0.1, -0.2], dtype=np.float64),
            diag={"constraints": []},
            objective_names=(
                "eta_comb_gap",
                "eta_exp_gap",
                "eta_gear_gap",
                "motion_law_penalty",
                "life_damage_penalty",
                "material_risk_penalty",
            ),
            constraint_names=("c0", "c1"),
            x_full=x_a,
            expansion_policy=expansion_a,
        )
        proxy_b = PrinciplesProxyResult(
            F=np.array([0.9, 1.1, 0.8, 0.25, 0.35, 0.15], dtype=np.float64),
            G=np.array([-0.05, -0.1], dtype=np.float64),
            diag={"constraints": []},
            objective_names=proxy_a.objective_names,
            constraint_names=proxy_a.constraint_names,
            x_full=x_b,
            expansion_policy=expansion_b,
        )
        align_a = PrinciplesAlignmentResult(
            F=np.array([1.05, 1.15, 0.92, 0.22, 0.28, 0.12], dtype=np.float64),
            G=np.array([-0.05, -0.1], dtype=np.float64),
            diag={"constraints": []},
            objective_names=proxy_a.objective_names,
            constraint_names=proxy_a.constraint_names,
        )
        align_b = PrinciplesAlignmentResult(
            F=np.array([0.95, 1.05, 0.85, 0.27, 0.33, 0.18], dtype=np.float64),
            G=np.array([-0.02, -0.08], dtype=np.float64),
            diag={"constraints": []},
            objective_names=proxy_a.objective_names,
            constraint_names=proxy_a.constraint_names,
        )
        return {
            "records": [
                {
                    "operating_point_index": 0,
                    "operating_point": {"rpm": 1800.0, "torque": 120.0, "source": "anchor_a"},
                    "weight_name": "eta_comb_basis",
                    "weight_vector": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    "seed_state_name": "nominal_profile",
                    "z_reduced": reduced_a,
                    "x_full": x_a,
                    "proxy": proxy_a,
                    "alignment": align_a,
                    "F_blend": np.array([1.02, 1.18, 0.91, 0.21, 0.29, 0.11], dtype=np.float64),
                    "G_combined": np.array([-0.05, -0.1], dtype=np.float64),
                    "hard_penalty": 0.0,
                    "proxy_feasible": True,
                    "alignment_feasible": True,
                    "combined_feasible": True,
                    "objective_delta_abs": np.abs(proxy_a.F - align_a.F),
                    "objective_delta_rel": np.full(6, 0.05, dtype=np.float64),
                    "constraint_delta_abs": np.abs(proxy_a.G - align_a.G),
                    "dominant_mismatch_objectives": ["eta_comb_gap"],
                    "dominant_mismatch_constraints": ["c0"],
                    "stage_records": [{"stage": "stage_a", "success": True}],
                    "weighted_score": 1.0,
                },
                {
                    "operating_point_index": 1,
                    "operating_point": {"rpm": 2600.0, "torque": 200.0, "source": "anchor_b"},
                    "weight_name": "durability_trade",
                    "weight_vector": np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float64),
                    "seed_state_name": "torque_biased",
                    "z_reduced": reduced_b,
                    "x_full": x_b,
                    "proxy": proxy_b,
                    "alignment": align_b,
                    "F_blend": np.array([0.93, 1.08, 0.83, 0.26, 0.34, 0.17], dtype=np.float64),
                    "G_combined": np.array([-0.02, -0.08], dtype=np.float64),
                    "hard_penalty": 0.0,
                    "proxy_feasible": True,
                    "alignment_feasible": True,
                    "combined_feasible": True,
                    "objective_delta_abs": np.abs(proxy_b.F - align_b.F),
                    "objective_delta_rel": np.full(6, 0.04, dtype=np.float64),
                    "constraint_delta_abs": np.abs(proxy_b.G - align_b.G),
                    "dominant_mismatch_objectives": ["life_damage_penalty"],
                    "dominant_mismatch_constraints": ["c1"],
                    "stage_records": [{"stage": "stage_c", "success": True}],
                    "weighted_score": 0.8,
                },
            ],
            "region_indices": [0, 1],
            "region_summary": {
                "n_operating_points": 2,
                "n_weight_vectors": 2,
                "n_stage_solves": 4,
                "n_proxy_feasible": 2,
                "n_alignment_feasible": 2,
                "n_region_candidates": 2,
                "envelope_coverage_rpm_fraction": 1.0,
                "envelope_coverage_torque_fraction": 1.0,
                "region_min_size_required": 2,
            },
            "proxy_vs_canonical": {
                "objective_delta_median": [0.05] * 6,
                "objective_delta_max": [0.08] * 6,
                "dominant_failure_constraints": [],
                "representative_candidate_agreement": [{"candidate_index": 0}],
            },
            "diagnosis": PrinciplesDiagnosis(
                classification="region_ready",
                metrics={"profile_name": profile_payload["profile_id"]},
            ),
        }

    monkeypatch.setattr(
        "larrak2.pipelines.principles_frontier._build_operating_points", _fake_points
    )
    monkeypatch.setattr(
        "larrak2.pipelines.principles_frontier.search_principles_region", _fake_search
    )

    result = synthesize_principles_frontier(
        outdir=tmp_path / "out",
        ctx=EvalContext(rpm=2000.0, torque=150.0, fidelity=0, seed=3),
        profile_name="iso_litvin_v2",
        seed=3,
        min_frontier_size=2,
        root_max_iter=8,
        export_archive_dir=tmp_path / "archive",
        alignment_mode="blend",
        alignment_fidelity=1,
    )

    assert result.store.n_candidates == 2
    assert result.region_summary["source_region_pass"] is True
    assert result.diagnosis["classification"] == "region_ready"
    assert Path(result.artifacts["principles_region_summary"]).exists()
    assert Path(result.artifacts["principles_proxy_vs_canonical"]).exists()
    assert Path(result.artifacts["principles_diagnosis"]).exists()
    assert Path(result.artifacts["principles_candidate_records"]).exists()
    assert Path(result.artifacts["principles_frontier_gate"]).exists()
    assert Path(result.artifacts["principles_export_archive_dir"]).exists()

    diagnosis_payload = json.loads(
        Path(result.artifacts["principles_diagnosis"]).read_text(encoding="utf-8")
    )
    assert diagnosis_payload["classification"] == "region_ready"
    region_summary = json.loads(
        Path(result.artifacts["principles_region_summary"]).read_text(encoding="utf-8")
    )
    assert region_summary["reduced_core"]["variable_names"] == list(REDUCED_VARIABLE_NAMES)
    assert region_summary["expansion_policy"]["pitch_coeff_5"] == 0.0
