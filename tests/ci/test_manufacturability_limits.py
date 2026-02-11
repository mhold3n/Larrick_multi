"""Tests for manufacturability-derived ratio-rate limits integration."""

import numpy as np

from larrak2.core.encoding import Candidate, GearParams, ThermoParams, encode_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext
from larrak2.gear.manufacturability_limits import (
    ManufacturingProcessParams,
    compute_manufacturable_ratio_rate_limits,
)


def _candidate() -> np.ndarray:
    thermo = ThermoParams(
        compression_duration=60.0,
        expansion_duration=90.0,
        heat_release_center=12.0,
        heat_release_width=25.0,
        lambda_af=1.0,
    )
    gear = GearParams(base_radius=40.0, pitch_coeffs=np.zeros(7))
    return encode_candidate(Candidate(thermo=thermo, gear=gear))


def test_ratio_rate_limit_envelope_is_finite_and_cached():
    gear = GearParams(base_radius=40.0, pitch_coeffs=np.zeros(7))
    proc = ManufacturingProcessParams(kerf_mm=0.2, overcut_mm=0.05, min_ligament_mm=0.3)
    durations = np.array([45.0, 90.0])

    env_a = compute_manufacturable_ratio_rate_limits(gear, process=proc, durations_deg=durations)
    env_b = compute_manufacturable_ratio_rate_limits(gear, process=proc, durations_deg=durations)

    assert np.all(np.isfinite(env_a.max_ratio_slope))
    assert np.all(env_a.max_ratio_slope >= 0.0)
    assert env_a is env_b  # cache reuse


def test_evaluator_uses_manufacturability_limit_for_thermo_constraint():
    x = _candidate()

    # aggressive process assumptions should tighten manufacturability envelope
    tight_ctx = EvalContext(
        rpm=3000.0,
        torque=200.0,
        fidelity=1,
        seed=7,
        gear_process_params={
            "kerf_mm": 1.2,
            "overcut_mm": 0.8,
            "min_ligament_mm": 2.0,
            "min_feature_radius_mm": 1.0,
        },
    )

    res = evaluate_candidate(x, tight_ctx)

    thermo_diag = res.diag["thermo"]
    limits_diag = res.diag["manufacturability_limits"]

    assert thermo_diag["ratio_slope_limit_source"] == "manufacturability"
    assert np.isfinite(thermo_diag["ratio_slope_limit_used"])
    assert thermo_diag["ratio_slope_limit_used"] <= 2.0  # bounded by legacy static constant
    assert np.isfinite(limits_diag["applied_ratio_slope_limit"])
    assert limits_diag["applied_ratio_slope_limit"] == thermo_diag["ratio_slope_limit_used"]
