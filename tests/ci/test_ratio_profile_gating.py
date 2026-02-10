"""Edge-case gating for ratio profile continuity."""

import numpy as np

from larrak2.core.types import EvalContext
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.encoding import Candidate, ThermoParams, GearParams, encode_candidate


def _make_candidate(compression, expansion, hr_center, hr_width):
    thermo = ThermoParams(
        compression_duration=compression,
        expansion_duration=expansion,
        heat_release_center=hr_center,
        heat_release_width=hr_width,
        lambda_af=1.0,
    )
    gear = GearParams(base_radius=40.0, pitch_coeffs=np.zeros(7))
    return encode_candidate(Candidate(thermo=thermo, gear=gear))


def test_ratio_profile_extremes_are_finite():
    ctx = EvalContext(rpm=3000, torque=200, fidelity=0, seed=1)
    # Edge heat release width small and compression near min
    x = _make_candidate(30.0, 120.0, 0.0, 10.0)
    res = evaluate_candidate(x, ctx)
    stats = res.diag["thermo"]["ratio_profile_stats"]
    assert stats["finite"]
    assert stats["min"] > 0.0
    assert stats["max_slope"] < 5.0
    assert stats["max_jerk"] < 1e6


def test_ratio_profile_fidelity1_extremes():
    ctx = EvalContext(rpm=3000, torque=200, fidelity=1, seed=1)
    x = _make_candidate(30.0, 120.0, 0.0, 10.0)
    res = evaluate_candidate(x, ctx)
    stats = res.diag["thermo"]["ratio_profile_stats"]
    assert stats["finite"]
    assert stats["min"] > 0.0
    assert stats["max_slope"] < 5.0
    assert stats["max_jerk"] < 1e6
