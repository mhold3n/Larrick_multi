"""Continuity checks for requested ratio profile."""

import numpy as np

from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.types import EvalContext


def _assert_profile_ok(stats: dict):
    assert stats["finite"] is True
    assert stats["min"] > 0.0
    assert stats["max_slope"] < 0.5  # smooth enough for gear synthesis


def test_ratio_profile_continuity_fidelity0():
    x = mid_bounds_candidate()
    ctx = EvalContext(rpm=3000, torque=200, fidelity=0, seed=1)
    res = evaluate_candidate(x, ctx)
    stats = res.diag["thermo"]["ratio_profile_stats"]
    _assert_profile_ok(stats)


def test_ratio_profile_continuity_fidelity1():
    x = mid_bounds_candidate()
    ctx = EvalContext(rpm=3000, torque=200, fidelity=1, seed=1)
    res = evaluate_candidate(x, ctx)
    stats = res.diag["thermo"]["ratio_profile_stats"]
    _assert_profile_ok(stats)
