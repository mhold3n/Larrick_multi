"""Test evaluation determinism."""

import numpy as np
import pytest

from larrak2.core.encoding import random_candidate, mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def test_determinism_same_inputs():
    """Test that same inputs produce identical outputs."""
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=42)
    x = mid_bounds_candidate()

    result1 = evaluate_candidate(x, ctx)
    result2 = evaluate_candidate(x, ctx)

    # F should be identical
    np.testing.assert_array_equal(
        result1.F, result2.F,
        err_msg="F differs between identical evaluations"
    )

    # G should be identical
    np.testing.assert_array_equal(
        result1.G, result2.G,
        err_msg="G differs between identical evaluations"
    )


def test_determinism_different_x():
    """Test that different x produces different outputs."""
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=42)

    rng = np.random.default_rng(42)
    x1 = random_candidate(rng)
    x2 = random_candidate(rng)

    # x1 and x2 should differ
    assert not np.allclose(x1, x2), "x1 and x2 should be different"

    result1 = evaluate_candidate(x1, ctx)
    result2 = evaluate_candidate(x2, ctx)

    # G should differ (constraints depend on params)
    assert not np.allclose(result1.G, result2.G), "Different x should give different G"


def test_determinism_different_context():
    """Test that different context produces different outputs."""
    x = mid_bounds_candidate()

    ctx1 = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=42)
    ctx2 = EvalContext(rpm=5000.0, torque=300.0, fidelity=0, seed=42)

    result1 = evaluate_candidate(x, ctx1)
    result2 = evaluate_candidate(x, ctx2)

    # Results should differ (gear loss depends on rpm/torque)
    # Note: thermo efficiency doesn't depend on rpm/torque in toy model,
    # but gear loss does
    assert not np.allclose(result1.F[1], result2.F[1]), (
        "Different context should give different gear loss"
    )


def test_determinism_seed_unused():
    """Test that seed doesn't affect deterministic evaluation.

    The evaluation should be fully deterministic without any
    randomness (unless explicitly modeled, which it isn't in toy physics).
    """
    x = mid_bounds_candidate()

    ctx1 = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=1)
    ctx2 = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=999)

    result1 = evaluate_candidate(x, ctx1)
    result2 = evaluate_candidate(x, ctx2)

    # For the toy model, seed should not affect results
    np.testing.assert_array_equal(
        result1.F, result2.F,
        err_msg="Different seeds should not affect deterministic evaluation"
    )
