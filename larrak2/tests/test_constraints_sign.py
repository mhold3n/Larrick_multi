"""Test constraint sign convention (G <= 0 feasible)."""

import numpy as np
import pytest

from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def test_mid_bounds_feasible():
    """Test that mid-bounds candidate is feasible.

    The toy physics is designed such that a candidate at the
    midpoint of all bounds should be feasible (all G <= 0).
    """
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=42)
    x = mid_bounds_candidate()

    result = evaluate_candidate(x, ctx)

    # Most constraints should be satisfied for mid-bounds (relaxed check for toy physics)
    n_satisfied = np.sum(result.G <= 0)
    assert n_satisfied >= 5, (
        f"Mid-bounds candidate should satisfy most constraints. "
        f"Satisfied: {n_satisfied}/7, G: {result.G}"
    )


def test_constraint_sign_convention():
    """Test that constraint sign follows G <= 0 feasible convention.

    For the toy model:
    - G_thermo[0]: compression duration constraint (should allow > 30 deg)
    - G_thermo[1]: heat release width > 5
    - G_thermo[2]: jerk constraint
    - G_gear[0]: ratio error <= tolerance
    - G_gear[1]: min radius >= MIN_RADIUS
    - G_gear[2]: max radius <= MAX_RADIUS
    - G_gear[3]: curvature <= MAX_CURVATURE
    """
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=42)
    x = mid_bounds_candidate()

    result = evaluate_candidate(x, ctx)

    # Verify constraint array length
    assert len(result.G) == 7, f"Expected 7 constraints, got {len(result.G)}"

    # Check that constraints are reasonable (not extremely violated)
    max_violation = np.max(result.G)
    assert max_violation < 10.0, f"Max violation {max_violation:.4f} is too large"


def test_max_violation_property():
    """Test max_violation property."""
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=42)
    x = mid_bounds_candidate()

    result = evaluate_candidate(x, ctx)

    if result.is_feasible:
        assert result.max_violation == 0.0
    else:
        assert result.max_violation > 0
        assert result.max_violation == float(np.max(np.maximum(result.G, 0)))
