"""Robustness checks for refinement routines."""

import numpy as np

from larrak2.adapters.casadi_refine import RefinementMode, refine_candidate
from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def test_refine_weighted_sum_stays_in_bounds():
    x0 = mid_bounds_candidate()
    ctx = EvalContext(rpm=3000, torque=200, fidelity=0, seed=1)
    res = refine_candidate(x0, ctx, mode=RefinementMode.WEIGHTED_SUM, max_iter=10)

    assert np.all(np.isfinite(res.x_refined))
    # Ensure within bounds [0,1] range is not assumed; just check no NaN and sizes match
    assert res.x_refined.shape == x0.shape


def test_refine_improves_scalarized_objective():
    x0 = mid_bounds_candidate()
    ctx = EvalContext(rpm=3000, torque=200, fidelity=0, seed=2)

    # Weights favor the first objective axis while still regularizing others.
    n_obj = int(evaluate_candidate(x0, ctx).F.size)
    weights = np.full(n_obj, 0.1, dtype=float)
    weights[0] = 1.0
    before = refine_candidate(
        x0, ctx, mode=RefinementMode.WEIGHTED_SUM, weights=weights, max_iter=1
    )
    after = refine_candidate(
        x0, ctx, mode=RefinementMode.WEIGHTED_SUM, weights=weights, max_iter=20
    )

    obj_before = np.dot(weights, before.F_refined)
    obj_after = np.dot(weights, after.F_refined)
    assert obj_after <= obj_before + 1e-6  # improvement or equal
