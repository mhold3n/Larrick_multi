"""Backend fallback tests for CasADi refinement."""

from __future__ import annotations

import numpy as np

from larrak2.adapters.casadi_refine import RefinementMode, refine_candidate
from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.types import EvalContext
from larrak2.optimization.slicing.slice_problem import SliceSolveResult


def test_casadi_failure_falls_back_to_scipy(monkeypatch):
    def _forced_fail(*_args, **_kwargs):
        return SliceSolveResult(
            x_opt=mid_bounds_candidate(),
            success=False,
            message="forced ipopt failure",
            ipopt_status="forced_failure",
            iterations=0,
            diagnostics={"forced": True},
        )

    monkeypatch.setattr("larrak2.adapters.casadi_refine.solve_slice_with_ipopt", _forced_fail)

    x0 = mid_bounds_candidate()
    ctx = EvalContext(rpm=3000, torque=200, fidelity=0, seed=7)

    result = refine_candidate(
        x0,
        ctx,
        mode=RefinementMode.WEIGHTED_SUM,
        backend="casadi",
        active_set=[0, 1, 2, 3],
        max_iter=5,
    )

    assert result.backend_used == "scipy_fallback"
    assert result.ipopt_status == "forced_failure"
    assert isinstance(result.success, bool)
    assert result.x_refined.shape == x0.shape
    assert np.all(np.isfinite(result.F_refined))
