"""Strict backend behavior tests for CasADi refinement."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from larrak2.adapters.casadi_refine import RefinementMode, refine_candidate
from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.types import EvalContext
from larrak2.optimization.slicing.slice_problem import SliceSolveResult


def test_casadi_failure_does_not_fallback(monkeypatch, tmp_path: Path):
    def _forced_fail(*_args, **_kwargs):
        return SliceSolveResult(
            x_opt=mid_bounds_candidate(),
            success=False,
            message="forced ipopt failure",
            ipopt_status="forced_failure",
            iterations=0,
            diagnostics={"forced": True, "nlp_formulation": "global_surrogate_symbolic"},
        )

    monkeypatch.setattr("larrak2.adapters.casadi_refine.solve_slice_with_ipopt", _forced_fail)

    x0 = mid_bounds_candidate()
    ctx = EvalContext(rpm=3000, torque=200, fidelity=1, seed=7)
    stack_model = tmp_path / "stack_f1_surrogate.npz"
    stack_model.write_bytes(b"placeholder")

    result = refine_candidate(
        x0,
        ctx,
        mode=RefinementMode.WEIGHTED_SUM,
        backend="casadi",
        active_set=[0, 1, 2, 3],
        surrogate_stack_path=str(stack_model),
        max_iter=5,
    )

    assert result.backend_used == "casadi"
    assert result.success is False
    assert result.ipopt_status == "forced_failure"
    assert result.x_refined.shape == x0.shape
    assert np.all(np.isfinite(result.F_refined))
