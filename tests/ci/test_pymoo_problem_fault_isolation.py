"""Fault-isolation behavior for ParetoProblem population evaluation."""

from __future__ import annotations

import numpy as np

from larrak2.adapters.pymoo_problem import ParetoProblem
from larrak2.core.constraints import get_constraint_names
from larrak2.core.encoding import N_TOTAL, mid_bounds_candidate
from larrak2.core.types import EvalContext, EvalResult


def test_pymoo_problem_penalizes_single_candidate_eval_error(monkeypatch) -> None:
    sentinel = 0.123456789
    n_obj = 6
    n_constr = len(get_constraint_names(1))

    def _fake_eval(x, ctx):  # noqa: ARG001
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size > 0 and abs(float(arr[0]) - sentinel) < 1e-12:
            raise RuntimeError("synthetic_eval_failure")
        return EvalResult(
            F=np.linspace(0.1, 0.6, n_obj, dtype=np.float64),
            G=-np.ones(n_constr, dtype=np.float64),
            diag={},
        )

    monkeypatch.setattr("larrak2.adapters.pymoo_problem.evaluate_candidate", _fake_eval)

    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=3)
    problem = ParetoProblem(ctx=ctx)

    x_ok = np.asarray(mid_bounds_candidate(), dtype=np.float64).reshape(N_TOTAL)
    x_bad = x_ok.copy()
    x_bad[0] = sentinel
    X = np.vstack([x_ok, x_bad])
    out: dict[str, np.ndarray] = {}
    problem._evaluate(X, out)

    assert out["F"].shape == (2, n_obj)
    assert out["G"].shape == (2, n_constr)
    assert np.allclose(out["F"][0], np.linspace(0.1, 0.6, n_obj, dtype=np.float64))
    assert np.all(out["G"][0] <= 0.0)

    assert np.allclose(out["F"][1], np.full(n_obj, 1.0e6, dtype=np.float64))
    assert np.allclose(out["G"][1], np.full(n_constr, 1.0e3, dtype=np.float64))
    assert problem.n_eval_errors == 1
    assert any("synthetic_eval_failure" in key for key in problem.eval_error_signatures)
