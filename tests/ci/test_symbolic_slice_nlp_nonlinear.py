"""Unit tests for nonlinear symbolic slice NLP builder/solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from larrak2.core.encoding import N_TOTAL, mid_bounds_candidate
from larrak2.core.types import EvalContext, EvalResult
from larrak2.optimization.slicing.symbolic_slice_problem import solve_symbolic_slice_with_ipopt
from larrak2.optimization.solvers.ipopt.types import IPOPTResult
from larrak2.surrogate.stack import DenseLayer, StackSurrogateArtifact, default_feature_names, save_stack_artifact


def test_symbolic_slice_solver_uses_nonlinear_nlp(monkeypatch, tmp_path: Path) -> None:
    ca = pytest.importorskip("casadi")
    _ = ca

    n_in = N_TOTAL + 2  # x_full + (rpm, torque)
    n_out = 3  # 2 objectives + 1 constraint

    rng = np.random.default_rng(9)
    layer0 = DenseLayer(
        weight=rng.normal(scale=0.1, size=(4, n_in)),
        bias=rng.normal(scale=0.05, size=(4,)),
    )
    layer1 = DenseLayer(
        weight=rng.normal(scale=0.1, size=(n_out, 4)),
        bias=rng.normal(scale=0.05, size=(n_out,)),
    )
    artifact = StackSurrogateArtifact(
        feature_names=default_feature_names(N_TOTAL),
        objective_names=("obj0", "obj1"),
        constraint_names=("g0",),
        hidden_layers=(4,),
        activation="relu",
        leaky_relu_slope=0.01,
        fidelity=1,
        x_mean=np.zeros(n_in, dtype=np.float64),
        x_std=np.ones(n_in, dtype=np.float64),
        y_mean=np.zeros(n_out, dtype=np.float64),
        y_std=np.ones(n_out, dtype=np.float64),
        layers=(layer0, layer1),
    )
    artifact_path = tmp_path / "stack_model.npz"
    save_stack_artifact(artifact, artifact_path)

    def _fake_eval(x, _ctx):
        x_arr = np.asarray(x, dtype=np.float64)
        return EvalResult(
            F=np.array([float(np.sum(x_arr**2)), float(np.mean(x_arr))], dtype=np.float64),
            G=np.array([-0.1], dtype=np.float64),
            diag={},
        )

    def _mock_ipopt_solve(self, nlp, *, x0, lbx, ubx, lbg, ubg, p=None):
        ca_local = pytest.importorskip("casadi")
        H = ca_local.hessian(nlp["f"], nlp["x"])[0]
        h_fn = ca_local.Function("h_nlp", [nlp["x"]], [H])
        h_val = np.asarray(h_fn(x0), dtype=np.float64)
        assert np.any(np.abs(h_val) > 0.0), "Expected nonlinear (nonzero Hessian) objective"

        g_fn = ca_local.Function("g_nlp", [nlp["x"]], [nlp["g"]])
        g_val = np.asarray(g_fn(x0), dtype=np.float64).reshape(-1)
        return IPOPTResult(
            x_opt=np.asarray(x0, dtype=np.float64).reshape(-1),
            f_opt=float(0.0),
            g_opt=g_val,
            success=True,
            status="Solve_Succeeded",
            iterations=3,
            cpu_time_s=0.0,
            stats={"mock": True},
        )

    monkeypatch.setattr(
        "larrak2.optimization.slicing.symbolic_slice_problem.evaluate_candidate",
        _fake_eval,
    )
    monkeypatch.setattr(
        "larrak2.optimization.slicing.symbolic_slice_problem.IPOPTSolver.solve",
        _mock_ipopt_solve,
    )

    x0 = mid_bounds_candidate()
    ctx = EvalContext(rpm=2400.0, torque=140.0, fidelity=1, seed=3)

    result = solve_symbolic_slice_with_ipopt(
        x0=x0,
        ctx=ctx,
        active_indices=[0, 1, 2],
        surrogate_stack_path=str(artifact_path),
        mode="eps_constraint",
        eps_constraints=np.array([1.0, 1.0], dtype=np.float64),
        trust_radius=0.2,
        fidelity=1,
    )

    assert result.success is True
    assert result.ipopt_status == "Solve_Succeeded"
    assert result.diagnostics["nlp_formulation"] == "global_surrogate_symbolic"
    assert result.x_opt.shape == x0.shape
