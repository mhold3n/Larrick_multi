"""Smoke tests for orchestration adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from larrak2.adapters.casadi_refine import RefinementResult
from larrak2.core.encoding import N_TOTAL, bounds
from larrak2.core.types import EvalContext
from larrak2.orchestration.adapters import (
    CasadiSolverAdapter,
    CEMAdapter,
    HifiSurrogateAdapter,
    PhysicsSimulationAdapter,
)


def _midpoint_candidate() -> dict[str, np.ndarray]:
    xl, xu = bounds()
    x = (xl + xu) * 0.5
    assert x.size == N_TOTAL
    return {"x": x}


def test_cem_adapter_generation_and_feasibility() -> None:
    adapter = CEMAdapter()
    rng = np.random.default_rng(0)
    batch = adapter.generate_batch({}, n=5, rng=rng)

    assert len(batch) == 5
    for cand in batch:
        assert np.asarray(cand["x"]).shape == (N_TOTAL,)

    feasible, score = adapter.check_feasibility(batch[0])
    assert isinstance(feasible, bool)
    assert 0.0 <= score <= 1.0

    repaired = adapter.repair({"x": np.asarray(batch[0]["x"]) * 100.0})
    x_rep = np.asarray(repaired["x"], dtype=np.float64)
    xl, xu = bounds()
    assert np.all(x_rep >= xl - 1e-12)
    assert np.all(x_rep <= xu + 1e-12)


def test_surrogate_solver_simulation_smoke(tmp_path: Path) -> None:
    candidate = _midpoint_candidate()
    ctx = EvalContext(rpm=2500.0, torque=120.0, fidelity=0, seed=123)

    surrogate = HifiSurrogateAdapter(
        model_dir=tmp_path / "missing_models",
        allow_heuristic_fallback=True,
        validation_mode="off",
    )
    pred, unc = surrogate.predict([candidate])
    assert pred.shape == (1,)
    assert unc.shape == (1,)

    solver = CasadiSolverAdapter(backend="scipy", mode="weighted_sum")
    refined = solver.refine(
        candidate,
        context=ctx,
        max_step=np.ones(N_TOTAL, dtype=np.float64) * 0.05,
    )
    x_ref = np.asarray(refined["x"], dtype=np.float64)
    assert x_ref.shape == (N_TOTAL,)

    sim = PhysicsSimulationAdapter(work_dir=tmp_path / "sim_runs")
    result = sim.evaluate(refined, context=ctx)
    assert "objective" in result
    assert np.isfinite(float(result["objective"]))


def test_casadi_solver_adapter_passes_stack_path_and_ipopt(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _mock_refine_candidate(x0, ctx, **kwargs):
        captured["surrogate_stack_path"] = kwargs.get("surrogate_stack_path")
        captured["fidelity"] = kwargs.get("fidelity")
        captured["ipopt_options"] = kwargs.get("ipopt_options")
        captured["trust_radius"] = kwargs.get("trust_radius")
        return RefinementResult(
            x_refined=np.asarray(x0, dtype=np.float64),
            F_refined=np.zeros(6, dtype=np.float64),
            G_refined=np.zeros(10, dtype=np.float64),
            diag={},
            success=True,
            message="ok",
            backend_used="casadi",
            ipopt_status="Solve_Succeeded",
        )

    monkeypatch.setattr(
        "larrak2.orchestration.adapters.solver_adapter.refine_candidate", _mock_refine_candidate
    )

    solver = CasadiSolverAdapter(
        backend="casadi",
        mode="weighted_sum",
        stack_model_path="outputs/artifacts/surrogates/stack_f2/stack_f2_surrogate.npz",
        ipopt_options={"max_iter": 17, "tol": 1e-7},
        trust_radius=0.12,
    )
    candidate = _midpoint_candidate()
    ctx = EvalContext(rpm=2500.0, torque=120.0, fidelity=2, seed=123)
    solver.refine(candidate, context=ctx, max_step=np.ones(N_TOTAL, dtype=np.float64) * 0.05)

    assert captured["surrogate_stack_path"] == (
        "outputs/artifacts/surrogates/stack_f2/stack_f2_surrogate.npz"
    )
    assert captured["fidelity"] == 2
    assert captured["ipopt_options"] == {"max_iter": 17, "tol": 1e-7}
    assert captured["trust_radius"] == 0.12
