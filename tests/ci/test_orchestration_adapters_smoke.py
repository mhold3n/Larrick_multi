"""Smoke tests for orchestration adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from larrak2.core.encoding import N_TOTAL, bounds
from larrak2.core.types import EvalContext
from larrak2.orchestration.adapters import (
    CEMAdapter,
    CasadiSolverAdapter,
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

    surrogate = HifiSurrogateAdapter(model_dir=tmp_path / "missing_models")
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

