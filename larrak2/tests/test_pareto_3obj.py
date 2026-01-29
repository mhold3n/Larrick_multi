"""Tests for Phase 9: Higher-Dimensional Pareto (3 objectives)."""

import numpy as np
import pytest
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from larrak2.adapters.pymoo_problem import ParetoProblem
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext
from larrak2.core.encoding import N_TOTAL


@pytest.fixture
def ctx_v1():
    return EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)


def test_evaluator_returns_3_objectives(ctx_v1):
    """Test that evaluate_candidate returns 3 objectives."""
    x = np.random.rand(N_TOTAL)
    result = evaluate_candidate(x, ctx_v1)
    
    assert len(result.F) == 3
    # Check structure: [-efficiency, loss, radius]
    eff_neg = result.F[0]
    loss = result.F[1]
    radius = result.F[2]
    
    # Basic sanity checks on values
    assert -1.0 <= eff_neg <= 0.0  # efficiency 0-100%
    assert loss >= 0.0
    assert radius > 0.0
    
    # Check diag has version info
    assert "versions" in result.diag


def test_pareto_problem_config(ctx_v1):
    """Test that ParetoProblem is configured for 3 objectives."""
    problem = ParetoProblem(ctx_v1)
    assert problem.N_OBJ == 3
    assert problem.n_obj == 3
    assert problem.n_constr == 7  # 3 thermo + 4 gear


def test_3obj_determinism(ctx_v1):
    """Test that 3-objective evaluation is deterministic."""
    x = np.random.rand(N_TOTAL)
    
    res1 = evaluate_candidate(x, ctx_v1)
    res2 = evaluate_candidate(x, ctx_v1)
    
    np.testing.assert_allclose(res1.F, res2.F, err_msg="Outcomes not deterministic")
    np.testing.assert_allclose(res1.G, res2.G, err_msg="Constraints not deterministic")


def test_nsga3_integration_smoke(ctx_v1):
    """Smoke test for NSGA-III integration (just 1 gen)."""
    problem = ParetoProblem(ctx_v1)
    
    # Minimal ref dirs
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=2)
    pop_size = len(ref_dirs) + 4 # Ensure > ref_dirs
    
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
    )
    
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 1),
        seed=1,
        verbose=False
    )
    
    assert res.F is not None
    assert res.F.shape[1] == 3
    # At least some evaluations happened
    assert problem.n_evals > 0
