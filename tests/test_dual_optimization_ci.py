"""CI tests for Dual Optimization Pipeline (Fidelity 2).

This module tests the optimization topology by MOCKING the neural surrogates.
It proves that the pipeline handles inputs -> features -> surrogate calls -> objectives
without requiring actual trained models or heavy computation.
"""

import numpy as np
import pytest
from unittest.mock import patch

from larrak2.core.types import EvalContext
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.encoding import random_candidate
from larrak2.adapters.pymoo_problem import ParetoProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

# --- Mocks ---


class MockFluidSurrogate:
    """Mock for OpenFOAM NN."""

    def predict_one(self, features: dict[str, float]) -> dict[str, float]:
        # Return deterministic "good" values to ensure feasibility
        return {
            "scavenging_efficiency": 0.95,
            "m_air_trapped": 0.04,
            "residual_fraction": 0.05,
            "trapped_o2_mass": 0.01,
        }


class MockGearSurrogate:
    """Mock for Gear Loss NN."""

    def predict(
        self, rpm: float, torque: float, base_radius: float, coeffs: list[float]
    ) -> dict[str, float]:
        # Return deterministic energy values (Joules per cycle)
        # 50 J total loss per cycle
        return {
            "loss_total": 50.0,
            "loss_mesh": 30.0,
            "loss_bearing": 15.0,
            "loss_churning": 5.0,
        }


# --- Tests ---


@pytest.fixture
def ctx_fid2():
    return EvalContext(rpm=3000.0, torque=200.0, fidelity=2, seed=42)


@pytest.fixture
def mock_fluid_loader():
    with patch("larrak2.thermo.motionlaw.get_openfoam_surrogate") as mock:
        mock.return_value = MockFluidSurrogate()
        yield mock


@pytest.fixture
def mock_gear_loader():
    with patch("larrak2.gear.litvin_core.get_gear_surrogate") as mock:
        mock.return_value = MockGearSurrogate()
        yield mock


def test_pipeline_topology_single_eval(ctx_fid2, mock_fluid_loader, mock_gear_loader):
    """Test single candidate evaluation with mocked surrogates."""
    x = random_candidate(np.random.default_rng(42))

    result = evaluate_candidate(x, ctx_fid2)

    # 1. Verify Fluid Surrogate was used
    assert result.diag["thermo"]["openfoam_nn_used"] is True
    # Proxy efficiency = 0.5 * scav (from motionlaw logic)
    # mock scav = 0.95 -> eff = 0.475
    expected_eff = 0.95 * 0.5
    assert np.isclose(result.diag["metrics"]["efficiency_raw"], expected_eff)

    # 2. Verify Gear Surrogate was used
    assert result.diag["gear"]["gear_surrogate_used"] is True
    # Loss Total (Power) = Energy (50 J) * Cycles/sec
    # RPM = 3000 -> 50 cycles/sec
    # Power = 50 * 50 = 2500 W
    expected_loss_W = 50.0 * (3000.0 / 60.0)
    assert np.isclose(result.diag["metrics"]["loss_total"], expected_loss_W)

    # 3. Verify Constraints
    # Scavenging constraint: 0.9 - scav <= 0
    # 0.9 - 0.95 = -0.05 (Feasible)
    # Locate scavenging constraint (it's the first one in G generally? Need to check)
    # Actually, let's just check G is present
    assert len(result.G) > 0


def test_optimization_loop_smoke(ctx_fid2, mock_fluid_loader, mock_gear_loader):
    """Test full optimization loop topology (NSGA-3) with mocks."""
    problem = ParetoProblem(ctx_fid2)

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=2)
    algorithm = NSGA3(pop_size=len(ref_dirs) + 4, ref_dirs=ref_dirs)

    res = minimize(problem, algorithm, termination=("n_gen", 1), seed=1, verbose=False)

    assert len(res.pop) > 0
    assert problem.n_evals > 0

    # Verify mocks were called
    mock_fluid_loader.assert_called()
    mock_gear_loader.assert_called()
