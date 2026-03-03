"""Two-zone thermo conservation and metadata checks."""

from __future__ import annotations

import numpy as np

from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def test_two_zone_conservation_metrics_present_and_bounded() -> None:
    x = mid_bounds_candidate()
    ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=1,
        seed=7,
        thermo_model="two_zone_eq_v1",
    )

    result = evaluate_candidate(x, ctx)
    thermo = result.diag.get("thermo", {})

    assert thermo.get("thermo_solver_status") == "ok"
    assert thermo.get("thermo_model_version") == "two_zone_eq_v1"
    assert np.isfinite(float(thermo.get("thermo_mass_residual", np.nan)))
    assert np.isfinite(float(thermo.get("thermo_energy_residual", np.nan)))
    assert float(thermo.get("thermo_mass_residual", 1.0)) <= 1e-4
    assert float(thermo.get("thermo_energy_residual", 1.0)) <= 3e-2
