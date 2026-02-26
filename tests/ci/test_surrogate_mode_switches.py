"""CI coverage for explicit surrogate mode switches (no implicit fallback)."""

from __future__ import annotations

import numpy as np
import pytest

from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def test_calculix_nn_mode_requires_model(monkeypatch):
    monkeypatch.setenv("LARRAK2_CALCULIX_NN_PATH", "/tmp/does_not_exist_calculix.pt")

    x = mid_bounds_candidate()
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=1, calculix_stress_mode="nn")

    with pytest.raises(FileNotFoundError):
        evaluate_candidate(x, ctx)


def test_calculix_analytical_mode_is_explicit_bypass(monkeypatch):
    monkeypatch.setenv("LARRAK2_CALCULIX_NN_PATH", "/tmp/does_not_exist_calculix.pt")

    x = mid_bounds_candidate()
    ctx = EvalContext(
        rpm=3000.0,
        torque=200.0,
        fidelity=0,
        seed=1,
        calculix_stress_mode="analytical",
    )

    res = evaluate_candidate(x, ctx)
    assert np.all(np.isfinite(res.F))
    assert np.all(np.isfinite(res.G))
    assert res.diag["gear"]["radius_strategy"]["calculix_stress_mode"] == "analytical"
    assert res.diag["gear"]["radius_strategy"]["stress_source"] == "analytical_proxy"


def test_gear_loss_nn_mode_requires_model_dir():
    x = mid_bounds_candidate()
    ctx = EvalContext(
        rpm=3000.0,
        torque=200.0,
        fidelity=2,
        seed=1,
        gear_loss_mode="nn",
        gear_loss_model_dir="/tmp/does_not_exist_gear_loss_dir",
    )

    with pytest.raises(FileNotFoundError):
        evaluate_candidate(x, ctx)
