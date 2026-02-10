"""Tests for strict fidelity=2 behavior (OpenFOAM NN required)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def test_fidelity2_requires_openfoam_nn(monkeypatch):
    # Point to a non-existent artifact path.
    monkeypatch.setenv("LARRAK2_OPENFOAM_NN_PATH", "/tmp/does_not_exist_openfoam_breathing.pt")

    x = mid_bounds_candidate()
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=2, seed=1)

    with pytest.raises(FileNotFoundError):
        evaluate_candidate(x, ctx)


def test_fidelity2_with_openfoam_nn_is_finite():
    # conftest ensures a synthetic artifact exists for tests
    assert os.environ.get("LARRAK2_OPENFOAM_NN_PATH")

    x = mid_bounds_candidate()
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=2, seed=1)

    res = evaluate_candidate(x, ctx)
    assert res.F.shape == (3,)
    assert np.all(np.isfinite(res.F))
    assert np.all(np.isfinite(res.G))

