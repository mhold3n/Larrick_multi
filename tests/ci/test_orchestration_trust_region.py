"""Tests for orchestration trust-region controller."""

from __future__ import annotations

import numpy as np

from larrak2.orchestration.trust_region import TrustRegion


def test_trust_region_bounds_step_and_updates_radius() -> None:
    tr = TrustRegion(n_vars=3)

    step = np.array([0.2, -0.2, 0.05], dtype=np.float64)
    bounded = tr.bound_step(step, uncertainty=0.0)
    assert np.all(np.abs(bounded) <= np.abs(step) + 1e-12)

    radius_before = np.mean(np.asarray(tr.radius, dtype=np.float64))
    tr.update(predicted_improvement=1.0, actual_improvement=0.98, uncertainty_at_step=0.1)
    radius_after_good = np.mean(np.asarray(tr.radius, dtype=np.float64))
    assert radius_after_good >= radius_before

    tr.update(predicted_improvement=1.0, actual_improvement=0.1, uncertainty_at_step=0.5)
    radius_after_bad = np.mean(np.asarray(tr.radius, dtype=np.float64))
    assert radius_after_bad <= radius_after_good
