"""Tests for sensitivity-ranked active set selection."""

from __future__ import annotations

from larrak2.core.encoding import N_TOTAL, group_indices, mid_bounds_candidate
from larrak2.core.types import EvalContext
from larrak2.optimization.slicing.active_set import select_active_set


def test_select_active_set_group_floor_and_deterministic():
    x0 = mid_bounds_candidate()
    ctx = EvalContext(rpm=2500.0, torque=150.0, fidelity=0, seed=1)

    first = select_active_set(x0, ctx, active_k=9, min_per_group=1, method="sensitivity")
    second = select_active_set(x0, ctx, active_k=9, min_per_group=1, method="sensitivity")

    assert first.active_indices == second.active_indices
    assert len(first.scores) == N_TOTAL
    assert len(first.active_indices) <= 9

    groups = group_indices()
    active = set(first.active_indices)
    for idxs in groups.values():
        assert any(i in active for i in idxs)

    # Active + frozen should partition the full variable index set.
    merged = sorted(first.active_indices + first.frozen_indices)
    assert merged == list(range(N_TOTAL))
