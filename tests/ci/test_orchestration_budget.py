"""Tests for orchestration budget manager."""

from __future__ import annotations

import numpy as np

from larrak2.orchestration.budget import BudgetAllocation, BudgetManager


def test_budget_allocation_normalizes() -> None:
    alloc = BudgetAllocation(1.0, 1.0, 1.0, 1.0)
    total = (
        alloc.best_fraction
        + alloc.uncertain_fraction
        + alloc.disagreement_fraction
        + alloc.random_fraction
    )
    assert np.isclose(total, 1.0)


def test_budget_select_consumes_budget_and_returns_unique_indices() -> None:
    manager = BudgetManager(total_sim_calls=20, reserve_for_validation=0.0, seed=123)
    candidates = [{"id": i} for i in range(10)]
    preds = np.linspace(0.0, 1.0, 10)
    unc = np.linspace(1.0, 0.0, 10)

    selected = manager.select(candidates, preds, unc, batch_size=4)
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert manager.state.used == 4
    assert manager.remaining() == 16


def test_budget_select_respects_remaining_budget() -> None:
    manager = BudgetManager(total_sim_calls=3, reserve_for_validation=0.0, seed=123)
    candidates = [{"id": i} for i in range(10)]
    preds = np.linspace(0.0, 1.0, 10)
    unc = np.linspace(1.0, 0.0, 10)

    selected = manager.select(candidates, preds, unc, batch_size=10)
    assert len(selected) == 3
    assert manager.exhausted() is True
