"""Tests for shared multi-objective production gate checks."""

from __future__ import annotations

from larrak2.optimization.production_gate import evaluate_production_gate, required_pareto_min


def test_required_pareto_min_balanced_profile() -> None:
    assert required_pareto_min(0) == 8
    assert required_pareto_min(10) == 8
    assert required_pareto_min(100) == 12


def test_production_gate_passes_when_all_balanced_checks_pass() -> None:
    payload = evaluate_production_gate(
        n_pareto=12,
        effective_pop=100,
        feasible_fraction=0.35,
        n_eval_errors=0,
        winner_present=True,
        frontier_gate_pass=True,
        frontier_gate_basis="hard_feasible",
        release_ready=True,
        used_heuristic_fallback=False,
        algorithm_used="nsga3",
        fidelity=2,
        constraint_phase="downselect",
    )
    assert payload["production_gate_pass"] is True
    assert payload["production_gate_failures"] == []


def test_production_gate_nonproduction_override_forces_nonrelease() -> None:
    payload = evaluate_production_gate(
        allow_nonproduction_paths=True,
        n_pareto=20,
        effective_pop=100,
        feasible_fraction=0.5,
        n_eval_errors=0,
        winner_present=True,
        frontier_gate_pass=True,
        frontier_gate_basis="hard_feasible",
        release_ready=True,
        used_heuristic_fallback=False,
        algorithm_used="nsga3",
        fidelity=2,
        constraint_phase="downselect",
    )
    assert payload["production_gate_pass"] is False
    assert "nonproduction_paths_enabled" in payload["production_gate_failures"]


def test_production_gate_disallows_placeholder_frontier_basis() -> None:
    payload = evaluate_production_gate(
        n_pareto=12,
        effective_pop=100,
        feasible_fraction=0.3,
        n_eval_errors=0,
        winner_present=True,
        frontier_gate_pass=True,
        frontier_gate_basis="placeholder_frontier",
        release_ready=True,
        used_heuristic_fallback=False,
        algorithm_used="slice_refine",
        fidelity=2,
        constraint_phase="downselect",
    )
    assert payload["production_gate_pass"] is False
    assert "placeholder_frontier_basis_disallowed" in payload["production_gate_failures"]
