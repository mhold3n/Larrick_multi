"""Shared production gate checks for multi-objective optimization workflows."""

from __future__ import annotations

import math
from typing import Any

STRICT_PRODUCTION_PROFILE = "strict_prod"
BALANCED_THRESHOLD_PROFILE = "balanced_v1"


def required_pareto_min(effective_pop: int) -> int:
    """Return minimum Pareto cardinality for the balanced production profile."""
    return int(max(8, math.ceil(0.12 * max(0, int(effective_pop)))))


def evaluate_production_gate(
    *,
    production_profile: str = STRICT_PRODUCTION_PROFILE,
    allow_nonproduction_paths: bool = False,
    fallback_paths_used: list[str] | None = None,
    nonproduction_overrides: list[str] | None = None,
    n_pareto: int | None = None,
    effective_pop: int | None = None,
    feasible_fraction: float | None = None,
    n_eval_errors: int | None = None,
    winner_present: bool | None = None,
    frontier_gate_pass: bool | None = None,
    frontier_gate_basis: str | None = None,
    release_ready: bool | None = None,
    used_heuristic_fallback: bool | None = None,
    algorithm_used: str = "",
    fidelity: int | None = None,
    constraint_phase: str = "",
) -> dict[str, Any]:
    """Evaluate strict production-readiness checks and return diagnostics payload."""
    fallback_paths = [str(v) for v in (fallback_paths_used or []) if str(v).strip()]
    overrides = [str(v) for v in (nonproduction_overrides or []) if str(v).strip()]
    failures: list[str] = []

    eff_pop_i = int(effective_pop) if effective_pop is not None else None
    n_pareto_i = int(n_pareto) if n_pareto is not None else None
    n_eval_errors_i = int(max(0, int(n_eval_errors or 0)))
    feasible_fraction_f = (
        float(feasible_fraction) if feasible_fraction is not None else None
    )
    phase = str(constraint_phase).strip()
    profile = str(production_profile).strip() or STRICT_PRODUCTION_PROFILE
    fidelity_i = int(fidelity) if fidelity is not None else None

    pareto_min_required: int | None = None
    if eff_pop_i is not None:
        pareto_min_required = required_pareto_min(eff_pop_i)

    # Strict production profile assumes release-ready runs only.
    if phase and phase != "downselect":
        failures.append("constraint_phase_not_downselect")
    if fidelity_i is not None and fidelity_i < 2:
        failures.append("fidelity_below_production")

    if n_eval_errors_i != 0:
        failures.append("n_eval_errors_nonzero")

    if (
        n_pareto_i is not None
        and pareto_min_required is not None
        and n_pareto_i < pareto_min_required
    ):
        failures.append("n_pareto_below_min")
    if feasible_fraction_f is not None and feasible_fraction_f < 0.20:
        failures.append("feasible_fraction_below_min")

    if winner_present is not None and not bool(winner_present):
        failures.append("no_hard_feasible_winner")
    if frontier_gate_pass is not None and not bool(frontier_gate_pass):
        failures.append("principles_frontier_gate_failed")
    if str(frontier_gate_basis or "").strip() == "placeholder_frontier":
        failures.append("placeholder_frontier_basis_disallowed")
    if release_ready is not None and not bool(release_ready):
        failures.append("release_readiness_false")
    if used_heuristic_fallback is not None and bool(used_heuristic_fallback):
        failures.append("heuristic_fallback_used")

    if bool(allow_nonproduction_paths):
        if "allow_nonproduction_paths" not in overrides:
            overrides.append("allow_nonproduction_paths")
        failures.append("nonproduction_paths_enabled")

    # Preserve deterministic ordering for stable manifests/tests.
    failures = sorted(set(failures))
    overrides = sorted(set(overrides))
    fallback_paths = sorted(set(fallback_paths))

    return {
        "production_profile": profile,
        "threshold_profile": BALANCED_THRESHOLD_PROFILE,
        "production_gate_pass": len(failures) == 0,
        "production_gate_failures": failures,
        "fallback_paths_used": fallback_paths,
        "nonproduction_overrides": overrides,
        "n_eval_errors": n_eval_errors_i,
        "algorithm_used": str(algorithm_used or ""),
        "fidelity": fidelity_i,
        "constraint_phase": phase,
        "thresholds": {
            "n_pareto_min": pareto_min_required,
            "feasible_fraction_min": 0.20,
            "n_eval_errors_required": 0,
            "winner_required": True,
            "frontier_gate_required": True,
            "placeholder_frontier_allowed": False,
            "release_ready_required": True,
            "heuristic_fallback_allowed": False,
            "constraint_phase_required": "downselect",
            "min_fidelity": 2,
        },
        "checked_metrics": {
            "n_pareto": n_pareto_i,
            "effective_pop": eff_pop_i,
            "feasible_fraction": feasible_fraction_f,
            "winner_present": None if winner_present is None else bool(winner_present),
            "frontier_gate_pass": (
                None if frontier_gate_pass is None else bool(frontier_gate_pass)
            ),
            "frontier_gate_basis": str(frontier_gate_basis or ""),
            "release_ready": None if release_ready is None else bool(release_ready),
            "used_heuristic_fallback": (
                None
                if used_heuristic_fallback is None
                else bool(used_heuristic_fallback)
            ),
        },
    }


__all__ = [
    "BALANCED_THRESHOLD_PROFILE",
    "STRICT_PRODUCTION_PROFILE",
    "evaluate_production_gate",
    "required_pareto_min",
]
