"""Slice NLP bridges.

Deprecated:
    - Linearized local slice solver (explicit legacy call only)

Canonical:
    - Global-surrogate symbolic nonlinear CasADi NLP solver
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.encoding import N_TOTAL
from ...core.types import EvalContext
from .symbolic_slice_problem import solve_symbolic_slice_with_ipopt


@dataclass
class SliceSolveResult:
    """Result of solving a local active-variable slice."""

    x_opt: np.ndarray
    success: bool
    message: str
    ipopt_status: str
    iterations: int
    diagnostics: dict[str, Any]


def solve_slice_with_ipopt(
    x0: np.ndarray,
    ctx: EvalContext,
    active_indices: list[int],
    *,
    surrogate_stack_path: str,
    mode: str = "eps_constraint",
    weights: np.ndarray | None = None,
    eps_constraints: np.ndarray | None = None,
    ipopt_options: dict[str, Any] | None = None,
    regularization: float = 1e-2,
    trust_radius: float | None = None,
    validation_attempts: int = 3,
    validation_tol: float = 1e-8,
    fidelity: int = 1,
) -> SliceSolveResult:
    """Solve nonlinear symbolic slice NLP via global surrogate stack."""
    res = solve_symbolic_slice_with_ipopt(
        x0,
        ctx,
        active_indices,
        surrogate_stack_path=surrogate_stack_path,
        mode=mode,
        weights=weights,
        eps_constraints=eps_constraints,
        ipopt_options=ipopt_options,
        regularization=regularization,
        trust_radius=trust_radius,
        validation_attempts=validation_attempts,
        validation_tol=validation_tol,
        fidelity=fidelity,
    )
    return SliceSolveResult(
        x_opt=res.x_opt,
        success=res.success,
        message=res.message,
        ipopt_status=res.ipopt_status,
        iterations=res.iterations,
        diagnostics=res.diagnostics,
    )


def solve_slice_with_ipopt_linearized(
    x0: np.ndarray,
    _ctx: EvalContext,
    _active_indices: list[int],
    *,
    mode: str = "weighted_sum",
    weights: np.ndarray | None = None,
    eps_constraints: np.ndarray | None = None,
    ipopt_options: dict[str, Any] | None = None,
    regularization: float = 1e-2,
    trust_radius: float | None = None,
) -> SliceSolveResult:
    """Deprecated linearized solver entrypoint.

    Kept only to avoid import/runtime breaks for explicit legacy calls.
    """
    _ = (
        mode,
        weights,
        eps_constraints,
        ipopt_options,
        regularization,
        trust_radius,
    )
    x = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x.size != N_TOTAL:
        return SliceSolveResult(
            x_opt=x,
            success=False,
            message=f"Expected x0 length {N_TOTAL}, got {x.size}",
            ipopt_status="invalid_x0",
            iterations=0,
            diagnostics={"nlp_formulation": "linearized_deprecated"},
        )
    return SliceSolveResult(
        x_opt=x,
        success=False,
        message="Linearized slice solver is deprecated. Use solve_slice_with_ipopt symbolic path.",
        ipopt_status="deprecated_linearized_solver",
        iterations=0,
        diagnostics={"nlp_formulation": "linearized_deprecated"},
    )
