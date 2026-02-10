"""CasADi refinement module.

Provides local refinement of Pareto candidates using gradient-based
scalarization methods (weighted sum, ε-constraint).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ..core.encoding import bounds
from ..core.evaluator import evaluate_candidate
from ..core.types import EvalContext


class RefinementMode(str, Enum):
    """Refinement strategy."""

    WEIGHTED_SUM = "weighted_sum"
    EPS_CONSTRAINT = "eps_constraint"


@dataclass
class RefinementResult:
    """Result from candidate refinement."""

    x_refined: np.ndarray
    F_refined: np.ndarray
    G_refined: np.ndarray
    diag: dict[str, Any]
    success: bool
    message: str


def refine_candidate(
    x0: np.ndarray,
    ctx: EvalContext,
    mode: RefinementMode = RefinementMode.WEIGHTED_SUM,
    weights: np.ndarray | None = None,
    eps_constraints: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> RefinementResult:
    """Refine candidate using local gradient-based optimization.

    Args:
        x0: Initial candidate.
        ctx: Evaluation context.
        mode: Refinement strategy.
        weights: Objective weights for weighted sum (length n_obj).
        eps_constraints: Upper bounds on objectives for ε-constraint.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        RefinementResult with refined candidate.

    Note:
        Currently uses finite-difference gradients as CasADi graph
        is not yet implemented. Interface is stable.
    """
    xl, xu = bounds()

    # Initial evaluation
    result0 = evaluate_candidate(x0, ctx)

    if mode == RefinementMode.WEIGHTED_SUM:
        w = weights if weights is not None else np.ones(len(result0.F))
        x_refined, diag = _weighted_sum_refine(x0, ctx, w, xl, xu, max_iter, tol)
    elif mode == RefinementMode.EPS_CONSTRAINT:
        x_refined, diag = _eps_constraint_refine(
            x0, ctx, eps_constraints or result0.F, xl, xu, max_iter, tol
        )
    else:
        return RefinementResult(
            x_refined=x0,
            F_refined=result0.F,
            G_refined=result0.G,
            diag={},
            success=False,
            message=f"Unknown mode: {mode}",
        )

    # Final evaluation
    result_final = evaluate_candidate(x_refined, ctx)

    return RefinementResult(
        x_refined=x_refined,
        F_refined=result_final.F,
        G_refined=result_final.G,
        diag=diag,
        success=True,
        message="Refinement completed",
    )


def _weighted_sum_refine(
    x0: np.ndarray,
    ctx: EvalContext,
    weights: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict]:
    """Weighted sum scalarization refinement.

    Uses scipy.optimize.minimize with finite-difference gradients.
    """
    try:
        from scipy.optimize import minimize

        def objective(x):
            # Bounds safety
            x_clamped = np.clip(x, xl, xu)
            result = evaluate_candidate(x_clamped, ctx)
            # Scalarized objective
            return float(np.dot(weights, result.F))

        def constraint_func(x):
            x_clamped = np.clip(x, xl, xu)
            result = evaluate_candidate(x_clamped, ctx)
            # Return -G for scipy (constraint >= 0 expected)
            return -result.G

        # Use dict-based constraints for SLSQP

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=list(zip(xl, xu)),
            constraints={"type": "ineq", "fun": constraint_func},
            options={"maxiter": max_iter, "ftol": tol, "disp": False},
        )

        x_final = np.clip(res.x, xl, xu)

        # Simple backtracking if NaN or violated
        if (not np.all(np.isfinite(x_final))) or (np.any(x_final < xl) or np.any(x_final > xu)):
            x_final = np.clip(x0, xl, xu)

        return x_final, {"scipy_result": str(res.message), "n_iter": res.nit}

    except ImportError:
        # Fallback: return original
        return x0, {"error": "scipy not available"}


def _eps_constraint_refine(
    x0: np.ndarray,
    ctx: EvalContext,
    eps: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict]:
    """ε-constraint method refinement.

    Minimize first objective while constraining others to <= eps.
    """
    try:
        from scipy.optimize import minimize

        def objective(x):
            x_clamped = np.clip(x, xl, xu)
            result = evaluate_candidate(x_clamped, ctx)
            return float(result.F[0])  # Minimize first objective

        def constraint_func(x):
            x_clamped = np.clip(x, xl, xu)
            result = evaluate_candidate(x_clamped, ctx)
            # Original constraints + ε-constraints on objectives 1..n
            eps_g = result.F[1:] - eps[1:]  # F[i] <= eps[i]
            all_g = np.concatenate([-result.G, -eps_g])
            return all_g

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=list(zip(xl, xu)),
            constraints={"type": "ineq", "fun": constraint_func},
            options={"maxiter": max_iter, "ftol": tol, "disp": False},
        )

        x_final = np.clip(res.x, xl, xu)
        if not np.all(np.isfinite(x_final)):
            x_final = np.clip(x0, xl, xu)

        return x_final, {"scipy_result": str(res.message), "n_iter": res.nit}

    except ImportError:
        return x0, {"error": "scipy not available"}
