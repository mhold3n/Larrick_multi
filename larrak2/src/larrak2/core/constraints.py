"""Constraint utilities.

Helpers for scaling, concatenating, and checking constraint feasibility.
All constraints follow the convention: G <= 0 is feasible.
"""

from __future__ import annotations

import numpy as np


def scale_constraint(g: float, scale: float = 1.0) -> float:
    """Scale a constraint value.

    Args:
        g: Raw constraint value.
        scale: Scaling factor (default 1.0).

    Returns:
        Scaled constraint value.
    """
    return g / scale


def concat_constraints(*constraints: np.ndarray) -> np.ndarray:
    """Concatenate multiple constraint arrays.

    Args:
        constraints: Variable number of constraint arrays.

    Returns:
        Single concatenated constraint array.
    """
    if not constraints:
        return np.array([], dtype=np.float64)
    return np.concatenate(constraints)


def is_feasible(G: np.ndarray, tol: float = 0.0) -> bool:
    """Check if all constraints are satisfied.

    Args:
        G: Constraint array.
        tol: Tolerance for feasibility (default 0).

    Returns:
        True if all G <= tol.
    """
    return bool(np.all(G <= tol))


def max_violation(G: np.ndarray) -> float:
    """Compute maximum constraint violation.

    Args:
        G: Constraint array.

    Returns:
        Maximum violation (0 if feasible).
    """
    if len(G) == 0:
        return 0.0
    return float(np.maximum(G, 0).max())


def feasibility_ratio(G: np.ndarray, tol: float = 0.0) -> float:
    """Compute fraction of constraints satisfied.

    Args:
        G: Constraint array.
        tol: Tolerance for satisfaction.

    Returns:
        Ratio in [0, 1] of satisfied constraints.
    """
    if len(G) == 0:
        return 1.0
    return float(np.mean(G <= tol))


def bound_constraint(value: float, lower: float, upper: float) -> tuple[float, float]:
    """Create bound constraints for a value.

    Returns (g_lower, g_upper) where:
        g_lower = lower - value  (<=0 if value >= lower)
        g_upper = value - upper  (<=0 if value <= upper)

    Args:
        value: Value to constrain.
        lower: Lower bound.
        upper: Upper bound.

    Returns:
        Tuple of (g_lower, g_upper) constraints.
    """
    g_lower = lower - value
    g_upper = value - upper
    return g_lower, g_upper
