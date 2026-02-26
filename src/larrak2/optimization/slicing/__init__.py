"""Variable slicing utilities for high-dimensional refinement."""

from .active_set import SliceSelection, select_active_set, sensitivity_scores
from .slice_problem import SliceSolveResult, solve_slice_with_ipopt

__all__ = [
    "SliceSelection",
    "SliceSolveResult",
    "select_active_set",
    "sensitivity_scores",
    "solve_slice_with_ipopt",
]
