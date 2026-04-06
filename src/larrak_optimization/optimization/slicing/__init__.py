"""Variable slicing utilities for high-dimensional refinement."""

from .active_set import SliceSelection, select_active_set, sensitivity_scores
from .slice_problem import (
    SliceSolveResult,
    solve_slice_with_ipopt,
    solve_slice_with_ipopt_linearized,
)
from .symbolic_slice_problem import SymbolicSliceSolveResult, solve_symbolic_slice_with_ipopt

__all__ = [
    "SliceSelection",
    "SliceSolveResult",
    "SymbolicSliceSolveResult",
    "select_active_set",
    "sensitivity_scores",
    "solve_slice_with_ipopt",
    "solve_slice_with_ipopt_linearized",
    "solve_symbolic_slice_with_ipopt",
]
