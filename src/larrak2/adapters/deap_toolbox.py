"""DEAP adapter placeholder.

This module provides a minimal placeholder for DEAP integration.
Only implemented if pymoo proves insufficient.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.types import EvalContext


def make_deap_toolbox(
    ctx: EvalContext,
    n_obj: int = 2,
) -> Any:
    """Create DEAP toolbox for multi-objective optimization.

    Args:
        ctx: Evaluation context.
        n_obj: Number of objectives.

    Returns:
        Configured DEAP toolbox.

    Raises:
        NotImplementedError: DEAP integration not yet implemented.
    """
    raise NotImplementedError(
        "DEAP adapter not yet implemented. "
        "Use pymoo via adapters.pymoo_problem.ParetoProblem instead."
    )


def deap_evaluate_wrapper(
    ctx: EvalContext,
) -> Callable[[list[float]], tuple[float, ...]]:
    """Create DEAP-compatible evaluation function.

    Args:
        ctx: Evaluation context.

    Returns:
        Function that takes individual and returns fitness tuple.

    Raises:
        NotImplementedError: DEAP integration not yet implemented.
    """
    raise NotImplementedError("DEAP evaluate wrapper not yet implemented.")
