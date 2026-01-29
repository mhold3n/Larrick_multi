"""PyMoo adapter for multi-objective optimization.

This module wraps the evaluate_candidate interface for use with pymoo.
"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from ..core.encoding import N_TOTAL, bounds
from ..core.evaluator import evaluate_candidate
from ..core.types import EvalContext


class ParetoProblem(Problem):
    """PyMoo Problem wrapper for larrak2 evaluation.

    Uses evaluate_candidate as the underlying evaluation function.
    """

    # Number of objectives (efficiency, loss, max_planet_radius)
    N_OBJ = 3

    # Number of constraints (3 thermo + 4 gear = 7)
    N_CONSTR = 7

    def __init__(
        self,
        ctx: EvalContext,
        **kwargs,
    ) -> None:
        """Initialize Pareto problem.

        Args:
            ctx: Evaluation context (rpm, torque, fidelity, seed).
            **kwargs: Additional arguments passed to pymoo Problem.
        """
        xl, xu = bounds()

        super().__init__(
            n_var=N_TOTAL,
            n_obj=self.N_OBJ,
            n_ieq_constr=self.N_CONSTR,
            xl=xl,
            xu=xu,
            **kwargs,
        )

        self.ctx = ctx
        self._n_evals = 0

    def _evaluate(
        self,
        X: np.ndarray,
        out: dict,
        *args,
        **kwargs,
    ) -> None:
        """Evaluate population.

        Args:
            X: Decision matrix of shape (pop_size, n_var).
            out: Output dict for F and G.
        """
        n_pop = X.shape[0]
        F = np.zeros((n_pop, self.N_OBJ), dtype=np.float64)
        G = np.zeros((n_pop, self.N_CONSTR), dtype=np.float64)

        for i, x in enumerate(X):
            result = evaluate_candidate(x, self.ctx)
            F[i] = result.F
            G[i] = result.G
            self._n_evals += 1

        out["F"] = F
        out["G"] = G

    @property
    def n_evals(self) -> int:
        """Total number of evaluations performed."""
        return self._n_evals


def create_problem(
    rpm: float = 3000.0,
    torque: float = 200.0,
    fidelity: int = 0,
    seed: int = 42,
) -> ParetoProblem:
    """Create ParetoProblem with given context.

    Args:
        rpm: Engine speed (rpm).
        torque: Torque demand (Nm).
        fidelity: Model fidelity level.
        seed: Random seed.

    Returns:
        Configured ParetoProblem.
    """
    ctx = EvalContext(rpm=rpm, torque=torque, fidelity=fidelity, seed=seed)
    return ParetoProblem(ctx=ctx)
