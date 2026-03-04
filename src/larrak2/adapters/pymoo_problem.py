"""PyMoo adapter for multi-objective optimization.

This module wraps the evaluate_candidate interface for use with pymoo.
"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from ..core.constraints import get_constraint_names
from ..core.encoding import N_TOTAL, bounds
from ..core.evaluator import evaluate_candidate
from ..core.types import EvalContext


class ParetoProblem(Problem):
    """PyMoo Problem wrapper for larrak2 evaluation.

    Uses evaluate_candidate as the underlying evaluation function.
    """

    # Objective count is inferred from canonical evaluator output.
    N_OBJ = 0

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

        self.N_CONSTR = len(get_constraint_names(ctx.fidelity))

        # Infer objective dimensionality from the canonical evaluator.
        x_probe = (np.asarray(xl, dtype=np.float64) + np.asarray(xu, dtype=np.float64)) * 0.5
        probe = evaluate_candidate(x_probe, ctx)
        self.N_OBJ = int(probe.F.size)
        if self.N_OBJ <= 0:
            raise ValueError("evaluate_candidate returned no objectives")

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
        self._n_eval_errors = 0
        self._eval_error_signatures: dict[str, int] = {}
        self._penalty_F = np.full(self.N_OBJ, 1.0e6, dtype=np.float64)
        self._penalty_G = np.full(self.N_CONSTR, 1.0e3, dtype=np.float64)

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
            try:
                result = evaluate_candidate(x, self.ctx)
                F[i] = result.F
                G[i] = result.G
            except Exception as exc:
                F[i] = self._penalty_F
                G[i] = self._penalty_G
                self._n_eval_errors += 1
                signature = f"{type(exc).__name__}:{str(exc).strip()}"
                self._eval_error_signatures[signature] = (
                    int(self._eval_error_signatures.get(signature, 0)) + 1
                )
            self._n_evals += 1

        out["F"] = F
        out["G"] = G

    @property
    def n_evals(self) -> int:
        """Total number of evaluations performed."""
        return self._n_evals

    @property
    def n_eval_errors(self) -> int:
        """Total number of candidate evaluations handled via deterministic penalty."""
        return int(self._n_eval_errors)

    @property
    def eval_error_signatures(self) -> dict[str, int]:
        """Counted evaluation error signatures observed while scoring populations."""
        return dict(self._eval_error_signatures)


def create_problem(
    rpm: float = 3000.0,
    torque: float = 200.0,
    fidelity: int = 2,
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
