"""Solver adapter using CasADi/Ipopt-first slice refinement."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from larrak2.adapters.casadi_refine import RefinementMode, refine_candidate
from larrak2.core.encoding import N_TOTAL
from larrak2.core.types import EvalContext

LOGGER = logging.getLogger(__name__)


class CasadiSolverAdapter:
    """Uses `larrak2.adapters.casadi_refine.refine_candidate` for local solve steps."""

    def __init__(
        self,
        *,
        backend: str = "casadi",
        mode: str = "weighted_sum",
        weights: np.ndarray | None = None,
        eps_constraints: np.ndarray | None = None,
        active_set: list[int] | None = None,
        ipopt_options: dict[str, Any] | None = None,
        slice_method: str = "sensitivity",
        active_k: int | None = None,
        min_per_group: int = 1,
    ) -> None:
        self.backend = str(backend)
        self.mode = str(mode)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.eps_constraints = (
            None if eps_constraints is None else np.asarray(eps_constraints, dtype=np.float64)
        )
        self.active_set = list(active_set) if active_set is not None else None
        self.ipopt_options = dict(ipopt_options or {})
        self.slice_method = str(slice_method)
        self.active_k = active_k
        self.min_per_group = int(min_per_group)

    def refine(
        self,
        candidate: dict[str, Any],
        *,
        context: EvalContext,
        max_step: np.ndarray,
    ) -> dict[str, Any]:
        updated = dict(candidate)
        x0 = np.asarray(candidate.get("x", []), dtype=np.float64).reshape(-1)
        if x0.size != N_TOTAL:
            updated["solver_error"] = f"expected x length {N_TOTAL}, got {x0.size}"
            return updated

        freeze_mask = None
        if max_step is not None:
            step = np.asarray(max_step, dtype=np.float64).reshape(-1)
            if step.size == N_TOTAL:
                freeze_mask = step <= 1e-12

        try:
            result = refine_candidate(
                x0=x0,
                ctx=context,
                mode=RefinementMode(self.mode),
                weights=self.weights,
                eps_constraints=self.eps_constraints,
                backend=self.backend,
                active_set=self.active_set,
                ipopt_options=self.ipopt_options,
                freeze_mask=freeze_mask,
                active_k=self.active_k,
                min_per_group=self.min_per_group,
                slice_method=self.slice_method,
            )
            updated["x"] = np.asarray(result.x_refined, dtype=np.float64)
            updated["solver_success"] = bool(result.success)
            updated["solver_backend"] = str(result.backend_used)
            updated["ipopt_status"] = str(result.ipopt_status)
            updated["solver_message"] = str(result.message)
            updated["solver_diag"] = result.diag
        except Exception as exc:
            LOGGER.warning("CasadiSolverAdapter refinement failed: %s", exc)
            updated["solver_success"] = False
            updated["solver_error"] = str(exc)
        return updated


class SimpleSolverAdapter:
    """Lightweight random local search fallback adapter."""

    def __init__(self, *, n_evals: int = 8, step_scale: float = 0.05, seed: int = 42) -> None:
        self.n_evals = int(max(1, n_evals))
        self.step_scale = float(max(1e-6, step_scale))
        self._rng = np.random.default_rng(int(seed))

    def refine(
        self,
        candidate: dict[str, Any],
        *,
        context: EvalContext,  # noqa: ARG002
        max_step: np.ndarray,
    ) -> dict[str, Any]:
        trial = dict(candidate)
        x0 = np.asarray(candidate.get("x", []), dtype=np.float64).reshape(-1)
        if x0.size != N_TOTAL:
            return trial

        step = np.asarray(max_step, dtype=np.float64).reshape(-1)
        if step.size != N_TOTAL:
            step = np.ones(N_TOTAL, dtype=np.float64) * self.step_scale

        best = x0.copy()
        best_score = -float(np.linalg.norm(best))
        for _ in range(self.n_evals):
            perturb = self._rng.uniform(-1.0, 1.0, size=N_TOTAL) * np.maximum(step, self.step_scale)
            cand = x0 + perturb
            score = -float(np.linalg.norm(cand))
            if score > best_score:
                best = cand
                best_score = score
        trial["x"] = best
        trial["solver_success"] = True
        trial["solver_backend"] = "simple_random_search"
        return trial


__all__ = [
    "CasadiSolverAdapter",
    "SimpleSolverAdapter",
]
