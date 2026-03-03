"""Multi-start helper for orchestration runs."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from .orchestrator import OrchestrationResult, Orchestrator

LOGGER = logging.getLogger(__name__)


def _perturb_numeric(
    values: dict[str, Any], rng: np.random.Generator, scale: float
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            out[key] = _perturb_numeric(value, rng, scale)
        elif isinstance(value, (int, float)):
            baseline = float(value)
            delta = float(rng.normal(0.0, 1.0)) * scale * max(abs(baseline), 1e-6)
            out[key] = baseline + delta
        else:
            out[key] = value
    return out


def optimize_with_multistart(
    orchestrator: Orchestrator,
    initial_params: dict[str, Any],
    n_starts: int = 3,
    perturbation_scale: float = 0.1,
    seed: int = 42,
) -> OrchestrationResult:
    """Run orchestration with multiple perturbed starts."""
    n_starts = max(1, int(n_starts))
    rng = np.random.default_rng(int(seed))
    results: list[OrchestrationResult] = []

    for attempt in range(n_starts):
        params = copy.deepcopy(initial_params)
        if attempt > 0:
            scale = float(perturbation_scale) * (1.0 + 0.5 * attempt)
            params = _perturb_numeric(params, rng, scale)
            LOGGER.info(
                "Multi-start attempt %d/%d perturbation_scale=%.4f", attempt + 1, n_starts, scale
            )

        result = orchestrator.optimize(initial_params=params)
        results.append(result)
        if _is_acceptable_result(result):
            return result

    return max(results, key=lambda r: float(r.best_objective))


def _is_acceptable_result(result: OrchestrationResult) -> bool:
    if not np.isfinite(float(result.best_objective)):
        return False
    if int(result.n_iterations) <= 0:
        return False
    if int(result.n_surrogate_calls) <= 0:
        return False
    return True


__all__ = ["optimize_with_multistart"]
