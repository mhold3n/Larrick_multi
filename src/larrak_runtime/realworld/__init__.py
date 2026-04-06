from __future__ import annotations

"""Optimization-loop real-world surrogates.

Lightweight surrogate representations of material, surface finish,
lubrication, and coating decisions used inside the pymoo evaluation loop.

Instead of scalar placeholders, the optimizer works with continuous
levels (0–1) that map to CEM enum tiers.  The surrogates produce
ordered feature-importance rankings and feasibility estimates without
calling the full CEM.
"""

from larrak_runtime.realworld.constraints import compute_realworld_constraints  # noqa: E402
from larrak_runtime.realworld.surrogates import (  # noqa: E402
    DEFAULT_REALWORLD_PARAMS,
    RealWorldSurrogateParams,
    evaluate_realworld_surrogates,
)

__all__ = [
    "DEFAULT_REALWORLD_PARAMS",
    "RealWorldSurrogateParams",
    "evaluate_realworld_surrogates",
    "compute_realworld_constraints",
]
