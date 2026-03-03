"""Thermo forward interface and ratio-profile helpers.

This module keeps the stable `eval_thermo` API used by evaluators while routing
all thermodynamic physics to the strict equation-first two-zone backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.encoding import ThermoParams
from ..core.types import EvalContext
from .two_zone import evaluate_two_zone_thermo


@dataclass
class ThermoResult:
    """Result from thermodynamic evaluation."""

    efficiency: float
    requested_ratio_profile: np.ndarray
    G: np.ndarray
    diag: dict[str, Any] = field(default_factory=dict)


def _ratio_profile_stats(profile: np.ndarray) -> dict[str, float | bool]:
    """Compute continuity/quality metrics for ratio profile."""
    profile = np.asarray(profile, dtype=np.float64).reshape(-1)
    finite = bool(np.all(np.isfinite(profile)))
    min_val = float(np.min(profile)) if profile.size else 0.0
    max_val = float(np.max(profile)) if profile.size else 0.0
    slope = np.gradient(profile) if profile.size > 1 else np.array([0.0], dtype=np.float64)
    jerk = np.gradient(slope) if slope.size > 1 else np.array([0.0], dtype=np.float64)
    return {
        "finite": finite,
        "min": min_val,
        "max": max_val,
        "max_slope": float(np.max(np.abs(slope))) if slope.size else 0.0,
        "max_jerk": float(np.max(np.abs(jerk))) if jerk.size else 0.0,
    }


def eval_thermo(
    params: ThermoParams,
    ctx: EvalContext,
    ratio_slope_limit: float | None = None,
) -> ThermoResult:
    """Evaluate thermodynamics via strict two-zone equation-first backend."""
    if str(ctx.thermo_model) != "two_zone_eq_v1":
        raise ValueError(
            f"Unsupported thermo_model='{ctx.thermo_model}'. "
            "Only 'two_zone_eq_v1' is permitted in strict mode."
        )

    res = evaluate_two_zone_thermo(params, ctx, ratio_slope_limit=ratio_slope_limit)
    return ThermoResult(
        efficiency=float(res.efficiency),
        requested_ratio_profile=np.asarray(res.requested_ratio_profile, dtype=np.float64),
        G=np.asarray(res.G, dtype=np.float64),
        diag=dict(res.diag),
    )
