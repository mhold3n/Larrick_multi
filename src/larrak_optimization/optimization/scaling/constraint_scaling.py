"""Constraint scaling helpers inspired by the legacy stack."""

from __future__ import annotations

import numpy as np


def compute_constraint_scaling(
    lbg: np.ndarray,
    ubg: np.ndarray,
    *,
    min_scale: float = 1e-8,
    max_scale: float = 1e8,
) -> np.ndarray:
    """Compute per-constraint scaling factors from bound magnitudes."""
    lbg = np.asarray(lbg, dtype=np.float64)
    ubg = np.asarray(ubg, dtype=np.float64)

    lo = np.where(np.isfinite(lbg), np.abs(lbg), 0.0)
    hi = np.where(np.isfinite(ubg), np.abs(ubg), 0.0)
    mag = np.maximum(lo, hi)
    mag = np.maximum(mag, 1.0)

    scale = 1.0 / mag
    return np.clip(scale, min_scale, max_scale)
