"""Variable scaling helpers inspired by the legacy stack."""

from __future__ import annotations

import numpy as np


def compute_variable_scaling(
    lbx: np.ndarray,
    ubx: np.ndarray,
    *,
    min_scale: float = 1e-8,
    max_scale: float = 1e8,
) -> np.ndarray:
    """Compute diagonal variable scaling from variable bounds."""
    lbx = np.asarray(lbx, dtype=np.float64)
    ubx = np.asarray(ubx, dtype=np.float64)
    span = np.abs(ubx - lbx)
    span[~np.isfinite(span)] = 1.0
    span = np.maximum(span, min_scale)
    scale = 1.0 / span
    return np.clip(scale, min_scale, max_scale)
