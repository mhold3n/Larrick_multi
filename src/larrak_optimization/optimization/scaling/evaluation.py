"""Quality metrics for scaling vectors."""

from __future__ import annotations

import numpy as np


def scaling_quality(scale: np.ndarray) -> dict[str, float]:
    """Return coarse quality metrics for a scale vector."""
    scale = np.asarray(scale, dtype=np.float64)
    finite = scale[np.isfinite(scale)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "condition": float("inf")}

    smin = float(np.min(finite))
    smax = float(np.max(finite))
    cond = float(smax / max(smin, 1e-16))
    return {"min": smin, "max": smax, "condition": cond}
