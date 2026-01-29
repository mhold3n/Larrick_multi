"""Combustion model placeholder.

This module will hold more detailed combustion physics.
Currently delegates to the toy Wiebe model in motionlaw.py.
"""

from __future__ import annotations

import numpy as np


def wiebe_burn_fraction(
    theta: np.ndarray,
    theta_start: float,
    duration: float,
    a: float = 5.0,
    m: float = 2.0,
) -> np.ndarray:
    """Compute cumulative mass fraction burned using Wiebe function.

    Args:
        theta: Crank angle array (degrees).
        theta_start: Start of combustion (degrees).
        duration: Burn duration (degrees).
        a: Wiebe parameter (typically 5).
        m: Shape parameter (typically 2).

    Returns:
        Mass fraction burned in [0, 1].
    """
    x_b = np.zeros_like(theta, dtype=np.float64)

    mask = theta >= theta_start
    if not np.any(mask):
        return x_b

    xi = np.clip((theta[mask] - theta_start) / duration, 0.0, 1.0)
    x_b[mask] = 1.0 - np.exp(-a * xi ** (m + 1))

    return x_b
