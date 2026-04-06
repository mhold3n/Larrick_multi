"""Expansion and compression relations.

Polytropic process utilities for p-V-T calculations.
"""

from __future__ import annotations

import numpy as np


def polytropic_pressure(
    v2: float | np.ndarray,
    v1: float,
    p1: float,
    n: float,
) -> float | np.ndarray:
    """Compute pressure from polytropic relation p*V^n = const.

    Args:
        v2: Final volume(s).
        v1: Initial volume.
        p1: Initial pressure.
        n: Polytropic exponent.

    Returns:
        Final pressure(s).
    """
    return p1 * (v1 / v2) ** n


def polytropic_temperature(
    v2: float | np.ndarray,
    v1: float,
    t1: float,
    n: float,
) -> float | np.ndarray:
    """Compute temperature from polytropic relation T*V^(n-1) = const.

    Args:
        v2: Final volume(s).
        v1: Initial volume.
        t1: Initial temperature.
        n: Polytropic exponent.

    Returns:
        Final temperature(s).
    """
    return t1 * (v1 / v2) ** (n - 1)


def isentropic_work(p1: float, v1: float, p2: float, v2: float, gamma: float) -> float:
    """Compute work for isentropic process.

    W = (p1*v1 - p2*v2) / (gamma - 1)

    Args:
        p1: Initial pressure.
        v1: Initial volume.
        p2: Final pressure.
        v2: Final volume.
        gamma: Ratio of specific heats.

    Returns:
        Work (positive for expansion).
    """
    return (p1 * v1 - p2 * v2) / (gamma - 1)


def compression_ratio_from_volumes(v_max: float, v_min: float) -> float:
    """Compute compression ratio from volume extremes.

    Args:
        v_max: Maximum volume (BDC).
        v_min: Minimum volume (TDC clearance).

    Returns:
        Compression ratio r_c = V_max / V_min.
    """
    return v_max / v_min


def ideal_otto_efficiency(r_c: float, gamma: float = 1.4) -> float:
    """Compute ideal Otto cycle efficiency.

    η = 1 - (1/r_c)^(γ-1)

    Args:
        r_c: Compression ratio.
        gamma: Ratio of specific heats.

    Returns:
        Ideal thermal efficiency.
    """
    return 1.0 - (1.0 / r_c) ** (gamma - 1)
