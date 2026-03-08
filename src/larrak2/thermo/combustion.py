"""Combustion closures for equation-first two-zone thermo."""

from __future__ import annotations

import numpy as np


def wiebe_burn_fraction(
    theta: np.ndarray,
    theta_start: float,
    duration: float,
    a: float = 5.0,
    m: float = 2.0,
) -> np.ndarray:
    """Single Wiebe cumulative burn fraction in [0,1]."""
    x_b = np.zeros_like(theta, dtype=np.float64)
    if duration <= 1e-9:
        return x_b

    mask = theta >= theta_start
    if not np.any(mask):
        return x_b

    xi = np.clip((theta[mask] - theta_start) / duration, 0.0, 1.0)
    x_b[mask] = 1.0 - np.exp(-float(a) * xi ** (float(m) + 1.0))
    return np.clip(x_b, 0.0, 1.0)


def double_wiebe_burn_fraction(
    theta: np.ndarray,
    *,
    theta_start: float,
    duration: float,
    split: float,
    a1: float,
    m1: float,
    a2: float,
    m2: float,
    second_stage_shift_frac: float = 0.35,
) -> np.ndarray:
    """Two-stage Wiebe burn law for premixed + diffusion-like tail."""
    if duration <= 1e-9:
        return np.zeros_like(theta, dtype=np.float64)

    s = float(np.clip(split, 0.0, 1.0))
    shift = float(second_stage_shift_frac) * float(duration)

    x1 = wiebe_burn_fraction(theta, theta_start, duration, a=float(a1), m=float(m1))
    x2 = wiebe_burn_fraction(
        theta,
        theta_start + shift,
        max(1e-9, duration - shift),
        a=float(a2),
        m=float(m2),
    )
    x = s * x1 + (1.0 - s) * x2
    # Enforce monotonicity and bounds to protect solver behavior.
    x = np.maximum.accumulate(np.clip(x, 0.0, 1.0))
    return x


def wrapped_double_wiebe_burn_fraction(
    theta: np.ndarray,
    *,
    theta_start: float,
    duration: float,
    split: float,
    a1: float,
    m1: float,
    a2: float,
    m2: float,
    second_stage_shift_frac: float = 0.35,
) -> np.ndarray:
    """Cycle-periodic double Wiebe profile that supports combustion crossing 360 deg."""
    theta_arr = np.asarray(theta, dtype=np.float64)
    start = float(np.mod(theta_start, 360.0))
    dur = max(float(duration), 1e-9)
    if start + dur <= 360.0:
        return double_wiebe_burn_fraction(
            theta_arr,
            theta_start=start,
            duration=dur,
            split=split,
            a1=a1,
            m1=m1,
            a2=a2,
            m2=m2,
            second_stage_shift_frac=second_stage_shift_frac,
        )
    theta_unwrapped = np.asarray(theta_arr, dtype=np.float64).copy()
    theta_unwrapped[theta_unwrapped < start] += 360.0
    return double_wiebe_burn_fraction(
        theta_unwrapped,
        theta_start=start,
        duration=dur,
        split=split,
        a1=a1,
        m1=m1,
        a2=a2,
        m2=m2,
        second_stage_shift_frac=second_stage_shift_frac,
    )


def burn_increment(
    burn_profile: np.ndarray,
    *,
    oxygen_completion_cap: float,
    vapor_fraction: np.ndarray | float,
) -> np.ndarray:
    """Per-step combustible fraction increment after oxygen/vapor limits."""
    x = np.asarray(burn_profile, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    dx = np.diff(np.concatenate([[0.0], np.maximum.accumulate(np.clip(x, 0.0, 1.0))]))

    cap = float(np.clip(oxygen_completion_cap, 0.0, 1.0))
    if np.isscalar(vapor_fraction):
        vap = np.full_like(dx, float(np.clip(vapor_fraction, 0.0, 1.0)))
    else:
        vap = np.asarray(vapor_fraction, dtype=np.float64).reshape(-1)
        if vap.size != dx.size:
            raise ValueError("vapor_fraction length mismatch")
        vap = np.clip(vap, 0.0, 1.0)

    dx_eff = dx * cap * vap
    # Do not let cumulative burn exceed cap.
    x_eff = np.minimum(np.cumsum(dx_eff), cap)
    out = np.diff(np.concatenate([[0.0], x_eff]))
    return np.clip(out, 0.0, 1.0)
