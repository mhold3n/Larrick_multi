"""Fuel vaporization dynamics for combustion availability."""

from __future__ import annotations

import numpy as np


def vaporization_time_constant(
    temp_k: float,
    pressure_pa: float,
    *,
    tau_ref_s: float,
    temp_ref_k: float = 500.0,
    pressure_ref_pa: float = 1.0e5,
    temp_exponent: float = -1.2,
    pressure_exponent: float = 0.2,
) -> float:
    """First-order vaporization time-scale model."""
    t = max(float(temp_k), 120.0)
    p = max(float(pressure_pa), 1.0)
    tau = float(tau_ref_s)
    tau *= (t / float(temp_ref_k)) ** float(temp_exponent)
    tau *= (p / float(pressure_ref_pa)) ** float(pressure_exponent)
    return float(np.clip(tau, 1e-5, 0.5))


def step_vapor_fraction(
    phi_prev: float,
    *,
    dt_s: float,
    temp_k: float,
    pressure_pa: float,
    tau_ref_s: float,
    temp_exponent: float,
    pressure_exponent: float,
) -> float:
    """Advance vapor fraction with first-order lag to equilibrium=1."""
    dt = max(float(dt_s), 0.0)
    phi0 = float(np.clip(phi_prev, 0.0, 1.0))
    tau = vaporization_time_constant(
        temp_k,
        pressure_pa,
        tau_ref_s=float(tau_ref_s),
        temp_exponent=float(temp_exponent),
        pressure_exponent=float(pressure_exponent),
    )
    if dt <= 0.0:
        return phi0
    k = np.exp(-dt / max(tau, 1e-9))
    phi = 1.0 - (1.0 - phi0) * k
    return float(np.clip(phi, 0.0, 1.0))


def integrate_vapor_fraction(
    theta_deg: np.ndarray,
    temp_k: np.ndarray,
    pressure_pa: np.ndarray,
    *,
    rpm: float,
    initial_vapor_fraction: float,
    tau_ref_s: float,
    temp_exponent: float,
    pressure_exponent: float,
) -> np.ndarray:
    """Compute vapor fraction profile across crank-angle trace."""
    theta = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    t = np.asarray(temp_k, dtype=np.float64).reshape(-1)
    p = np.asarray(pressure_pa, dtype=np.float64).reshape(-1)
    if theta.size == 0:
        return np.zeros(0, dtype=np.float64)
    if t.size != theta.size or p.size != theta.size:
        raise ValueError("theta/temp/pressure length mismatch")

    phi = np.zeros(theta.size, dtype=np.float64)
    phi[0] = float(np.clip(initial_vapor_fraction, 0.0, 1.0))
    dtheta = np.diff(theta)
    dtheta = np.concatenate([dtheta, [max(1e-6, 360.0 - (theta[-1] - theta[0]))]])
    dt = dtheta / max(6.0 * float(rpm), 1e-9)

    for i in range(1, theta.size):
        phi[i] = step_vapor_fraction(
            float(phi[i - 1]),
            dt_s=float(dt[i - 1]),
            temp_k=float(t[i - 1]),
            pressure_pa=float(p[i - 1]),
            tau_ref_s=float(tau_ref_s),
            temp_exponent=float(temp_exponent),
            pressure_exponent=float(pressure_exponent),
        )
    return phi
