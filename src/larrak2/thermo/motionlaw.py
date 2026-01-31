"""Motion law and thermodynamic cycle evaluation.

This module implements toy ideal-gas cycle physics for initial testing.
The API is stable; only the internals will get more realistic later.

Outputs:
    - efficiency: thermal efficiency W/Q_in
    - i_req_profile: required ratio profile over theta (N_POINTS values)
    - G_thermo: constraint array (all G <= 0 feasible)
    - diag: diagnostics dict with p, V, T, Qdot arrays
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.encoding import ThermoParams
from ..core.types import EvalContext
from ..core.constants import RATIO_SLOPE_LIMIT_FID0, RATIO_SLOPE_LIMIT_FID1

# Number of discretization points
N_POINTS = 360

# Physical constants for toy model
GAMMA = 1.4  # Ratio of specific heats (ideal diatomic gas)
R_GAS = 287.0  # J/(kg·K) - specific gas constant for air
P_ATM = 101325.0  # Pa - atmospheric pressure
T_ATM = 300.0  # K - ambient temperature

# Constraint limits
MIN_COMPRESSION_DURATION = 30.0  # degrees
MAX_JERK = 1e6  # placeholder jerk limit


@dataclass
class ThermoResult:
    """Result from thermodynamic evaluation."""

    efficiency: float
    requested_ratio_profile: np.ndarray
    G: np.ndarray
    diag: dict[str, Any] = field(default_factory=dict)


def _sinusoidal_volume(theta: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
    """Generate sinusoidal piston volume profile.

    Args:
        theta: Angle array in degrees [0, 360).
        v_min: Clearance volume (m³).
        v_max: Maximum volume (m³).

    Returns:
        Volume array (m³).
    """
    # Simple sinusoidal motion: V = V_min + (V_max - V_min) * (1 - cos(theta))/2
    theta_rad = np.radians(theta)
    return v_min + (v_max - v_min) * (1 - np.cos(theta_rad)) / 2


def _wiebe_heat_release(
    theta: np.ndarray,
    theta_start: float,
    duration: float,
    q_total: float,
    a: float = 5.0,
    m: float = 2.0,
) -> np.ndarray:
    """Toy Wiebe-like heat release profile.

    Args:
        theta: Angle array in degrees.
        theta_start: Start of combustion (degrees).
        duration: Burn duration (degrees).
        q_total: Total heat release (J).
        a: Wiebe parameter (typically 5).
        m: Wiebe shape parameter.

    Returns:
        Heat release rate dQ/dtheta (J/deg).
    """
    dqdtheta = np.zeros_like(theta, dtype=np.float64)

    mask = (theta >= theta_start) & (theta < theta_start + duration)
    if not np.any(mask):
        return dqdtheta

    # Normalized progress
    xi = (theta[mask] - theta_start) / duration

    # Wiebe function: x_b = 1 - exp(-a * xi^(m+1))
    # dQ/dtheta = Q_total * a * (m+1) / duration * xi^m * exp(-a * xi^(m+1))
    xi_safe = np.clip(xi, 1e-10, 1.0)
    exp_term = np.exp(-a * xi_safe ** (m + 1))
    dqdtheta[mask] = q_total * a * (m + 1) / duration * xi_safe**m * exp_term

    return dqdtheta


def _polytropic_pressure(
    v: np.ndarray,
    v_ref: float,
    p_ref: float,
    n: float,
) -> np.ndarray:
    """Compute pressure from polytropic relation p*V^n = const.

    Args:
        v: Volume array.
        v_ref: Reference volume.
        p_ref: Reference pressure.
        n: Polytropic exponent.

    Returns:
        Pressure array.
    """
    return p_ref * (v_ref / v) ** n


def _compute_work(p: np.ndarray, v: np.ndarray) -> float:
    """Compute indicated work W = ∮ p dV using trapezoidal integration.

    Args:
        p: Pressure array (Pa).
        v: Volume array (m³).

    Returns:
        Work (J).
    """
    # dV = diff(V), use trapezoidal rule
    dv = np.diff(v)
    p_avg = (p[:-1] + p[1:]) / 2
    return float(np.sum(p_avg * dv))


def _generate_ratio_profile(
    params: ThermoParams,
    theta: np.ndarray,
) -> np.ndarray:
    """Generate required gear ratio profile from thermo params.

    The ratio profile represents the desired instantaneous gear ratio
    i(theta) that the gear system must track.

    Args:
        params: Thermodynamic parameters.
        theta: Angle array in degrees.

    Returns:
        Ratio profile array (dimensionless).
    """
    # Toy model: ratio varies sinusoidally with thermo params affecting amplitude/phase
    theta_rad = np.radians(theta)

    # Base ratio = 1.0, with modulation from thermo params
    compression_effect = 0.2 * np.sin(theta_rad - np.radians(params.compression_duration))
    expansion_effect = 0.15 * np.sin(2 * theta_rad - np.radians(params.expansion_duration))

    ratio = 1.0 + compression_effect + expansion_effect

    # Ensure positive ratios
    return np.maximum(ratio, 0.1)


def _ratio_profile_stats(profile: np.ndarray) -> dict[str, float | bool]:
    """Compute basic continuity/quality metrics for ratio profile."""
    profile = np.asarray(profile)
    finite = np.all(np.isfinite(profile))
    min_val = float(np.min(profile)) if profile.size else 0.0
    max_val = float(np.max(profile)) if profile.size else 0.0
    slope = np.gradient(profile) if profile.size > 1 else np.array([0.0])
    max_slope = float(np.max(np.abs(slope)))
    return {
        "finite": bool(finite),
        "min": min_val,
        "max": max_val,
        "max_slope": max_slope,
    }


def eval_thermo(params: ThermoParams, ctx: EvalContext) -> ThermoResult:
    """Evaluate thermodynamic cycle.

    Fidelity routing:
    - fidelity=0: toy ideal-gas physics (current)
    - fidelity>=1: v1 Wiebe-based model

    Args:
        params: Thermodynamic control parameters.
        ctx: Evaluation context (rpm, torque, fidelity).

    Returns:
        ThermoResult with efficiency, ratio profile, constraints, and diagnostics.
    """
    # Fidelity routing
    if ctx.fidelity >= 1:
        from ..ports.larrak_v1.thermo_forward import v1_eval_thermo_forward

        v1_result = v1_eval_thermo_forward(params, ctx)
        stats = _ratio_profile_stats(v1_result.requested_ratio_profile)
        return ThermoResult(
            efficiency=v1_result.efficiency,
            requested_ratio_profile=v1_result.requested_ratio_profile,
            G=v1_result.G,
            diag={**v1_result.diag, "ratio_profile_stats": stats},
        )

    # Discretization
    theta = np.linspace(0, 360, N_POINTS, endpoint=False)
    d_theta = theta[1] - theta[0]

    # Geometry (toy fixed values)
    bore = 0.08  # m
    stroke = 0.09  # m
    v_displacement = np.pi / 4 * bore**2 * stroke  # m³
    compression_ratio = 10.0
    v_clearance = v_displacement / (compression_ratio - 1)
    v_max = v_clearance + v_displacement

    # Volume profile
    v = _sinusoidal_volume(theta, v_clearance, v_max)

    # Heat release (toy: centered at TDC + heat_release_center)
    theta_ignition = 180.0 + params.heat_release_center  # TDC at 180°
    q_total = 2000.0  # J (toy value)
    dqdtheta = _wiebe_heat_release(
        theta,
        theta_ignition,
        params.heat_release_width,
        q_total,
    )

    # Pressure trace (simplified: polytropic compression/expansion + heat addition)
    p = np.zeros_like(theta, dtype=np.float64)
    t = np.zeros_like(theta, dtype=np.float64)

    # Initial conditions at BDC (theta=0)
    p[0] = P_ATM
    t[0] = T_ATM
    m_gas = p[0] * v[0] / (R_GAS * t[0])  # kg

    # Step through cycle
    for i in range(1, N_POINTS):
        # Polytropic change
        p_poly = p[i - 1] * (v[i - 1] / v[i]) ** GAMMA

        # Add heat release (constant volume approximation for simplicity)
        dq = dqdtheta[i] * d_theta
        cv = R_GAS / (GAMMA - 1)
        dt_heat = dq / (m_gas * cv)

        # Update state
        t[i] = t[i - 1] * (v[i - 1] / v[i]) ** (GAMMA - 1) + dt_heat
        p[i] = m_gas * R_GAS * t[i] / v[i]

        # Blend: mostly polytropic, with heat release perturbation
        p[i] = 0.7 * p_poly + 0.3 * p[i]

    # Work and efficiency
    w_indicated = _compute_work(p, v)
    q_in = np.sum(dqdtheta) * d_theta
    efficiency = w_indicated / q_in if q_in > 0 else 0.0

    # Clamp efficiency to physical bounds for toy model
    efficiency = float(np.clip(efficiency, 0.0, 0.6))

    # Generate ratio profile
    requested_ratio_profile = _generate_ratio_profile(params, theta)
    ratio_stats = _ratio_profile_stats(requested_ratio_profile)

    # Constraints (G <= 0 feasible)
    constraints = []

    # C1: Compression duration >= MIN_COMPRESSION_DURATION
    g_compression = MIN_COMPRESSION_DURATION - params.compression_duration
    constraints.append(g_compression)

    # C2: Heat release width > 0 (already bounded, but add constraint)
    g_hr_width = 5.0 - params.heat_release_width  # width >= 5
    constraints.append(g_hr_width)

    # C3: Smoothness constraint (toy: max jerk from ratio profile)
    d_ratio = np.gradient(requested_ratio_profile)
    dd_ratio = np.gradient(d_ratio)
    max_jerk_actual = np.max(np.abs(dd_ratio))
    g_jerk = max_jerk_actual - MAX_JERK
    constraints.append(g_jerk)

    # C4: Max slope of ratio profile (gear capability proxy)
    slope_limit = RATIO_SLOPE_LIMIT_FID1 if ctx.fidelity >= 1 else RATIO_SLOPE_LIMIT_FID0
    g_slope = ratio_stats["max_slope"] - slope_limit
    constraints.append(g_slope)

    G = np.array(constraints, dtype=np.float64)

    # Diagnostics
    ratio_stats = _ratio_profile_stats(requested_ratio_profile)
    diag = {
        "theta": theta,
        "p": p,
        "v": v,
        "t": t,
        "dqdtheta": dqdtheta,
        "w_indicated": w_indicated,
        "q_in": q_in,
        "requested_ratio_profile": requested_ratio_profile,
        "ratio_profile_stats": ratio_stats,
    }

    return ThermoResult(
        efficiency=efficiency,
        requested_ratio_profile=requested_ratio_profile,
        G=G,
        diag=diag,
    )
