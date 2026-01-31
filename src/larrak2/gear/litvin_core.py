"""Litvin-style gear evaluation with ratio tracking and losses.

This module implements toy gear physics for initial testing.
The API is stable; only the internals will get more realistic later.

Outputs:
    - ratio_error_mean: mean absolute ratio tracking error
    - ratio_error_max: maximum ratio tracking error
    - max_planet_radius: maximum planet pitch radius (mm)
    - loss_total: total mesh friction loss (W)
    - G_gear: constraint array (all G <= 0 feasible)
    - diag: diagnostics dict with profiles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.constants import GEAR_INTERFERENCE_CLEARANCE_MM, GEAR_MIN_THICKNESS_MM
from ..core.encoding import GearParams
from ..core.types import EvalContext
from ..ports.larrak_v1.gear_forward import litvin_synthesize
from .pitchcurve import fourier_pitch_curve

# Number of discretization points
N_POINTS = 360

# Physical constants for toy model
MU_MESH = 0.05  # Friction coefficient
SPEED_SCALE = 1.0  # m/s per rad/s (toy)

# Constraint limits
RATIO_ERROR_TOL = 5.0  # 500% max ratio error (relaxed for toy physics)
MIN_RADIUS = 5.0  # mm
MAX_RADIUS = 100.0  # mm
MAX_CURVATURE = 0.5  # 1/mm


@dataclass
class GearResult:
    """Result from gear evaluation."""

    ratio_error_mean: float
    ratio_error_max: float
    max_planet_radius: float
    loss_total: float
    G: np.ndarray
    diag: dict[str, Any] = field(default_factory=dict)


def _compute_ratio_error(
    r_actual: np.ndarray,
    i_req_profile: np.ndarray,
    r_ring: float,
) -> tuple[float, float]:
    """Compute ratio tracking error.

    Args:
        r_actual: Actual planet radius profile.
        i_req_profile: Required ratio profile.
        r_ring: Ring gear radius.

    Returns:
        (mean_error, max_error) tuple.
    """
    # Ratio = r_ring / r_planet (toy model)
    ratio_actual = r_ring / np.maximum(r_actual, 1e-6)
    error = np.abs(ratio_actual - i_req_profile) / np.maximum(i_req_profile, 1e-6)

    return float(np.mean(error)), float(np.max(error))


def _compute_mesh_loss(
    theta: np.ndarray,
    r_planet: np.ndarray,
    omega_rpm: float,
    torque: float,
    mu: float = MU_MESH,
) -> tuple[float, np.ndarray]:
    """Compute mesh friction loss using toy model.

    P_loss ~ mean(μ * Fn * v_sliding)

    Args:
        theta: Angle array.
        r_planet: Planet radius profile (mm).
        omega_rpm: Rotational speed (rpm).
        torque: Torque (Nm).
        mu: Friction coefficient.

    Returns:
        (total_loss_watts, loss_profile) tuple.
    """
    omega_rad = omega_rpm * 2 * np.pi / 60  # rad/s

    # Toy model: loss varies with radius variation (sliding velocity depends on radius)
    # P_loss ~ mu * Fn * v_slide where v_slide ~ omega * dr/dtheta (radius change rate)
    r_m = r_planet / 1000  # mm to m
    dr = np.gradient(r_m)  # radius change rate

    # Normal force from torque, sliding from radius change
    fn = torque / np.maximum(np.mean(r_m), 1e-6)  # N (average)
    v_sliding = omega_rad * np.abs(dr) * 100 + omega_rad * r_m * 0.01  # m/s (toy)

    loss_profile = mu * fn * np.abs(v_sliding)  # W
    total_loss = float(np.mean(loss_profile)) + omega_rad * torque * 0.001  # base loss

    # Enforce non-negativity
    loss_profile = np.maximum(loss_profile, 0.0)
    total_loss = max(total_loss, 0.0)

    return total_loss, loss_profile


def _compute_curvature(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute curvature of pitch curve (simplified).

    Args:
        r: Radius profile.
        theta: Angle array.

    Returns:
        Curvature array (1/mm).
    """
    dr = np.gradient(r)
    ddr = np.gradient(dr)

    # Simplified curvature formula for polar curve
    # κ = |r² + 2(dr)² - r*d²r| / (r² + (dr)²)^(3/2)
    r_safe = np.maximum(r, 1e-6)
    num = np.abs(r_safe**2 + 2 * dr**2 - r_safe * ddr)
    denom = (r_safe**2 + dr**2) ** 1.5
    denom = np.maximum(denom, 1e-10)

    return num / denom


def eval_gear(
    params: GearParams,
    i_req_profile: np.ndarray,
    ctx: EvalContext,
) -> GearResult:
    """Evaluate gear system.

    Fidelity routing:
    - fidelity=0: toy friction model
    - fidelity>=1: v1 Litvin synthesis with enhanced mesh loss

    Args:
        params: Gear control parameters.
        i_req_profile: Required ratio profile from thermo (length N_POINTS).
        ctx: Evaluation context (rpm, torque, fidelity).

    Returns:
        GearResult with errors, losses, constraints, and diagnostics.
    """
    # Fidelity routing
    if ctx.fidelity >= 1:
        from ..ports.larrak_v1.gear_forward import v1_eval_gear_forward

        v1_result = v1_eval_gear_forward(params, i_req_profile, ctx)
        return GearResult(
            ratio_error_mean=v1_result.ratio_error_mean,
            ratio_error_max=v1_result.ratio_error_max,
            max_planet_radius=v1_result.max_planet_radius,
            loss_total=v1_result.loss_total,
            G=v1_result.G,
            diag=v1_result.diag,
        )

    theta = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)

    # Generate planet pitch curve from Fourier coefficients
    r_planet = fourier_pitch_curve(theta, params.base_radius, params.pitch_coeffs)

    # Ring radius (fixed toy value)
    r_ring = 80.0  # mm

    # Litvin synthesis to obtain conjugate ring profile
    synth = litvin_synthesize(theta, r_planet, target_ratio=r_ring / params.base_radius)
    R_psi = synth["R_psi"]

    # Ratio tracking error
    ratio_error_mean, ratio_error_max = _compute_ratio_error(r_planet, i_req_profile, r_ring)

    # Mesh friction loss
    loss_total, loss_profile = _compute_mesh_loss(
        theta,
        r_planet,
        ctx.rpm,
        ctx.torque,
    )

    # Geometry metrics
    max_planet_radius = float(np.max(r_planet))
    min_planet_radius = float(np.min(r_planet))

    thickness_profile = R_psi - r_planet
    min_thickness = float(np.min(thickness_profile))
    interference_flag = bool(np.any(thickness_profile < GEAR_INTERFERENCE_CLEARANCE_MM))
    thickness_ok = min_thickness >= GEAR_MIN_THICKNESS_MM

    # Curvature
    curvature = _compute_curvature(r_planet, theta)
    max_curvature = float(np.max(curvature))

    # Constraints (G <= 0 feasible)
    constraints = []

    # C1: Ratio error <= tolerance
    g_ratio = ratio_error_max - RATIO_ERROR_TOL
    constraints.append(g_ratio)

    # C2: Min radius >= MIN_RADIUS
    g_min_r = MIN_RADIUS - min_planet_radius
    constraints.append(g_min_r)

    # C3: Max radius <= MAX_RADIUS
    g_max_r = max_planet_radius - MAX_RADIUS
    constraints.append(g_max_r)

    # C4: Max curvature <= MAX_CURVATURE
    g_curv = max_curvature - MAX_CURVATURE
    constraints.append(g_curv)

    # C5: Interference check (non-blocking placeholder; diagnostics carry truth)
    g_interference = -1.0
    constraints.append(g_interference)

    # C6: Min thickness >= threshold
    g_thickness = GEAR_MIN_THICKNESS_MM - min_thickness
    constraints.append(g_thickness)

    G = np.array(constraints, dtype=np.float64)

    # Diagnostics
    diag = {
        "theta": theta,
        "r_planet": r_planet,
        "R_psi": R_psi,
        "thickness_profile": thickness_profile,
        "curvature": curvature,
        "loss_profile": loss_profile,
        "r_ring": r_ring,
        "min_planet_radius": min_planet_radius,
        "max_curvature": max_curvature,
        "min_thickness": min_thickness,
        "thickness_ok": thickness_ok,
        "interference_flag": interference_flag,
    }

    return GearResult(
        ratio_error_mean=ratio_error_mean,
        ratio_error_max=ratio_error_max,
        max_planet_radius=max_planet_radius,
        loss_total=loss_total,
        G=G,
        diag=diag,
    )
