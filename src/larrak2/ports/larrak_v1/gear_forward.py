"""Gear forward-evaluation ported from Larrak v1.

This module contains pure forward-evaluation logic copied from:
- campro/physics/geometry/curvature.py
- campro/physics/geometry/litvin.py

NO optimizer code, NO CasADi, NO v1 imports.

Attribution: Original code from Larrak v1 by Max Holden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from ...core.encoding import GearParams
from ...core.types import EvalContext
from ...gear.pitchcurve import fourier_pitch_curve

# Constants from v1 (hardcoded to avoid imports)
MIN_RADIUS_MM = 5.0
MAX_RADIUS_MM = 100.0
MAX_CURVATURE = 0.5  # 1/mm
MU_MESH = 0.05  # friction coefficient
N_POINTS = 360


@dataclass
class V1GearResult:
    """Result from v1 gear forward evaluation."""

    ratio_error_mean: float
    ratio_error_max: float
    max_planet_radius: float
    loss_total: float
    G: np.ndarray  # G <= 0 feasible
    diag: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Curvature computation (copied from campro/physics/geometry/curvature.py)
# =============================================================================


def _compute_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute derivative using finite differences (gradient)."""
    if len(x) < 2:
        return np.zeros_like(y)
    return np.gradient(y, x)


def compute_curvature(theta: np.ndarray, r_theta: np.ndarray) -> dict[str, np.ndarray]:
    """Compute curvature and osculating radius for polar curve.

    Ported from campro/physics/geometry/curvature.py CurvatureComponent.compute()

    Args:
        theta: Cam angles (radians), length N.
        r_theta: Radius values r(θ), length N.

    Returns:
        Dict with keys: kappa (curvature), rho (osculating radius),
                       r_prime, r_double_prime
    """
    r_prime = _compute_derivative(theta, r_theta)
    r_double_prime = _compute_derivative(theta, r_prime)

    # Curvature for polar curve: κ = (r² + 2r'² - rr'') / (r² + r'²)^(3/2)
    r_squared = r_theta**2
    r_prime_squared = r_prime**2

    numerator = r_squared + 2 * r_prime_squared - r_theta * r_double_prime
    denominator = (r_squared + r_prime_squared) ** 1.5

    # Avoid division by zero
    kappa = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )

    # Osculating radius: ρ = 1/κ
    rho = np.divide(
        1.0, kappa, out=np.full_like(kappa, np.inf), where=kappa != 0
    )

    return {
        "kappa": kappa,
        "rho": rho,
        "r_prime": r_prime,
        "r_double_prime": r_double_prime,
    }


# =============================================================================
# Litvin synthesis (adapted from campro/physics/geometry/litvin.py)
# =============================================================================


def _periodic_gradient(values: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Central difference gradient with periodic wrap-around."""
    n = values.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)

    period = 2.0 * np.pi
    theta_ext = np.concatenate([angles, [angles[0] + period]])
    values_ext = np.concatenate([values, [values[0]]])
    slopes = np.diff(values_ext) / np.diff(theta_ext)

    grad = np.empty(n, dtype=float)
    grad[0] = 0.5 * (slopes[0] + slopes[-1])
    grad[1:] = 0.5 * (slopes[1:] + slopes[:-1])
    return grad


def _periodic_pchip(psi: np.ndarray, values: np.ndarray) -> PchipInterpolator:
    """Construct a periodic PCHIP interpolator to avoid wrap-around spikes."""
    if psi.size == 0:
        raise ValueError("psi must contain at least one sample")

    period = (psi[-1] - psi[0]) + (psi[1] - psi[0]) if psi.size > 1 else 2.0 * np.pi
    psi_ext = np.concatenate([[psi[0] - period], psi, [psi[-1] + period]])
    values_ext = np.concatenate([[values[-1]], values, [values[0]]])

    return PchipInterpolator(psi_ext, values_ext, extrapolate=True)


def litvin_synthesize(
    theta: np.ndarray,
    r_profile: np.ndarray,
    target_ratio: float = 1.0,
    min_radius: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Build conjugate ring profile R(ψ) for given cam profile r_c(θ).

    Ported from campro/physics/geometry/litvin.py LitvinSynthesis.synthesize_from_cam_profile()

    Args:
        theta: Cam angle grid (radians), length N, [0, 2π).
        r_profile: Cam polar radius r_c(θ), length N.
        target_ratio: Desired average transmission ratio.
        min_radius: Minimum allowed R(ψ).

    Returns:
        Dict with psi, R_psi, rho_c arrays.
    """
    theta = np.asarray(theta)
    r_profile = np.asarray(r_profile)

    # Compute curvature and osculating radius ρ_c(θ)
    curv = compute_curvature(theta, r_profile)
    rho_c = curv["rho"]

    # Use supplied polar profile as the bounded ring radius
    R_theta = np.maximum(r_profile, min_radius)
    dpsi_dtheta = np.divide(rho_c, np.maximum(R_theta, min_radius))

    # Integrate to obtain ψ(θ)
    if len(theta) > 1:
        increments = 0.5 * (dpsi_dtheta[1:] + dpsi_dtheta[:-1]) * np.diff(theta)
        psi = np.concatenate([[0.0], np.cumsum(increments)])
    else:
        psi = np.array([0.0], dtype=float)

    # Enforce periodicity and ratio scaling
    span = psi[-1] - psi[0]
    if span <= 0.0:
        span = 1.0
    scale = (2.0 * np.pi) / span
    psi = (psi - psi[0]) * scale

    # Apply ratio offset
    psi_offset = float(target_ratio) * 1e-3
    psi = psi + psi_offset

    # Interpolate R(ψ) from bounded reference profile
    psi_grid = psi.copy()
    eps = 1e-9
    for i in range(1, len(psi_grid)):
        if psi_grid[i] <= psi_grid[i - 1]:
            psi_grid[i] = psi_grid[i - 1] + eps

    R_interp = _periodic_pchip(psi_grid, R_theta)
    R_psi = np.maximum(R_interp(psi), min_radius)

    return {"psi": psi, "R_psi": R_psi, "rho_c": rho_c}


# =============================================================================
# Gear geometry metrics (adapted from campro/physics/geometry/litvin.py)
# =============================================================================


def compute_gear_geometry(
    theta: np.ndarray,
    r_profile: np.ndarray,
    psi: np.ndarray,
    R_psi: np.ndarray,
    target_average_radius: float,
    module: float = 1.0,
    max_pressure_angle_deg: float = 35.0,
) -> dict[str, Any]:
    """Compute gear geometry metrics from synthesized profiles.

    Ported from campro/physics/geometry/litvin.py LitvinGearGeometry.from_synthesis()

    Returns:
        Dict with base circles, pressure angle, contact ratio, interference flag, etc.
    """
    r_cam_avg = float(np.mean(r_profile))
    r_ring_avg = float(target_average_radius)

    # Base circles (20° nominal pressure angle)
    phi_nom = np.radians(20.0)
    base_cam = max(1e-3, r_cam_avg * np.cos(phi_nom))
    base_ring = max(1e-3, r_ring_avg * np.cos(phi_nom))

    # Teeth counts
    min_teeth = 8
    z_cam = max(min_teeth, int(np.round((2.0 * np.pi * r_cam_avg) / module)))
    z_ring = max(min_teeth, int(np.round((2.0 * np.pi * r_ring_avg) / module)))

    # Pressure angle estimation
    ring_radii = np.full_like(R_psi, r_ring_avg)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_phi_local = np.clip(base_ring / np.maximum(ring_radii, 1e-9), 0.0, 1.0)
        phi_local = np.arccos(cos_phi_local)

    # Path of contact
    phi_bound = np.radians(max_pressure_angle_deg)
    valid = np.isfinite(phi_local) & (phi_local <= phi_bound)
    s_path = float(np.trapz(ring_radii[valid], psi[valid])) if np.any(valid) else 0.0

    # Contact ratio
    circular_pitch = (2.0 * np.pi * r_ring_avg) / max(z_ring, 1)
    contact_ratio = s_path / max(circular_pitch, 1e-9)

    # Interference check
    interference = bool(np.any(~valid) or (base_ring > np.min(ring_radii)))

    return {
        "base_circle_cam": base_cam,
        "base_circle_ring": base_ring,
        "pressure_angle_rad": phi_local,
        "contact_ratio": contact_ratio,
        "path_of_contact": s_path,
        "z_cam": z_cam,
        "z_ring": z_ring,
        "interference_flag": interference,
    }


# =============================================================================
# Mesh friction loss (enhanced from toy model)
# =============================================================================


def compute_mesh_loss(
    theta: np.ndarray,
    r_planet: np.ndarray,
    rho_c: np.ndarray,
    omega_rpm: float,
    torque: float,
    mu: float = MU_MESH,
) -> tuple[float, np.ndarray]:
    """Compute mesh friction loss using Litvin-based sliding velocity.

    Enhanced from toy model with proper sliding velocity from curvature.

    Args:
        theta: Angle array (radians).
        r_planet: Planet radius profile (mm).
        rho_c: Osculating radius (mm).
        omega_rpm: Rotational speed (rpm).
        torque: Torque (Nm).
        mu: Friction coefficient.

    Returns:
        (total_loss_watts, loss_profile)
    """
    omega_rad = omega_rpm * 2 * np.pi / 60  # rad/s

    # Convert to meters
    r_m = r_planet / 1000
    rho_m = np.abs(rho_c) / 1000

    # Sliding velocity: v_slide ~ omega * |r - rho_c| (simplified Litvin)
    v_sliding = omega_rad * np.abs(r_m - rho_m)

    # Normal force from torque
    fn = torque / np.maximum(np.mean(r_m), 1e-6)

    # Power loss
    loss_profile = mu * fn * v_sliding
    total_loss = float(np.mean(loss_profile))

    return total_loss, loss_profile


# =============================================================================
# Main forward-evaluation function
# =============================================================================


def v1_eval_gear_forward(
    params: GearParams,
    i_req_profile: np.ndarray,
    ctx: EvalContext,
) -> V1GearResult:
    """Evaluate gear using v1 Litvin synthesis.

    This is the main entry point for fidelity=1 gear evaluation.

    Args:
        params: Gear control parameters (base_radius, pitch_coeffs).
        i_req_profile: Required ratio profile from thermo (length N_POINTS).
        ctx: Evaluation context (rpm, torque, fidelity).

    Returns:
        V1GearResult with errors, losses, constraints (G<=0 feasible), and diagnostics.
    """
    theta = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)

    # Generate planet pitch curve from Fourier coefficients
    r_planet = fourier_pitch_curve(theta, params.base_radius, params.pitch_coeffs)

    # Ring radius (fixed reference)
    r_ring = 80.0  # mm

    # Litvin synthesis
    synth = litvin_synthesize(
        theta, r_planet, target_ratio=r_ring / params.base_radius
    )
    psi = synth["psi"]
    R_psi = synth["R_psi"]
    rho_c = synth["rho_c"]

    # Gear geometry
    geom = compute_gear_geometry(
        theta, r_planet, psi, R_psi, target_average_radius=r_ring
    )

    # Ratio tracking error
    ratio_actual = r_ring / np.maximum(r_planet, 1e-6)
    ratio_error = np.abs(ratio_actual - i_req_profile) / np.maximum(i_req_profile, 1e-6)
    ratio_error_mean = float(np.mean(ratio_error))
    ratio_error_max = float(np.max(ratio_error))

    # Mesh friction loss (enhanced)
    loss_total, loss_profile = compute_mesh_loss(
        theta, r_planet, rho_c, ctx.rpm, ctx.torque
    )

    # Geometry metrics
    max_planet_radius = float(np.max(r_planet))
    min_planet_radius = float(np.min(r_planet))

    # Curvature
    curv = compute_curvature(theta, r_planet)
    max_curvature = float(np.max(np.abs(curv["kappa"])))

    # Constraints (G <= 0 feasible)
    # More realistic tolerances than toy model
    RATIO_ERROR_TOL = 0.5  # 50% max (still relaxed for testing)
    constraints = []

    # C1: Ratio error <= tolerance
    g_ratio = ratio_error_max - RATIO_ERROR_TOL
    constraints.append(g_ratio)

    # C2: Min radius >= MIN_RADIUS
    g_min_r = MIN_RADIUS_MM - min_planet_radius
    constraints.append(g_min_r)

    # C3: Max radius <= MAX_RADIUS
    g_max_r = max_planet_radius - MAX_RADIUS_MM
    constraints.append(g_max_r)

    # C4: Max curvature <= MAX_CURVATURE
    g_curv = max_curvature - MAX_CURVATURE
    constraints.append(g_curv)

    G = np.array(constraints, dtype=np.float64)

    # Diagnostics
    diag = {
        "theta": theta,
        "r_planet": r_planet,
        "psi": psi,
        "R_psi": R_psi,
        "rho_c": rho_c,
        "curvature": curv["kappa"],
        "loss_profile": loss_profile,
        "r_ring": r_ring,
        "min_planet_radius": min_planet_radius,
        "max_curvature": max_curvature,
        "contact_ratio": geom["contact_ratio"],
        "interference_flag": geom["interference_flag"],
        "v1_port": True,
    }

    return V1GearResult(
        ratio_error_mean=ratio_error_mean,
        ratio_error_max=ratio_error_max,
        max_planet_radius=max_planet_radius,
        loss_total=loss_total,
        G=G,
        diag=diag,
    )
