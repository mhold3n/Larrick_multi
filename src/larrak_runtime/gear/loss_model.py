"""Modular loss components for gear system."""

from __future__ import annotations

from typing import Any

import numpy as np


def mesh_loss(theta, r_planet, rho_c, omega_rpm, torque, mu=0.05):
    """Mesh sliding loss (Litvin-based)."""
    omega_rad = omega_rpm * 2 * np.pi / 60
    r_m = r_planet / 1000
    rho_m = np.abs(rho_c) / 1000
    v_sliding = omega_rad * np.abs(r_m - rho_m)
    fn = torque / np.maximum(np.mean(r_m), 1e-6)
    loss_profile = np.maximum(mu * fn * v_sliding, 0.0)
    return float(np.mean(loss_profile)), loss_profile


def windage_loss(theta, r_planet, omega_rpm, rho_air=1.2, drag_coeff=0.5):
    """Simple windage/churning loss model."""
    omega_rad = omega_rpm * 2 * np.pi / 60
    r_m = r_planet / 1000
    area = np.pi * (r_m**2)
    v_tip = omega_rad * r_m
    loss_profile = 0.5 * rho_air * drag_coeff * area * v_tip**3
    return float(np.mean(loss_profile)), loss_profile


def bearing_loss(
    omega_rpm: float,
    f_radial: float,
    coeffs: tuple[float, float, float, float] = (0.01, 1e-4, 1e-6, 0.002),
) -> float:
    """Compute bearing loss torque and power.

    Model: T_loss = C0 + C1*w + C2*w^2 + C3*|F_rad|
    Power = T_loss * w

    Args:
        omega_rpm: Speed in RPM.
        f_radial: Radial load (N).
        coeffs: (C0, C1, C2, C3).

    Returns:
        Power loss (W).
    """
    omega_rad = omega_rpm * 2 * np.pi / 60
    c0, c1, c2, c3 = coeffs

    t_loss = c0 + c1 * omega_rad + c2 * (omega_rad**2) + c3 * abs(f_radial)
    return float(t_loss * omega_rad)


def churning_loss(
    omega_rpm: float,
    r_feature: float,
    submerged_factor: float = 0.0,
    coeffs: tuple[float, float] = (1e-5, 1e-7),
) -> float:
    """Compute churning/windage loss.

    Model: T_churn = k1 * w^2 + k2 * w^3 (approximated as Drag ~ v^2, Torque ~ v^2, Power ~ v^3)
    Actually, usually T ~ w^2.
    Here we use: Power = C_drag * w^3

    Args:
        omega_rpm: Speed in RPM.
        r_feature: Characteristic radius (mm).
        submerged_factor: 0.0=Air, 1.0=Oil.
        coeffs: (k_air, k_oil).

    Returns:
        Power loss (W).
    """
    omega_rad = omega_rpm * 2 * np.pi / 60
    r_m = r_feature / 1000.0
    v_tip = omega_rad * r_m

    k_air, k_oil = coeffs

    # Power ~ v^3
    p_air = k_air * (v_tip**3)
    p_oil = k_oil * (v_tip**3) * submerged_factor

    return float(p_air + p_oil)


def total_loss(
    theta: np.ndarray,
    r_planet: np.ndarray,
    rho_c: np.ndarray,
    omega_rpm: float,
    torque: float,
    enable_windage: bool = False,  # retained for backward compatibility flags
    enable_detailed: bool = False,
    loss_coeffs: dict[str, tuple[float, ...]] | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    # Extract Coeffs
    mu = 0.05
    c_bearing = (0.01, 1e-4, 1e-6, 0.002)
    c_churning = (1e-5, 1e-7)

    if loss_coeffs:
        if "mesh" in loss_coeffs:
            mu = loss_coeffs["mesh"][0]
        if "bearing" in loss_coeffs:
            c_bearing = loss_coeffs["bearing"]  # type: ignore
        if "churning" in loss_coeffs:
            c_churning = loss_coeffs["churning"]  # type: ignore

    # Mesh Loss
    lm, lp_m = mesh_loss(theta, r_planet, rho_c, omega_rpm, torque, mu=mu)

    lb = 0.0
    lc = 0.0

    if enable_detailed or enable_windage:
        # Approximate average radial load as proportional to torque/radius (simplification)
        r_avg = float(np.mean(r_planet)) / 1000.0
        f_rad = abs(torque) / max(r_avg, 1e-3)  # Simplified Fn ~ Fr assumption for now

        lb = bearing_loss(omega_rpm, f_rad, coeffs=c_bearing)
        lc = churning_loss(omega_rpm, r_planet.mean(), coeffs=c_churning)

    loss_profile = lp_m  # Mesh is the only profile-varying component usually dominated
    # We could distribute lb/lc across profile, but mean is fine for scalar tally

    total = lm + lb + lc

    diag = {
        "mesh_loss": lm,
        "bearing_loss": lb,
        "churning_loss": lc,
        "windage_loss": lc,  # Aliased for compatibility
    }
    return total, loss_profile, diag
