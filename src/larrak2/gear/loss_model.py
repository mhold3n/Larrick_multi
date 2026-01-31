"""Modular loss components for gear system."""

from __future__ import annotations

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


def total_loss(theta, r_planet, rho_c, omega_rpm, torque, enable_windage=False):
    lm, lp_m = mesh_loss(theta, r_planet, rho_c, omega_rpm, torque)
    lw = 0.0
    lp_w = np.zeros_like(lp_m)
    if enable_windage:
        lw, lp_w = windage_loss(theta, r_planet, omega_rpm)

    total_profile = lp_m + lp_w
    total = max(lm + lw, 0.0)

    diag = {
        "mesh_loss": lm,
        "windage_loss": lw,
        "enable_windage": enable_windage,
    }
    return total, total_profile, diag
