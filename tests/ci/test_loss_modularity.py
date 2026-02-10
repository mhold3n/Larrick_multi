"""Tests for modular loss model toggles."""

import numpy as np

from larrak2.gear.loss_model import total_loss


def test_loss_components_sum():
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r_planet = np.full_like(theta, 50.0)
    rho_c = np.full_like(theta, 80.0)

    total, profile, diag = total_loss(theta, r_planet, rho_c, omega_rpm=3000, torque=200, enable_windage=True)
    mesh_only, _, diag_mesh = total_loss(theta, r_planet, rho_c, omega_rpm=3000, torque=200, enable_windage=False)

    assert total >= mesh_only
    assert total >= 0.0
    assert np.all(profile >= 0.0)
    # Component breakdown present
    assert "mesh_loss" in diag
    assert "windage_loss" in diag
