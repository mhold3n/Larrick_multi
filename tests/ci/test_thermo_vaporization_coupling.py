"""Fuel vaporization coupling checks."""

from __future__ import annotations

import numpy as np

from larrak2.thermo.combustion import burn_increment
from larrak2.thermo.vaporization import integrate_vapor_fraction


def test_vaporization_speed_increases_with_temperature() -> None:
    theta = np.linspace(0.0, 359.0, 360)
    p = np.full_like(theta, 2.0e5)

    vapor_cold = integrate_vapor_fraction(
        theta,
        np.full_like(theta, 350.0),
        p,
        rpm=2500.0,
        initial_vapor_fraction=0.1,
        tau_ref_s=0.004,
        temp_exponent=-1.2,
        pressure_exponent=0.2,
    )
    vapor_hot = integrate_vapor_fraction(
        theta,
        np.full_like(theta, 700.0),
        p,
        rpm=2500.0,
        initial_vapor_fraction=0.1,
        tau_ref_s=0.004,
        temp_exponent=-1.2,
        pressure_exponent=0.2,
    )

    assert float(vapor_hot[-1]) > float(vapor_cold[-1])


def test_burn_increment_limited_by_vapor_fraction() -> None:
    theta = np.linspace(0.0, 359.0, 360)
    burn_profile = np.linspace(0.0, 1.0, theta.size)

    dx_low = burn_increment(
        burn_profile,
        oxygen_completion_cap=1.0,
        vapor_fraction=0.2,
    )
    dx_high = burn_increment(
        burn_profile,
        oxygen_completion_cap=1.0,
        vapor_fraction=0.9,
    )

    assert np.sum(dx_high) > np.sum(dx_low)
