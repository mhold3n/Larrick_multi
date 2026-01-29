"""Baseline analytic mesh friction losses.

Simple friction models for gear mesh losses without surrogates.
"""

from __future__ import annotations

import numpy as np


def sliding_velocity(
    omega1: float,
    omega2: float,
    r1: float,
    r2: float,
    pressure_angle: float = 20.0,
) -> float:
    """Compute sliding velocity at mesh point.

    Args:
        omega1: Angular velocity of gear 1 (rad/s).
        omega2: Angular velocity of gear 2 (rad/s).
        r1: Pitch radius of gear 1 (mm).
        r2: Pitch radius of gear 2 (mm).
        pressure_angle: Pressure angle (degrees).

    Returns:
        Sliding velocity (m/s).
    """
    # Convert to meters
    r1_m = r1 / 1000
    r2_m = r2 / 1000
    phi = np.radians(pressure_angle)

    # Simplified sliding velocity at pitch point
    v_pitch1 = omega1 * r1_m
    v_pitch2 = omega2 * r2_m

    # Sliding component (tangent to line of action)
    v_slide = abs(v_pitch1 - v_pitch2) * np.sin(phi)

    return float(v_slide)


def hertzian_contact_width(
    fn: float,
    r1: float,
    r2: float,
    e_star: float = 2.1e11,
    face_width: float = 20.0,
) -> float:
    """Compute Hertzian contact half-width.

    Args:
        fn: Normal force per unit width (N/mm).
        r1: Radius of curvature 1 (mm).
        r2: Radius of curvature 2 (mm).
        e_star: Equivalent modulus (Pa).
        face_width: Face width (mm).

    Returns:
        Contact half-width (mm).
    """
    # Equivalent radius
    r_eq = (r1 * r2) / (r1 + r2) if r1 + r2 > 0 else r1

    # Hertzian formula (simplified)
    # a = sqrt(4 * F * R / (π * E' * L))
    f_total = fn * face_width
    r_eq_m = r_eq / 1000
    l_m = face_width / 1000

    a_sq = 4 * f_total * r_eq_m / (np.pi * e_star * l_m)
    a = np.sqrt(max(a_sq, 0))

    return float(a * 1000)  # back to mm


def coulomb_friction_loss(
    fn: float,
    v_slide: float,
    mu: float = 0.05,
) -> float:
    """Compute Coulomb friction power loss.

    P_loss = μ * Fn * v_slide

    Args:
        fn: Normal force (N).
        v_slide: Sliding velocity (m/s).
        mu: Friction coefficient.

    Returns:
        Power loss (W).
    """
    return mu * fn * abs(v_slide)


def gear_mesh_efficiency(
    power_in: float,
    power_loss: float,
) -> float:
    """Compute gear mesh efficiency.

    Args:
        power_in: Input power (W).
        power_loss: Friction loss (W).

    Returns:
        Efficiency in [0, 1].
    """
    if power_in <= 0:
        return 1.0
    return max(0.0, (power_in - power_loss) / power_in)
