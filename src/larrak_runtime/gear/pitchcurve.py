"""Pitch curve representation and utilities.

This module provides Fourier-based pitch curve generation and utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PitchCurve:
    """Pitch curve representation in polar coordinates.

    Attributes:
        theta: Angle array (radians).
        r: Radius array (mm).
    """

    theta: np.ndarray
    r: np.ndarray

    @property
    def n_points(self) -> int:
        """Number of discretization points."""
        return len(self.theta)

    @property
    def min_radius(self) -> float:
        """Minimum radius."""
        return float(np.min(self.r))

    @property
    def max_radius(self) -> float:
        """Maximum radius."""
        return float(np.max(self.r))

    @property
    def mean_radius(self) -> float:
        """Mean radius."""
        return float(np.mean(self.r))


def fourier_pitch_curve(
    theta: np.ndarray,
    base_radius: float,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Generate pitch curve radius from Fourier coefficients.

    r(θ) = base_radius + Σ coeffs[k] * cos(k*θ) for k=1..n

    Args:
        theta: Angle array (radians).
        base_radius: Mean radius (mm).
        coeffs: Fourier coefficients for harmonics 1..n.

    Returns:
        Radius array (mm).
    """
    r = np.full_like(theta, base_radius, dtype=np.float64)

    for k, c in enumerate(coeffs, start=1):
        r += c * np.cos(k * theta)

    # Ensure positive radii
    return np.maximum(r, 1.0)


def spline_pitch_curve(
    theta: np.ndarray,
    control_radii: np.ndarray,
) -> np.ndarray:
    """Generate pitch curve via cubic spline interpolation.

    Args:
        theta: Evaluation angle array (radians).
        control_radii: Control point radii (equally spaced in angle).

    Returns:
        Interpolated radius array.
    """
    from scipy.interpolate import CubicSpline

    n_ctrl = len(control_radii)
    theta_ctrl = np.linspace(0, 2 * np.pi, n_ctrl, endpoint=False)

    # Periodic cubic spline
    cs = CubicSpline(theta_ctrl, control_radii, bc_type="periodic")
    return cs(theta % (2 * np.pi))


def pitch_curve_derivative(
    theta: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Compute dr/dθ using central differences.

    Args:
        theta: Angle array.
        r: Radius array.

    Returns:
        Derivative array dr/dθ.
    """
    return np.gradient(r, theta)


def pitch_curve_arc_length(
    theta: np.ndarray,
    r: np.ndarray,
) -> float:
    """Compute total arc length of pitch curve.

    L = ∫ √(r² + (dr/dθ)²) dθ

    Args:
        theta: Angle array.
        r: Radius array.

    Returns:
        Total arc length (mm).
    """
    dr = pitch_curve_derivative(theta, r)
    ds = np.sqrt(r**2 + dr**2)
    dtheta = np.diff(theta)

    # Trapezoidal integration
    ds_avg = (ds[:-1] + ds[1:]) / 2
    return float(np.sum(ds_avg * dtheta))
