"""Candidate encoding and decoding.

This module defines the parameter structure and provides pack/unpack
functions for the flat decision vector x.

Layout (ENCODING_VERSION = "0.2"):
    x[0:5]   - ThermoParams (5 floats)
    x[5:13]  - GearParams (8 floats)
    Total: 13 decision variables
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENCODING_VERSION = "0.2"

# Decision variable count
N_THERMO = 5
N_GEAR = 8
N_TOTAL = N_THERMO + N_GEAR


@dataclass
class ThermoParams:
    """Thermodynamic control parameters.

    Attributes:
        compression_duration: Compression phase duration (degrees), [30, 90].
        expansion_duration: Expansion phase duration (degrees), [60, 120].
        heat_release_center: Center of heat release (degrees), [0, 30].
        heat_release_width: Heat release duration (degrees), [10, 60].
        lambda_af: Air-fuel equivalence ratio λ [-], where λ < 1 is rich and λ > 1 is lean.
    """

    compression_duration: float
    expansion_duration: float
    heat_release_center: float
    heat_release_width: float
    lambda_af: float

    def to_array(self) -> np.ndarray:
        """Convert to flat array."""
        return np.array(
            [
                self.compression_duration,
                self.expansion_duration,
                self.heat_release_center,
                self.heat_release_width,
                self.lambda_af,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> ThermoParams:
        """Create from flat array."""
        return cls(
            compression_duration=float(arr[0]),
            expansion_duration=float(arr[1]),
            heat_release_center=float(arr[2]),
            heat_release_width=float(arr[3]),
            lambda_af=float(arr[4]),
        )


@dataclass
class GearParams:
    """Gear synthesis control parameters.

    Attributes:
        base_radius: Base circle radius (mm), [20, 60].
        pitch_coeffs: Fourier coefficients for pitch curve (7 floats).
    """

    base_radius: float
    pitch_coeffs: np.ndarray  # 7 coefficients

    def to_array(self) -> np.ndarray:
        """Convert to flat array."""
        return np.concatenate([[self.base_radius], self.pitch_coeffs])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> GearParams:
        """Create from flat array."""
        return cls(
            base_radius=float(arr[0]),
            pitch_coeffs=np.array(arr[1:], dtype=np.float64),
        )


@dataclass
class Candidate:
    """Complete candidate solution.

    Combines thermo and gear parameters into a single structured object.
    """

    thermo: ThermoParams
    gear: GearParams


def decode_candidate(x: np.ndarray) -> Candidate:
    """Decode flat array to structured Candidate.

    Args:
        x: Flat decision vector of length N_TOTAL.

    Returns:
        Candidate with ThermoParams and GearParams.
    """
    if len(x) != N_TOTAL:
        raise ValueError(f"Expected {N_TOTAL} variables, got {len(x)}")

    thermo = ThermoParams.from_array(x[:N_THERMO])
    gear = GearParams.from_array(x[N_THERMO:])

    return Candidate(thermo=thermo, gear=gear)


def encode_candidate(candidate: Candidate) -> np.ndarray:
    """Encode structured Candidate to flat array.

    Args:
        candidate: Structured candidate solution.

    Returns:
        Flat decision vector of length N_TOTAL.
    """
    return np.concatenate(
        [candidate.thermo.to_array(), candidate.gear.to_array()],
        dtype=np.float64,
    )


def bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds for decision variables.

    Returns:
        (xl, xu) tuple of bound arrays, each of length N_TOTAL.
    """
    # ThermoParams bounds
    # compression, expansion, hr_center, hr_width, lambda_af
    thermo_lb = np.array([30.0, 60.0, 0.0, 10.0, 0.6])
    thermo_ub = np.array([90.0, 120.0, 30.0, 60.0, 1.6])

    # GearParams bounds: base_radius + 7 pitch coefficients
    gear_lb = np.array([20.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    gear_ub = np.array([60.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    xl = np.concatenate([thermo_lb, gear_lb])
    xu = np.concatenate([thermo_ub, gear_ub])

    return xl, xu


def random_candidate(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random candidate within bounds.

    Args:
        rng: Random number generator (uses default if None).

    Returns:
        Random decision vector within bounds.
    """
    if rng is None:
        rng = np.random.default_rng()

    xl, xu = bounds()
    return rng.uniform(xl, xu)


def mid_bounds_candidate() -> np.ndarray:
    """Return candidate at midpoint of bounds (likely feasible for toy physics)."""
    xl, xu = bounds()
    return (xl + xu) / 2
