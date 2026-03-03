"""Core constants and configuration for Larrak2.

This module defines system-wide invariants such as:
- Discretization size (N_THETA)
- Model version hashes (for caching)
- Physical constants (if universal)
"""

from __future__ import annotations

# Discretization
# Number of points for a full 360-degree crank rotation
N_THETA = 360

# Model Versioning for Caching
# Update these when the underlying physics logic changes
# format: "v{version}_{date}_{githash_short}" if possible, or manual string
MODEL_VERSION_THERMO_V1 = "v2.0_20260228_two_zone_eq"
MODEL_VERSION_GEAR_V1 = "v1.0_20260125_initial"

# Physical constants (universal)
P_ATM = 101325.0  # Pa
T_REF = 298.15  # K

# Constraint/tolerance knobs
RATIO_SLOPE_LIMIT_FID0 = 0.8
RATIO_SLOPE_LIMIT_FID1 = 2.0
GEAR_INTERFERENCE_CLEARANCE_MM = 0.0
GEAR_MIN_THICKNESS_MM = -0.05

# Real-world constraint targets
REALWORLD_LAMBDA_MIN_TARGET = 1.0  # Full EHL target
REALWORLD_SCUFF_MARGIN_MIN = 0.0  # °C minimum margin
