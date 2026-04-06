"""Lubrication delivery modeling for gear cooling and film formation.

Models oil delivery modes and their effect on film thickness, cooling,
and churning losses.  Calibrated to high-speed gear operating regimes
per NASA oil-jet and API 677 guidance.

References:
    NASA oil-jet lubrication studies (radial jets, cooling effectiveness)
    API 677 (dip avoidance at >15 m/s pitch-line velocity)
    NASA windage/churning loss work (jet vs sump)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class LubricationMode(Enum):
    """Oil delivery architecture."""

    DRY = "dry"
    SPLASH_BATH = "splash_bath"
    PRESSURIZED_JET = "pressurized_jet"
    PHASE_GATED_JET = "phase_gated_jet"


@dataclass(frozen=True)
class LubricationParams:
    """Lubrication system operating parameters.

    Attributes:
        mode: Oil delivery architecture.
        supply_temp_C: Oil supply temperature (°C).
        flow_rate_L_min: Volumetric flow rate (L/min).
        viscosity_40C_cSt: Kinematic viscosity at 40 °C (cSt).
        viscosity_100C_cSt: Kinematic viscosity at 100 °C (cSt).
        vi_index: Viscosity index (dimensionless).
    """

    mode: LubricationMode = LubricationMode.PRESSURIZED_JET
    supply_temp_C: float = 80.0
    flow_rate_L_min: float = 2.0
    viscosity_40C_cSt: float = 68.0
    viscosity_100C_cSt: float = 10.0
    vi_index: float = 150.0


# ---------------------------------------------------------------------------
# Viscosity–temperature approximation (Walther equation simplified)
# ---------------------------------------------------------------------------


def effective_viscosity(params: LubricationParams, contact_temp_C: float) -> float:
    """Estimate viscosity at contact inlet temperature.

    Uses log-linear interpolation between the two reference temperatures
    (simplified Walther equation).

    Args:
        params: Lubrication parameters with reference viscosities.
        contact_temp_C: Temperature at the contact inlet (°C).

    Returns:
        Estimated kinematic viscosity (cSt) at contact_temp_C.
    """
    if params.viscosity_40C_cSt <= 0 or params.viscosity_100C_cSt <= 0:
        return 1.0

    # Log-linear interpolation in (1/T, log(ν)) space
    T1, T2 = 40.0, 100.0
    v1 = max(params.viscosity_40C_cSt, 0.5)
    v2 = max(params.viscosity_100C_cSt, 0.5)

    if abs(T2 - T1) < 1e-6:
        return v1

    # Slope in log-space
    slope = np.log(v2 / v1) / (T2 - T1)

    # Extrapolate (with clamping for extreme temperatures)
    T_clamp = float(np.clip(contact_temp_C, -20.0, 400.0))
    visc = v1 * np.exp(slope * (T_clamp - T1))

    return float(np.clip(visc, 0.5, 1e4))


# ---------------------------------------------------------------------------
# Cooling effectiveness model
# ---------------------------------------------------------------------------

# Mode-dependent cooling effectiveness factors (0–1 scale).
# Higher = better cooling capability at the contact zone.
_COOLING_BASE: dict[LubricationMode, float] = {
    LubricationMode.DRY: 0.0,
    LubricationMode.SPLASH_BATH: 0.3,
    LubricationMode.PRESSURIZED_JET: 0.75,
    LubricationMode.PHASE_GATED_JET: 0.90,
}


def cooling_effectiveness(params: LubricationParams, pitch_line_vel_m_s: float) -> float:
    """Compute normalized cooling effectiveness (0–1).

    Higher values indicate better thermal management at the mesh.
    Accounts for:
    - Oil delivery mode (jets >> splash >> dry)
    - Flow rate (more oil = more cooling, diminishing returns)
    - Pitch-line velocity (bath loses effectiveness at high speed)

    Args:
        params: Lubrication system parameters.
        pitch_line_vel_m_s: Pitch-line velocity (m/s).

    Returns:
        Normalized cooling effectiveness (0.0–1.0).
    """
    base = _COOLING_BASE.get(params.mode, 0.0)

    # Flow rate factor (diminishing returns above ~5 L/min)
    flow_factor = 1.0 - np.exp(-params.flow_rate_L_min / 3.0) if params.flow_rate_L_min > 0 else 0.0

    # Speed penalty for bath/splash (API 677: avoid dip above ~15 m/s)
    speed_penalty = 1.0
    if params.mode == LubricationMode.SPLASH_BATH and pitch_line_vel_m_s > 10.0:
        speed_penalty = max(0.1, 1.0 - 0.05 * (pitch_line_vel_m_s - 10.0))

    effectiveness = base * float(flow_factor) * speed_penalty
    return float(np.clip(effectiveness, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Churning loss factor
# ---------------------------------------------------------------------------

# Churning cost factors by mode (relative power loss).
_CHURN_BASE: dict[LubricationMode, float] = {
    LubricationMode.DRY: 0.0,
    LubricationMode.SPLASH_BATH: 1.0,
    LubricationMode.PRESSURIZED_JET: 0.15,
    LubricationMode.PHASE_GATED_JET: 0.10,
}


def churning_loss_factor(params: LubricationParams, pitch_line_vel_m_s: float) -> float:
    """Compute relative churning/windage loss factor.

    Scale: 0.0 = no churning loss, 1.0 = maximum bath churning at speed.
    Jet-lubricated designs avoid sump contact → much lower churning.

    Args:
        params: Lubrication system parameters.
        pitch_line_vel_m_s: Pitch-line velocity (m/s).

    Returns:
        Relative churning loss factor (0.0–1.0+).
    """
    base = _CHURN_BASE.get(params.mode, 0.5)

    # Churning scales roughly with velocity² for bath contact
    if params.mode == LubricationMode.SPLASH_BATH:
        vel_factor = (pitch_line_vel_m_s / 10.0) ** 2
    else:
        vel_factor = pitch_line_vel_m_s / 30.0  # Windage (lower scaling)

    return float(np.clip(base * vel_factor, 0.0, 5.0))


def mode_from_level(level: float) -> LubricationMode:
    """Map a continuous level (0–1) to a discrete lubrication mode.

    Args:
        level: 0.0 = dry, 0.33 = splash/bath, 0.67 = pressurized jet,
               1.0 = phase-gated jet.

    Returns:
        Nearest LubricationMode.
    """
    if level < 0.17:
        return LubricationMode.DRY
    elif level < 0.50:
        return LubricationMode.SPLASH_BATH
    elif level < 0.83:
        return LubricationMode.PRESSURIZED_JET
    else:
        return LubricationMode.PHASE_GATED_JET
