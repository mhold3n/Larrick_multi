"""V1 config mapping to larrak2.

Maps v1 constants and config values to larrak2 schema.
Avoids importing v1 modules by hardcoding known defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

# ==============================================================================
# Hardcoded v1 constants (from campro/constants.py)
# ==============================================================================

# Wiebe parameters
DEFAULT_WIEBE_M = 2.0  # shape factor
DEFAULT_WIEBE_A = 6.9  # efficiency factor (99% burn complete)

# Combustion timing
DEFAULT_COMBUSTION_START_DEG = 10.0
DEFAULT_COMBUSTION_DURATION_DEG = 60.0

# Thermodynamic
GAMMA_AIR = 1.4
GAMMA_COMBUSTION = 1.3
R_SPECIFIC_AIR = 287.0  # J/(kg·K)
P_ATM = 101325.0  # Pa
T_REF = 300.0  # K

# Gear/Litvin
DEFAULT_MODULE = 1.0  # mm
DEFAULT_PRESSURE_ANGLE_DEG = 20.0
MIN_TEETH = 8
MU_MESH = 0.05  # friction coefficient

# Flame speed
FLAME_SPEED_REF_TEMP = 300.0  # K
FLAME_SPEED_REF_PRESSURE = 101325.0  # Pa
FLAME_SPEED_TEMP_EXPONENT = 2.0
FLAME_SPEED_PRESSURE_EXPONENT = -0.5

# Ignition delay
IGNITION_DELAY_PRE_EXP = 1e-6  # s
IGNITION_ACTIVATION_ENERGY = 125000.0  # J/mol


@dataclass
class V1CombustionConfig:
    """V1 combustion configuration mapped to larrak2."""

    m_wiebe: float = DEFAULT_WIEBE_M
    a_wiebe: float = DEFAULT_WIEBE_A
    theta_start: float = DEFAULT_COMBUSTION_START_DEG
    theta_duration: float = DEFAULT_COMBUSTION_DURATION_DEG
    gamma: float = GAMMA_COMBUSTION


@dataclass
class V1GearConfig:
    """V1 gear configuration mapped to larrak2."""

    module: float = DEFAULT_MODULE
    pressure_angle_deg: float = DEFAULT_PRESSURE_ANGLE_DEG
    min_teeth: int = MIN_TEETH
    mu_mesh: float = MU_MESH


def get_v1_combustion_config(fuel_type: str = "gasoline") -> V1CombustionConfig:
    """Get combustion config for a fuel type.

    Fuel-specific configs from campro/physics/chem.py create_combustion_parameters()
    """
    configs = {
        "gasoline": V1CombustionConfig(
            m_wiebe=2.0,
            a_wiebe=6.9,
            theta_start=10.0,
            theta_duration=60.0,
        ),
        "diesel": V1CombustionConfig(
            m_wiebe=1.5,
            a_wiebe=6.9,
            theta_start=5.0,
            theta_duration=80.0,
        ),
        "natural_gas": V1CombustionConfig(
            m_wiebe=2.5,
            a_wiebe=4.0,
            theta_start=15.0,
            theta_duration=70.0,
        ),
        "hydrogen": V1CombustionConfig(
            m_wiebe=3.0,
            a_wiebe=3.0,
            theta_start=20.0,
            theta_duration=50.0,
        ),
    }
    return configs.get(fuel_type, configs["gasoline"])


def get_v1_gear_config() -> V1GearConfig:
    """Get default gear config from v1."""
    return V1GearConfig()
