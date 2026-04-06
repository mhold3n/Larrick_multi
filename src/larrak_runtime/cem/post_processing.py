"""Post-processing modifiers: coatings and heat treatment.

Models the effect of surface coatings and heat treatment on material
properties relevant to tribology and longevity.

Coating data sourced from:
- Oerlikon Balzers BALINIT CNI (CrN): max 700 °C, 18 GPa
- Platit ta-C (DLC3): max 450 °C, 35–55 GPa
- Scientific Reports Si-ta-C: stable to ~600 °C
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoatingType(Enum):
    """Surface coating families."""

    NONE = "none"
    CRN = "CrN"
    TA_C = "ta_C"
    SI_TA_C = "Si_ta_C"
    W_DLC_CRN = "W_DLC_CrN"


@dataclass(frozen=True)
class CoatingProperties:
    """Coating properties relevant to gear tribology.

    Attributes:
        name: Human-readable coating name.
        max_service_temp_C: Maximum continuous service temperature (°C).
        hardness_GPa: Nanoindentation hardness (GPa).
        friction_coeff_reduction: Fractional reduction in friction
            coefficient vs uncoated (0.0 = no change, 0.5 = halved).
        scuff_resistance_multiplier: Multiplier on scuffing critical
            temperature / margin (1.0 = no benefit).
        cost_tier: Relative coating process cost (0 = none, 1–5).
    """

    name: str
    max_service_temp_C: float
    hardness_GPa: float
    friction_coeff_reduction: float
    scuff_resistance_multiplier: float
    cost_tier: int


# ---------------------------------------------------------------------------
# Placeholder coating database
# ---------------------------------------------------------------------------

COATING_DB: dict[CoatingType, CoatingProperties] = {
    CoatingType.NONE: CoatingProperties(
        name="Uncoated",
        max_service_temp_C=999.0,
        hardness_GPa=0.0,
        friction_coeff_reduction=0.0,
        scuff_resistance_multiplier=1.0,
        cost_tier=0,
    ),
    CoatingType.CRN: CoatingProperties(
        name="CrN (PVD)",
        max_service_temp_C=700.0,
        hardness_GPa=18.0,
        friction_coeff_reduction=0.15,
        scuff_resistance_multiplier=1.5,
        cost_tier=2,
    ),
    CoatingType.TA_C: CoatingProperties(
        name="ta-C (hydrogen-free DLC)",
        max_service_temp_C=450.0,
        hardness_GPa=45.0,
        friction_coeff_reduction=0.40,
        scuff_resistance_multiplier=2.0,
        cost_tier=3,
    ),
    CoatingType.SI_TA_C: CoatingProperties(
        name="Si-doped ta-C",
        max_service_temp_C=600.0,
        hardness_GPa=40.0,
        friction_coeff_reduction=0.35,
        scuff_resistance_multiplier=2.0,
        cost_tier=4,
    ),
    CoatingType.W_DLC_CRN: CoatingProperties(
        name="W-DLC / CrN duplex",
        max_service_temp_C=500.0,
        hardness_GPa=25.0,
        friction_coeff_reduction=0.30,
        scuff_resistance_multiplier=2.5,
        cost_tier=5,
    ),
}


class HeatTreatment(Enum):
    """Heat treatment families."""

    CARBURIZED_STD = "carburized_standard"
    CARBURIZED_HOT_HARD = "carburized_hot_hard"
    NITRIDED = "nitrided"
    THROUGH_HARDENED = "through_hardened"


@dataclass(frozen=True)
class HeatTreatProperties:
    """Heat treatment effects on material behaviour.

    Attributes:
        name: Human-readable name.
        hardness_retention_300C: Fraction of room-temperature case
            hardness retained at 300 °C continuous exposure.
        distortion_risk: Qualitative distortion risk (0–1 scale).
        fatigue_bonus: Additive multiplier on fatigue life (e.g. 0.2
            means 20% longer life than untreated baseline).
        cost_multiplier: Relative cost vs standard carburizing.
    """

    name: str
    hardness_retention_300C: float
    distortion_risk: float
    fatigue_bonus: float
    cost_multiplier: float


HEAT_TREAT_DB: dict[HeatTreatment, HeatTreatProperties] = {
    HeatTreatment.CARBURIZED_STD: HeatTreatProperties(
        name="Standard carburizing (low temper)",
        hardness_retention_300C=0.85,
        distortion_risk=0.5,
        fatigue_bonus=0.0,
        cost_multiplier=1.0,
    ),
    HeatTreatment.CARBURIZED_HOT_HARD: HeatTreatProperties(
        name="Hot-hard carburizing (high temper)",
        hardness_retention_300C=0.97,
        distortion_risk=0.6,
        fatigue_bonus=0.3,
        cost_multiplier=1.5,
    ),
    HeatTreatment.NITRIDED: HeatTreatProperties(
        name="Gas / plasma nitriding",
        hardness_retention_300C=0.95,
        distortion_risk=0.2,
        fatigue_bonus=0.1,
        cost_multiplier=1.3,
    ),
    HeatTreatment.THROUGH_HARDENED: HeatTreatProperties(
        name="Through hardened (Q&T)",
        hardness_retention_300C=0.75,
        distortion_risk=0.7,
        fatigue_bonus=-0.1,
        cost_multiplier=0.8,
    ),
}


def get_coating(ct: CoatingType) -> CoatingProperties:
    """Look up coating properties."""
    return COATING_DB[ct]


def get_heat_treat(ht: HeatTreatment) -> HeatTreatProperties:
    """Look up heat treatment properties."""
    return HEAT_TREAT_DB[ht]


def apply_coating_modifiers(
    base_scuff_margin: float,
    base_friction: float,
    coating: CoatingType,
) -> tuple[float, float]:
    """Apply coating effects to scuffing margin and friction.

    Returns:
        (modified_scuff_margin, modified_friction_coeff)
    """
    cp = COATING_DB[coating]
    modified_margin = base_scuff_margin * cp.scuff_resistance_multiplier
    modified_friction = base_friction * (1.0 - cp.friction_coeff_reduction)
    return modified_margin, modified_friction


def apply_heat_treat_modifiers(
    base_fatigue_life_mult: float,
    heat_treat: HeatTreatment,
    operating_temp_C: float = 200.0,
) -> float:
    """Apply heat treatment effects to fatigue life multiplier.

    Accounts for hardness retention at operating temperature.

    Returns:
        Modified fatigue life multiplier.
    """
    ht = HEAT_TREAT_DB[heat_treat]
    # Temperature-adjusted fatigue bonus
    if operating_temp_C > 200.0:
        retention_factor = ht.hardness_retention_300C
    else:
        retention_factor = 1.0
    return base_fatigue_life_mult * retention_factor * (1.0 + ht.fatigue_bonus)


def coating_from_level(level: float) -> CoatingType:
    """Map continuous level (0–1) to a discrete coating type.

    Args:
        level: 0.0 = none, increasing → more advanced coatings.
    """
    if level < 0.15:
        return CoatingType.NONE
    elif level < 0.35:
        return CoatingType.CRN
    elif level < 0.55:
        return CoatingType.TA_C
    elif level < 0.75:
        return CoatingType.SI_TA_C
    else:
        return CoatingType.W_DLC_CRN
