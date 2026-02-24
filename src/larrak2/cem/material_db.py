"""Material property registry for gear substrate alloys.

Placeholder table values sourced from:
- NASA Glenn M50NiL vs 9310 endurance tests (10k rpm, 1.71 GPa)
- Carpenter CBS-50 NiL datasheet (service temp ≤316 °C)
- Pyrowear 53 datasheet (case hardness vs temper temperature)
- Ferrium C61/C64 NASA single-tooth bending fatigue data

Replace placeholder values with validated experimental data via the
DatasetRegistry when datasets are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MaterialClass(Enum):
    """Candidate gear substrate alloys."""

    AISI_9310 = "AISI_9310"
    PYROWEAR_53 = "Pyrowear_53"
    CBS50_NIL = "CBS50_NiL"
    M50_NIL = "M50NiL"
    FERRIUM_C64 = "Ferrium_C64"


@dataclass(frozen=True)
class MaterialProperties:
    """Gear substrate material properties.

    Attributes:
        name: Human-readable alloy name.
        max_service_temp_C: Maximum recommended continuous service
            temperature (°C) before hardness degradation.
        case_hardness_HRC: Typical case hardness after carburizing and
            tempering at the service temperature.
        core_hardness_HRC: Typical core hardness.
        fatigue_life_multiplier: Surface fatigue life relative to
            AISI 9310 baseline (1.0 = 9310 level).
        youngs_modulus_GPa: Young's Modulus (E) in GPa.
        poissons_ratio: Poisson's ratio (ν).
        cost_tier: Relative cost index (1 = cheapest, 5 = most expensive).
    """

    name: str
    max_service_temp_C: float
    case_hardness_HRC: float
    core_hardness_HRC: float
    fatigue_life_multiplier: float
    youngs_modulus_GPa: float
    poissons_ratio: float
    cost_tier: int


# ---------------------------------------------------------------------------
# Placeholder material database
# Values from research docs — replace with validated datasets.
# ---------------------------------------------------------------------------

MATERIAL_DB: dict[MaterialClass, MaterialProperties] = {
    MaterialClass.AISI_9310: MaterialProperties(
        name="AISI 9310 (baseline)",
        max_service_temp_C=200.0,
        case_hardness_HRC=60.0,
        core_hardness_HRC=37.0,
        fatigue_life_multiplier=1.0,
        youngs_modulus_GPa=205.0,
        poissons_ratio=0.29,
        cost_tier=1,
    ),
    MaterialClass.PYROWEAR_53: MaterialProperties(
        name="Pyrowear 53",
        max_service_temp_C=288.0,
        case_hardness_HRC=61.0,
        core_hardness_HRC=35.0,
        fatigue_life_multiplier=2.5,
        youngs_modulus_GPa=200.0,
        poissons_ratio=0.30,
        cost_tier=2,
    ),
    MaterialClass.CBS50_NIL: MaterialProperties(
        name="CBS-50 NiL (Carpenter)",
        max_service_temp_C=316.0,
        case_hardness_HRC=62.0,
        core_hardness_HRC=40.0,
        fatigue_life_multiplier=4.5,
        youngs_modulus_GPa=195.0,
        poissons_ratio=0.30,
        cost_tier=3,
    ),
    MaterialClass.M50_NIL: MaterialProperties(
        name="M50NiL (VIM-VAR)",
        max_service_temp_C=316.0,
        case_hardness_HRC=63.0,
        core_hardness_HRC=42.0,
        fatigue_life_multiplier=11.5,
        youngs_modulus_GPa=202.0,
        poissons_ratio=0.29,
        cost_tier=4,
    ),
    MaterialClass.FERRIUM_C64: MaterialProperties(
        name="Ferrium C64",
        max_service_temp_C=300.0,
        case_hardness_HRC=63.0,
        core_hardness_HRC=49.0,
        fatigue_life_multiplier=6.0,
        youngs_modulus_GPa=207.0,
        poissons_ratio=0.28,
        cost_tier=5,
    ),
}


def get_material(cls: MaterialClass) -> MaterialProperties:
    """Look up material properties by class.

    First queries the DatasetRegistry for explicitly loaded experimental data.
    If no experimental override exists for this alloy, falls back to the
    theoretical baseline value defined in MATERIAL_DB.

    Raises:
        KeyError: If material class is not in the database and no dataset covers it.
    """
    from larrak2.cem.registry import get_registry

    reg = get_registry()
    table = reg.load_table("material_properties")

    # Check if experimental data has been populated
    if "alloy" in table and len(table["alloy"]) > 0:
        for i, alloy_name in enumerate(table["alloy"]):
            if alloy_name == cls.name or alloy_name == cls.value:
                return MaterialProperties(
                    name=alloy_name,
                    max_service_temp_C=float(table["max_service_temp_C"][i]),
                    case_hardness_HRC=float(table["case_hardness_HRC"][i]),
                    core_hardness_HRC=float(table["core_hardness_HRC"][i]),
                    fatigue_life_multiplier=float(table["fatigue_life_multiplier"][i]),
                    youngs_modulus_GPa=float(table.get("youngs_modulus_GPa", [MATERIAL_DB[cls].youngs_modulus_GPa])[min(i, len(table.get("youngs_modulus_GPa", [])) - 1)] if len(table.get("youngs_modulus_GPa", [])) > 0 else MATERIAL_DB[cls].youngs_modulus_GPa),
                    poissons_ratio=float(table.get("poissons_ratio", [MATERIAL_DB[cls].poissons_ratio])[min(i, len(table.get("poissons_ratio", [])) - 1)] if len(table.get("poissons_ratio", [])) > 0 else MATERIAL_DB[cls].poissons_ratio),
                    cost_tier=int(table["cost_tier"][i]),
                )

    # Fallback to theoretical default
    return MATERIAL_DB[cls]


def list_materials() -> list[MaterialClass]:
    """Return all available material classes."""
    return list(MATERIAL_DB.keys())
