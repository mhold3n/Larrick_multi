"""Surface finish tiers and their effect on tribology / life.

Tiers are derived from:
- API 677 requirements (0.8 µm Ra at >20 m/s pitch-line velocity)
- REM/AGMA FZG testing (superfinished ≈0.10 µm Ra → negligible micropitting)
- NASA scuffing data (superfinished → significantly higher TOF)

Replace placeholder multipliers with validated dataset when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SurfaceFinishTier(Enum):
    """Surface finish quality tiers."""

    AS_GROUND = "as_ground"
    FINE_GROUND = "fine_ground"
    SUPERFINISHED = "superfinished"


@dataclass(frozen=True)
class FinishProperties:
    """Properties associated with a surface finish tier.

    Attributes:
        Ra_um: Typical arithmetic mean roughness (µm).
        Rz_um: Typical peak-to-valley roughness (µm).
        composite_roughness_factor: Multiplier on composite roughness
            relative to as-ground baseline (1.0).
        micropitting_life_multiplier: Relative micropitting endurance
            compared to as-ground (1.0).
        scuffing_TOF_multiplier: Relative time-to-failure under
            scuffing conditions (1.0 = as-ground).
        cost_multiplier: Relative processing cost (1.0 = as-ground).
    """

    Ra_um: float
    Rz_um: float
    composite_roughness_factor: float
    micropitting_life_multiplier: float
    scuffing_TOF_multiplier: float
    cost_multiplier: float


# ---------------------------------------------------------------------------
# Placeholder finish property table
# ---------------------------------------------------------------------------

FINISH_PROPERTIES: dict[SurfaceFinishTier, FinishProperties] = {
    SurfaceFinishTier.AS_GROUND: FinishProperties(
        Ra_um=0.50,
        Rz_um=3.0,
        composite_roughness_factor=1.0,
        micropitting_life_multiplier=1.0,
        scuffing_TOF_multiplier=1.0,
        cost_multiplier=1.0,
    ),
    SurfaceFinishTier.FINE_GROUND: FinishProperties(
        Ra_um=0.25,
        Rz_um=1.5,
        composite_roughness_factor=0.5,
        micropitting_life_multiplier=3.0,
        scuffing_TOF_multiplier=2.0,
        cost_multiplier=1.5,
    ),
    SurfaceFinishTier.SUPERFINISHED: FinishProperties(
        Ra_um=0.08,
        Rz_um=0.5,
        composite_roughness_factor=0.16,
        micropitting_life_multiplier=10.0,
        scuffing_TOF_multiplier=5.0,
        cost_multiplier=3.0,
    ),
}


def get_finish_properties(tier: SurfaceFinishTier) -> FinishProperties:
    """Look up finish properties by tier.

    Checks the DatasetRegistry for explicitly loaded experimental data.
    Falls back to FINISH_PROPERTIES if not overridden.
    """
    from larrak_runtime.cem.registry import get_registry

    reg = get_registry()
    table = reg.load_table("surface_finish_endurance")

    if "finish_method" in table and len(table["finish_method"]) > 0:
        for i, method in enumerate(table["finish_method"]):
            if method == tier.name or method == tier.value:
                return FinishProperties(
                    Ra_um=float(table.get("Ra_um", [0.0])[i]),
                    Rz_um=float(table.get("Rz_um", [0.0])[i]),
                    composite_roughness_factor=float(
                        table.get("composite_roughness_factor", [1.0])[i]
                    ),
                    micropitting_life_multiplier=float(
                        table.get("micropitting_life_multiplier", [1.0])[i]
                    ),
                    scuffing_TOF_multiplier=float(table.get("scuffing_TOF_multiplier", [1.0])[i]),
                    cost_multiplier=float(table.get("cost_multiplier", [1.0])[i]),
                )

    return FINISH_PROPERTIES[tier]


def effective_composite_roughness(tier: SurfaceFinishTier, base_Ra_um: float = 0.50) -> float:
    """Compute effective composite roughness for a finish tier.

    Args:
        tier: Surface finish quality tier.
        base_Ra_um: Baseline Ra for as-ground condition (µm).

    Returns:
        Effective composite roughness σ* (µm) for use in λ calculations.
        Assumes both flanks have the same finish (σ* = √2 · Ra).
    """
    props = FINISH_PROPERTIES[tier]
    Ra = base_Ra_um * props.composite_roughness_factor
    # Composite roughness for two surfaces: σ* = sqrt(Ra1² + Ra2²)
    # Assuming both flanks same finish: σ* = Ra * sqrt(2)
    return float(Ra * (2.0**0.5))


def tier_from_level(level: float) -> SurfaceFinishTier:
    """Map a continuous level (0–1) to a discrete finish tier.

    Args:
        level: 0.0 = as-ground, 0.5 = fine-ground, 1.0 = superfinished.

    Returns:
        Nearest SurfaceFinishTier.
    """
    if level < 0.33:
        return SurfaceFinishTier.AS_GROUND
    elif level < 0.67:
        return SurfaceFinishTier.FINE_GROUND
    else:
        return SurfaceFinishTier.SUPERFINISHED
