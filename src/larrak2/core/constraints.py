"""Centralized constraint assembly and scaling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Default per-constraint scaling to keep magnitudes comparable.

DEFAULT_SCALES: dict[str, float] = {
    "thermo_compression_min": 10.0,  # degrees
    "thermo_heat_release_width_min": 10.0,  # degrees
    "thermo_ratio_jerk_max": 1e6,  # jerk cap
    "thermo_eff_min": 1.0,
    "thermo_eff_max": 1.0,
    "thermo_pmax_norm": 1.0,
    "system_power_balance": 1e4,  # W (10 kW scale)
    "gear_ratio_error_max": 1.0,
    "gear_min_radius": 10.0,  # mm
    "gear_max_radius": 10.0,  # mm
    "gear_max_curvature": 0.5,  # 1/mm
    "gear_interference": 1.0,
    "gear_min_thickness": 5.0,  # placeholder scaling
    "gear_contact_ratio_min": 1.0,
    "gear_self_intersection": 1.0,
    "gear_stress_hotspot": 500.0,  # MPa headroom scale
    "tol_budget": 10.0,  # weighted penalty
    "tooling_cost": 1.0,  # weighted penalty
    # Real-world constraints
    "rw_lambda_min": 1.0,  # dimensionless (λ)
    "rw_scuff_margin": 100.0,  # °C scale
    "rw_micropitting_sf": 1.0,  # safety factor
    "rw_material_temp": 100.0,  # °C scale
    "rw_cost_index": 5.0,  # cost units
    # life damage is compressed upstream with log10(D_total), so this stays dimensionless
    "rw_life_damage_10k": 1.0,
    "rw_material_snap_dist": 0.4,  # normalized material-route distance
}

DEFAULT_KIND: dict[str, str] = {
    # Thermo
    "thermo_compression_min": "hard",
    "thermo_heat_release_width_min": "hard",
    "thermo_ratio_jerk_max": "soft",
    "thermo_ratio_slope_max": "soft",
    "thermo_eff_min": "hard",
    "thermo_eff_max": "hard",
    "thermo_pmax_norm": "hard",
    "system_power_balance": "hard",
    # Gear
    "gear_ratio_error_max": "soft",
    "gear_min_radius": "hard",
    "gear_max_radius": "hard",
    "gear_max_curvature": "soft",
    "gear_interference": "soft",
    "gear_min_thickness": "hard",
    "gear_contact_ratio_min": "soft",
    "gear_self_intersection": "soft",
    "gear_stress_hotspot": "hard",
    # Machining
    "tol_budget": "hard",
    "tooling_cost": "soft",
    # Real-world
    # Strategic choice: material constraints are soft during geometry exploration.
    # They are still tracked and reported, but do not block Pareto feasibility.
    "rw_lambda_min": "soft",
    "rw_scuff_margin": "soft",
    "rw_micropitting_sf": "soft",
    "rw_material_temp": "soft",
    "rw_cost_index": "soft",  # Cost is a soft preference
    "rw_life_damage_10k": "soft",
    "rw_material_snap_dist": "soft",
}

DEFAULT_REASON: dict[str, str] = {
    "thermo_compression_min": "compression duration below minimum",
    "thermo_heat_release_width_min": "heat release width below minimum",
    "thermo_ratio_jerk_max": "ratio profile jerk too high",
    "thermo_ratio_slope_max": "ratio profile slope too high",
    "thermo_eff_min": "efficiency below zero",
    "thermo_eff_max": "efficiency above physical bound",
    "thermo_pmax_norm": "peak pressure exceeds limit",
    "system_power_balance": "indicated power below demanded (including gear loss)",
    "gear_ratio_error_max": "gear ratio tracking error too high",
    "gear_min_radius": "planet radius below minimum",
    "gear_max_radius": "planet radius above maximum",
    "gear_max_curvature": "curvature exceeds limit",
    "gear_interference": "ring/planet interference detected",
    "gear_min_thickness": "tooth thickness below minimum",
    "gear_contact_ratio_min": "contact ratio below 1.0",
    "gear_self_intersection": "psi mapping non-monotonic (self-intersection risk)",
    "gear_stress_hotspot": "radius strategy induces excessive local stress concentration",
    "tol_budget": "tolerance budget violation (requires tighter tolerances than allowed)",
    "tooling_cost": "excessive tooling cost (requires non-standard/micro tools)",
    # Real-world
    "rw_lambda_min": "specific film thickness below full EHL target (λ < 1.0)",
    "rw_scuff_margin": "scuffing temperature margin insufficient",
    "rw_micropitting_sf": "micropitting safety factor below 1.0",
    "rw_material_temp": "operating temperature exceeds material service limit",
    "rw_cost_index": "combined material/surface/coating cost exceeds threshold",
    "rw_life_damage_10k": "accumulated damage exceeds 10,000 h service life",
    "rw_material_snap_dist": "material-state request too far from feasible manufacturing routes",
}


THERMO_CONSTRAINTS_FID0 = [
    "thermo_eff_min",
    "thermo_eff_max",
    "thermo_pmax_norm",
    "thermo_ratio_slope_max",
    "system_power_balance",
]

THERMO_CONSTRAINTS_FID1 = [
    "thermo_eff_min",
    "thermo_eff_max",
    "thermo_pmax_norm",
    "thermo_ratio_slope_max",
    "system_power_balance",
]

GEAR_CONSTRAINTS = [
    "gear_ratio_error_max",
    "gear_min_radius",
    "gear_max_radius",
    "gear_max_curvature",
    "gear_interference",
    "gear_min_thickness",
    "gear_contact_ratio_min",
    "gear_self_intersection",
    "gear_stress_hotspot",
]

MACHINING_CONSTRAINTS = [
    "tol_budget",
    "tooling_cost",
]

REALWORLD_CONSTRAINTS = [
    "rw_lambda_min",
    "rw_scuff_margin",
    "rw_micropitting_sf",
    "rw_material_temp",
    "rw_cost_index",
    "rw_life_damage_10k",
    "rw_material_snap_dist",
]

MATERIAL_CONSTRAINTS = [
    "rw_lambda_min",
    "rw_scuff_margin",
    "rw_micropitting_sf",
    "rw_material_temp",
    "rw_cost_index",
    "rw_life_damage_10k",
    "rw_material_snap_dist",
]


def get_constraint_names(fidelity: int) -> list[str]:
    """Return ordered constraint names for a given fidelity."""
    thermo = THERMO_CONSTRAINTS_FID1 if fidelity >= 1 else THERMO_CONSTRAINTS_FID0
    return (
        list(thermo)
        + list(GEAR_CONSTRAINTS)
        + list(MACHINING_CONSTRAINTS)
        + list(REALWORLD_CONSTRAINTS)
    )


def get_constraint_scales() -> dict[str, float]:
    """Expose default constraint scales for downstream metadata."""
    return DEFAULT_SCALES.copy()


def get_constraint_reasons() -> dict[str, str]:
    """Expose default constraint rationale/reason text."""
    return DEFAULT_REASON.copy()


def get_constraint_kinds() -> dict[str, str]:
    """Expose default hard/soft metadata for downstream diagnostics."""
    return DEFAULT_KIND.copy()


def get_constraint_kinds_for_phase(phase: str = "explore") -> dict[str, str]:
    """Return constraint kinds for an optimization phase.

    Phases:
        - explore: soft manufacturability/material constraints to preserve geometry coverage
        - downselect: promote manufacturability/material constraints to hard gates
    """
    kinds = DEFAULT_KIND.copy()

    # Gear validity constraints are always hard. Softening these can mask
    # true geometry errors and produce misleading downstream results.
    for name in GEAR_CONSTRAINTS:
        kinds[name] = "hard"

    # Keep manufacturability feasibility hard in both phases.
    kinds["tol_budget"] = "hard"

    if phase == "explore":
        # Explore mode softens material/lifetime only.
        for name in MATERIAL_CONSTRAINTS:
            kinds[name] = "soft"
    elif phase == "downselect":
        # Downselect mode enforces all material/lifetime constraints.
        for name in MATERIAL_CONSTRAINTS:
            kinds[name] = "hard"
    else:
        raise ValueError(f"Unsupported constraint phase '{phase}'. Use 'explore' or 'downselect'.")

    return kinds


def get_material_constraint_names() -> list[str]:
    """Expose constraint names tied to material/lubrication behavior."""
    return list(MATERIAL_CONSTRAINTS)


@dataclass
class ConstraintRecord:
    name: str
    raw: float
    scale: float
    scaled: float

    @property
    def feasible(self) -> bool:
        return self.scaled <= 0.0


def combine_constraints(
    thermo_G: Sequence[float],
    gear_G: Sequence[float],
    thermo_names: Sequence[str],
    gear_names: Sequence[str],
    scale_overrides: Mapping[str, float] | None = None,
    kind_overrides: Mapping[str, str] | None = None,
    realworld_G: Sequence[float] | None = None,
    realworld_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Merge thermo + gear + realworld constraints with scaling and reason codes.

    Hard vs soft enforcement:
        - hard: G_i = scaled raw value (positive => infeasible)
        - soft: G_i = min(scaled raw value, 0.0) so it never blocks feasibility
          while retaining positive violation in diagnostics.

    Returns:
        G_scaled: np.ndarray, effective constraints used by optimizers.
        diag_list: list of dicts with raw and scaled diagnostics.
    """
    scale_overrides = scale_overrides or {}
    kind_overrides = kind_overrides or {}

    all_names = list(thermo_names) + list(gear_names)
    all_raw = list(thermo_G) + list(gear_G)

    if realworld_G is not None and realworld_names is not None:
        all_names.extend(realworld_names)
        all_raw.extend(realworld_G)

    if len(all_names) != len(all_raw):
        raise ValueError(
            f"Constraint name/value length mismatch: {len(all_names)} names vs {len(all_raw)} values"
        )

    G_scaled = np.zeros(len(all_raw), dtype=np.float64)
    diag_list: list[dict] = []

    for i, (name, raw) in enumerate(zip(all_names, all_raw)):
        scale = float(scale_overrides.get(name, DEFAULT_SCALES.get(name, 1.0)))
        scale = scale if scale != 0 else 1.0
        kind = str(kind_overrides.get(name, DEFAULT_KIND.get(name, "hard")))
        scaled_raw = raw / scale
        scaled_effective = min(scaled_raw, 0.0) if kind == "soft" else scaled_raw
        soft_violation = max(scaled_raw, 0.0) if kind == "soft" else 0.0

        G_scaled[i] = scaled_effective
        diag_list.append(
            {
                "name": name,
                "raw": float(raw),
                "scale": scale,
                # Raw scaled value before hard/soft policy.
                "scaled_raw": float(scaled_raw),
                # Effective value after hard/soft policy (this is what goes to G).
                "scaled": float(scaled_effective),
                "soft_violation": float(soft_violation),
                "feasible": bool(scaled_effective <= 0.0),
                "kind": kind,
                "material_constraint": name in MATERIAL_CONSTRAINTS,
                "reason": DEFAULT_REASON.get(name, ""),
            }
        )

    return G_scaled, diag_list
