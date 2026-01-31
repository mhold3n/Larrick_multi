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
    "gear_ratio_error_max": 1.0,
    "gear_min_radius": 10.0,  # mm
    "gear_max_radius": 10.0,  # mm
    "gear_max_curvature": 0.5,  # 1/mm
    "gear_interference": 1.0,
    "gear_min_thickness": 5.0,  # placeholder scaling
}


THERMO_CONSTRAINTS_FID0 = [
    "thermo_compression_min",
    "thermo_heat_release_width_min",
    "thermo_ratio_jerk_max",
    "thermo_ratio_slope_max",
]

THERMO_CONSTRAINTS_FID1 = [
    "thermo_eff_min",
    "thermo_eff_max",
    "thermo_pmax_norm",
    "thermo_ratio_slope_max",
]

GEAR_CONSTRAINTS = [
    "gear_ratio_error_max",
    "gear_min_radius",
    "gear_max_radius",
    "gear_max_curvature",
    "gear_interference",
    "gear_min_thickness",
]


def get_constraint_names(fidelity: int) -> list[str]:
    """Return ordered constraint names for a given fidelity."""
    thermo = THERMO_CONSTRAINTS_FID1 if fidelity >= 1 else THERMO_CONSTRAINTS_FID0
    return list(thermo) + list(GEAR_CONSTRAINTS)


def get_constraint_scales() -> dict[str, float]:
    """Expose default constraint scales for downstream metadata."""
    return DEFAULT_SCALES.copy()


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
) -> tuple[np.ndarray, list[dict]]:
    """Merge thermo + gear constraints with scaling and reason codes.

    Returns:
        G_scaled: np.ndarray, concatenated constraints (scaled) with sign convention G<=0 feasible.
        diag_list: list of dicts with raw/scaled/name.
    """
    scale_overrides = scale_overrides or {}

    all_names = list(thermo_names) + list(gear_names)
    all_raw = list(thermo_G) + list(gear_G)

    if len(all_names) != len(all_raw):
        raise ValueError(
            f"Constraint name/value length mismatch: {len(all_names)} names vs {len(all_raw)} values"
        )

    G_scaled = np.zeros(len(all_raw), dtype=np.float64)
    diag_list: list[dict] = []

    for i, (name, raw) in enumerate(zip(all_names, all_raw)):
        scale = float(scale_overrides.get(name, DEFAULT_SCALES.get(name, 1.0)))
        scale = scale if scale != 0 else 1.0
        scaled = raw / scale
        G_scaled[i] = scaled
        diag_list.append(
            {
                "name": name,
                "raw": float(raw),
                "scale": scale,
                "scaled": float(scaled),
                "feasible": bool(scaled <= 0.0),
            }
        )

    return G_scaled, diag_list
