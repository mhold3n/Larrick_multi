"""Reduced-state handoff contract helpers for coupled combustion validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from .models import ValidationCaseSpec


@dataclass(frozen=True)
class ReducedStateHandoffBundle:
    """Minimal conserved state exchanged between chemistry and CFD legs."""

    bundle_id: str
    mechanism_id: str
    fuel_name: str
    pressure_Pa: float
    temperature_K: float
    species_mole_fractions: dict[str, float]
    vapor_fraction: float
    mixture_homogeneity_index: float
    velocity_m_s: float
    turbulence_intensity: float
    stage_marker: str
    cycle_coordinate_deg: float
    total_mass_kg: float
    total_energy_J: float

    def validate(self) -> list[str]:
        """Return structural/physics errors for the handoff bundle."""
        errors: list[str] = []
        if not self.bundle_id:
            errors.append("bundle_id is required")
        if not self.mechanism_id:
            errors.append("mechanism_id is required")
        if not self.fuel_name:
            errors.append("fuel_name is required")
        if self.pressure_Pa <= 0.0:
            errors.append("pressure_Pa must be positive")
        if self.temperature_K <= 0.0:
            errors.append("temperature_K must be positive")
        if not self.species_mole_fractions:
            errors.append("species_mole_fractions must not be empty")
        total_species = float(sum(float(v) for v in self.species_mole_fractions.values()))
        if total_species <= 0.0:
            errors.append("species_mole_fractions must sum to a positive value")
        if total_species > 1.05:
            errors.append("species_mole_fractions sum must not exceed 1.05")
        if not 0.0 <= self.vapor_fraction <= 1.0:
            errors.append("vapor_fraction must be within [0, 1]")
        if not 0.0 <= self.mixture_homogeneity_index <= 1.0:
            errors.append("mixture_homogeneity_index must be within [0, 1]")
        if self.velocity_m_s < 0.0:
            errors.append("velocity_m_s must be non-negative")
        if self.turbulence_intensity < 0.0:
            errors.append("turbulence_intensity must be non-negative")
        if self.total_mass_kg <= 0.0:
            errors.append("total_mass_kg must be positive")
        if self.total_energy_J <= 0.0:
            errors.append("total_energy_J must be positive")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bundle to a plain dict."""
        return asdict(self)


def _first_float(mapping: dict[str, Any], candidates: list[str], default: float) -> float:
    for key in candidates:
        if key in mapping and mapping[key] is not None:
            return float(mapping[key])
    return float(default)


def _pressure_from_case(case_spec: ValidationCaseSpec) -> float:
    operating_point = dict(case_spec.operating_point or {})
    if "pressure_Pa" in operating_point:
        return float(operating_point["pressure_Pa"])
    if "pressure_bar" in operating_point:
        return float(operating_point["pressure_bar"]) * 1.0e5
    if "ambient_pressure_bar" in operating_point:
        return float(operating_point["ambient_pressure_bar"]) * 1.0e5
    return 1.5e6


def _temperature_from_case(case_spec: ValidationCaseSpec) -> float:
    operating_point = dict(case_spec.operating_point or {})
    return _first_float(
        operating_point,
        ["temperature_K", "ambient_gas_temperature_K", "pilot_temperature_K"],
        900.0,
    )


def build_reduced_state_handoff(
    state_spec: dict[str, Any],
    *,
    chemistry_data: dict[str, Any] | None = None,
    spray_data: dict[str, Any] | None = None,
    reacting_data: dict[str, Any] | None = None,
    closed_cylinder_data: dict[str, Any] | None = None,
    case_spec: ValidationCaseSpec | None = None,
) -> ReducedStateHandoffBundle:
    """Construct a reduced-state handoff bundle from config and upstream metrics."""
    chemistry = dict(chemistry_data or {})
    spray = dict(spray_data or {})
    reacting = dict(reacting_data or {})
    closed_cylinder = dict(closed_cylinder_data or {})
    merged = dict(state_spec or {})

    mechanism_id = str(
        merged.get("mechanism_id")
        or dict(chemistry.get("mechanism_provenance", {}) or {}).get("mechanism_file")
        or dict(chemistry.get("mechanism_provenance", {}) or {}).get("mechanism_id")
        or "unknown_mechanism"
    )
    fuel_name = str(
        merged.get("fuel_name")
        or dict(chemistry.get("mechanism_provenance", {}) or {}).get("fuel_name")
        or "gasoline"
    )
    pressure_Pa = float(
        merged.get("pressure_Pa")
        or (
            float(closed_cylinder.get("peak_pressure_bar", 0.0)) * 1.0e5
            if "peak_pressure_bar" in closed_cylinder
            else 0.0
        )
        or _pressure_from_case(
            case_spec or ValidationCaseSpec(case_id="handoff", regime="full_handoff")
        )
    )
    temperature_K = float(
        merged.get("temperature_K")
        or reacting.get("pilot_temperature_K_flameD_xd1")
        or _temperature_from_case(
            case_spec or ValidationCaseSpec(case_id="handoff", regime="full_handoff")
        )
    )
    species = dict(merged.get("species_mole_fractions", {}) or {})
    if not species:
        species = {
            "IC8H18": float(merged.get("fuel_mole_fraction", 0.02)),
            "O2": float(merged.get("oxidizer_mole_fraction", 0.21)),
            "N2": float(merged.get("diluent_mole_fraction", 0.77)),
        }
    velocity_m_s = float(
        merged.get("velocity_m_s")
        or reacting.get("jet_bulk_velocity_m_s_flameD")
        or spray.get("gas_axial_velocity_m_s_sprayG_z15mm_t1ms")
        or 15.0
    )
    vapor_fraction = float(
        merged.get("vapor_fraction")
        or min(
            1.0,
            max(
                0.0,
                float(spray.get("vapor_spreading_angle_deg_sprayG", 0.0) / 120.0),
            ),
        )
    )
    homogeneity = float(
        merged.get("mixture_homogeneity_index") or min(1.0, max(0.0, vapor_fraction * 0.9 + 0.05))
    )
    turbulence_intensity = float(merged.get("turbulence_intensity", 0.12))
    cycle_coordinate_deg = float(merged.get("cycle_coordinate_deg", 365.0))
    total_mass_kg = float(merged.get("total_mass_kg", 4.0e-4))
    total_energy_J = float(
        merged.get("total_energy_J")
        or (
            float(closed_cylinder.get("heat_release_kJ", 0.0)) * 1000.0
            if "heat_release_kJ" in closed_cylinder
            else 0.0
        )
        or total_mass_kg * temperature_K * 900.0
    )
    stage_marker = str(merged.get("stage_marker", "reacting_flow_entry"))
    bundle_id = str(merged.get("bundle_id", f"{stage_marker}_{fuel_name}"))

    return ReducedStateHandoffBundle(
        bundle_id=bundle_id,
        mechanism_id=mechanism_id,
        fuel_name=fuel_name,
        pressure_Pa=pressure_Pa,
        temperature_K=temperature_K,
        species_mole_fractions=species,
        vapor_fraction=vapor_fraction,
        mixture_homogeneity_index=homogeneity,
        velocity_m_s=velocity_m_s,
        turbulence_intensity=turbulence_intensity,
        stage_marker=stage_marker,
        cycle_coordinate_deg=cycle_coordinate_deg,
        total_mass_kg=total_mass_kg,
        total_energy_J=total_energy_J,
    )


def build_handoff_state_chain(
    *,
    base_state: dict[str, Any],
    state_overrides: list[dict[str, Any]] | None = None,
    chemistry_data: dict[str, Any] | None = None,
    spray_data: dict[str, Any] | None = None,
    reacting_data: dict[str, Any] | None = None,
    closed_cylinder_data: dict[str, Any] | None = None,
    case_spec: ValidationCaseSpec | None = None,
) -> list[ReducedStateHandoffBundle]:
    """Build an ordered state chain for conservation checks."""
    overrides = list(state_overrides or [])
    if not overrides:
        overrides = [
            {"stage_marker": "chemistry_exit"},
            {"stage_marker": "spray_exit"},
            {"stage_marker": "reacting_flow_entry"},
        ]

    base_bundle = build_reduced_state_handoff(
        base_state,
        chemistry_data=chemistry_data,
        spray_data=spray_data,
        reacting_data=reacting_data,
        closed_cylinder_data=closed_cylinder_data,
        case_spec=case_spec,
    )
    bundles: list[ReducedStateHandoffBundle] = []
    for idx, override in enumerate(overrides):
        merged = dict(base_state or {})
        merged.update(dict(override or {}))
        merged.setdefault("bundle_id", f"{base_bundle.bundle_id}_{idx}")
        bundle = build_reduced_state_handoff(
            merged,
            chemistry_data=chemistry_data,
            spray_data=spray_data,
            reacting_data=reacting_data,
            closed_cylinder_data=closed_cylinder_data,
            case_spec=case_spec,
        )
        # Preserve the base state unless the override intentionally perturbs it.
        if idx == 0:
            bundles.append(bundle)
            continue
        bundles.append(
            replace(
                bundle,
                total_mass_kg=float(merged.get("total_mass_kg", base_bundle.total_mass_kg)),
                total_energy_J=float(merged.get("total_energy_J", base_bundle.total_energy_J)),
                pressure_Pa=float(merged.get("pressure_Pa", base_bundle.pressure_Pa)),
                temperature_K=float(merged.get("temperature_K", base_bundle.temperature_K)),
            )
        )
    return bundles


def compute_handoff_conservation(
    bundles: list[ReducedStateHandoffBundle],
    *,
    mass_tolerance: float,
    energy_tolerance: float,
) -> dict[str, Any]:
    """Compute per-transition and overall conservation errors."""
    if len(bundles) < 2:
        return {
            "state_conservation_mass": 0.0,
            "state_conservation_energy": 0.0,
            "handoff_states": [],
        }

    max_mass_error = 0.0
    max_energy_error = 0.0
    handoff_states: list[dict[str, Any]] = []
    for prev, curr in zip(bundles[:-1], bundles[1:]):
        mass_error = abs(curr.total_mass_kg - prev.total_mass_kg)
        energy_error = abs(curr.total_energy_J - prev.total_energy_J)
        max_mass_error = max(max_mass_error, mass_error)
        max_energy_error = max(max_energy_error, energy_error)
        handoff_states.append(
            {
                "from_phase": prev.stage_marker,
                "to_phase": curr.stage_marker,
                "conservation_error": max(mass_error, energy_error),
                "conservation_tolerance": max(mass_tolerance, energy_tolerance),
                "mass_error": mass_error,
                "energy_error": energy_error,
                "bundle_in": prev.to_dict(),
                "bundle_out": curr.to_dict(),
            }
        )

    return {
        "state_conservation_mass": max_mass_error,
        "state_conservation_energy": max_energy_error,
        "handoff_states": handoff_states,
    }
