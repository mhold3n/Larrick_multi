"""OpenFOAM Execution Pipeline.

This module consolidates the orchestration logic for running OpenFOAM cases,
including template cloning, parameter substitution, geometry generation,
and the complex meshing/solving sequence (overset, sliding mesh, etc.).
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from larrak2.adapters.docker_openfoam import DockerOpenFoam, DockerOpenFoamConfig
from larrak2.adapters.openfoam import OpenFoamRunner
from larrak2.simulation_validation.engine_results import emit_engine_results_artifact

DEFAULT_ENGINE_TEMPLATE_DIR = Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case")
DEFAULT_ENGINE_SOLVER_SOURCE_DIR = Path("openfoam_custom_solvers/larrakEngineFoam")
DEFAULT_ENGINE_PACKAGE_DIR = Path("mechanisms/openfoam/v2512/chem323_reduced")
DEFAULT_ENGINE_RUNTIME_CHEMISTRY_TABLE_DIR = Path(
    "mechanisms/openfoam/v2512/runtime_tables/chem323_engine_ignition_v2"
)
TRACKED_ENGINE_SPECIES = ("IC8H18", "O2", "N2", "CO2", "H2O")
MOLECULAR_WEIGHTS = {
    "IC8H18": 114.23,
    "O2": 31.998,
    "N2": 28.0134,
    "CO2": 44.01,
    "H2O": 18.01528,
}

DEFAULT_ENGINE_PROOF_MODE = "full_cycle_breathing"
REACTING_ENGINE_CALIBRATION_MODE = "reacting_calibration_window"
STAGED_REACTING_ENGINE_CALIBRATION_MODE = "reacting_staged_ignition"


def _openfoam_word_token(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    if not text:
        return default
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", text)
    if not normalized:
        return default
    if not re.match(r"^[A-Za-z_]", normalized):
        normalized = f"w_{normalized}"
    return normalized


def _package_manifest(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "package_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"OpenFOAM chemistry package manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {manifest_path}")
    return payload


def _package_staged_inputs(package_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "source": str(package_dir / "reactions"),
            "target": "constant/reactions",
        },
        {
            "source": str(package_dir / "thermo.compressibleGas"),
            "target": "constant/thermo.compressibleGas",
        },
        {
            "source": str(package_dir / "transportProperties"),
            "target": "constant/transportProperties",
        },
        {
            "source": str(package_dir / "package_manifest.json"),
            "target": "chemistry/package_manifest.json",
        },
    ]


def _runtime_chemistry_table_manifest(table_dir: Path) -> dict[str, Any]:
    manifest_path = table_dir / "runtime_chemistry_table_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Runtime chemistry table manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {manifest_path}")
    return payload


def _runtime_chemistry_table_staged_inputs(table_dir: Path) -> list[dict[str, str]]:
    staged = [
        {
            "source": str(table_dir / "runtimeChemistryTable"),
            "target": "constant/runtimeChemistryTable",
        },
        {
            "source": str(table_dir / "runtime_chemistry_table_manifest.json"),
            "target": "chemistry/runtime_chemistry_table_manifest.json",
        },
    ]
    for source_name, target_name in (
        ("runtime_chemistry_table_data.npz", "chemistry/runtime_chemistry_table_data.npz"),
        ("runtime_chemistry_jacobian_csr.npz", "chemistry/runtime_chemistry_jacobian_csr.npz"),
        ("runtime_chemistry_jacobian.json", "chemistry/runtime_chemistry_jacobian.json"),
    ):
        source_path = table_dir / source_name
        if not source_path.exists():
            continue
        staged.append(
            {
                "source": str(source_path),
                "target": target_name,
            }
        )
    return staged


def _resolve_engine_package_dir(
    params: dict[str, Any],
    default_package_dir: Path,
) -> Path:
    override = str(params.get("openfoam_chemistry_package_dir", "") or "").strip()
    return Path(override) if override else default_package_dir


def _resolve_engine_runtime_chemistry_table_dir(
    params: dict[str, Any],
    default_table_dir: Path,
) -> Path | None:
    override = str(params.get("openfoam_runtime_chemistry_table_dir", "") or "").strip()
    candidate = Path(override) if override else default_table_dir
    return candidate if (candidate / "runtime_chemistry_table_manifest.json").exists() else None


def _engine_stage_profile_completion_defaults(profile_id: str) -> tuple[float, float]:
    normalized = str(profile_id).strip().lower()
    if normalized == "closed_valve_ignition_fast_runtime_v1":
        return 0.02, 5.0e-8
    if normalized == "closed_valve_ignition_low_clamp_v1":
        return 0.005, 1.5e-8
    return 0.01, 2.5e-8


def _default_species_mole_fractions(lambda_af: float, residual_fraction: float) -> dict[str, float]:
    lam = max(float(lambda_af), 0.3)
    residual = max(0.0, min(0.4, float(residual_fraction)))
    air_moles = 59.5238095238 * lam
    fresh_fuel = 1.0
    fresh_o2 = 0.21 * air_moles
    fresh_n2 = 0.79 * air_moles
    residual_co2 = residual * 0.08 * max(air_moles, 1.0)
    residual_h2o = residual * 0.12 * max(air_moles, 1.0)
    total = fresh_fuel + fresh_o2 + fresh_n2 + residual_co2 + residual_h2o
    return {
        "IC8H18": fresh_fuel / total,
        "O2": fresh_o2 / total,
        "N2": fresh_n2 / total,
        "CO2": residual_co2 / total,
        "H2O": residual_h2o / total,
    }


def _to_mass_fractions(mole_fractions: dict[str, float]) -> dict[str, float]:
    weighted = {
        species: max(float(value), 0.0) * float(MOLECULAR_WEIGHTS.get(species, 28.0))
        for species, value in mole_fractions.items()
    }
    total = sum(weighted.values())
    if total <= 0.0:
        return {}
    return {species: value / total for species, value in weighted.items()}


def _engine_cycle_duration_s(
    *,
    rpm: float,
    start_angle_deg: float,
    end_angle_deg: float,
) -> float:
    safe_rpm = max(abs(float(rpm)), 1.0)
    angle_span_deg = abs(float(end_angle_deg) - float(start_angle_deg))
    if angle_span_deg <= 0.0:
        angle_span_deg = 360.0
    return angle_span_deg / 360.0 * 60.0 / safe_rpm


def _end_time_for_target_angle(
    *,
    rpm: float,
    initial_angle_deg: float,
    target_angle_deg: float,
) -> float:
    delta_deg = float(target_angle_deg) - float(initial_angle_deg)
    if delta_deg <= 0.0:
        delta_deg += 360.0
    return delta_deg / 360.0 * 60.0 / max(abs(float(rpm)), 1.0)


def _apply_engine_runtime_profile(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    mode = str(normalized.get("engine_proof_mode", "") or "").strip().lower()
    if not mode:
        return normalized
    if mode not in {
        DEFAULT_ENGINE_PROOF_MODE,
        REACTING_ENGINE_CALIBRATION_MODE,
        STAGED_REACTING_ENGINE_CALIBRATION_MODE,
    }:
        raise ValueError(f"Unsupported engine proof mode: {mode}")

    if mode == REACTING_ENGINE_CALIBRATION_MODE:
        normalized.setdefault("engine_start_angle_deg", -40.0)
        normalized.setdefault("engine_end_angle_deg", 20.0)
    elif mode == STAGED_REACTING_ENGINE_CALIBRATION_MODE:
        normalized.setdefault("engine_start_angle_deg", -10.0)
        normalized.setdefault("engine_end_angle_deg", 12.0)

    start_angle_deg = float(normalized.get("engine_start_angle_deg", -180.0))
    end_angle_deg = float(normalized.get("engine_end_angle_deg", 180.0))
    rpm = float(normalized.get("rpm", 1800.0))
    derived_end_time = _engine_cycle_duration_s(
        rpm=rpm,
        start_angle_deg=start_angle_deg,
        end_angle_deg=end_angle_deg,
    )

    if mode == DEFAULT_ENGINE_PROOF_MODE:
        proof_defaults: dict[str, Any] = {
            "chemistry_enabled": False,
            "combustion_enabled": False,
            "engine_min_temperature_K": 300.0,
            "engine_max_temperature_K": 1350.0,
            "engine_energy_mode": "sensibleInternalEnergy",
            "mesh_nx": 10,
            "mesh_ny": 10,
            "mesh_nz": 14,
            "surface_level": 0,
            "interface_level": 0,
            "maxCo": 2.0,
            "writeInterval": 50,
            "metricWriteInterval": 50,
        }
    else:
        proof_defaults = {
            "chemistry_enabled": True,
            "combustion_enabled": True,
            "engine_min_temperature_K": 300.0,
            "engine_max_temperature_K": 1350.0,
            "engine_max_thermo_delta_K": 35.0,
            "engine_min_pressure_Pa": 5000.0,
            "engine_min_density_kg_m3": 0.02,
            "engine_energy_mode": "sensibleInternalEnergy",
            "mesh_nx": 10,
            "mesh_ny": 10,
            "mesh_nz": 14,
            "surface_level": 0,
            "interface_level": 0,
            "maxCo": 0.5,
            "writeInterval": 10,
            "metricWriteInterval": 10,
            "chemistry_initial_timestep_s": 1.0e-7,
            "chemistry_abs_tol": 1.0e-12,
            "chemistry_rel_tol": 1.0e-7,
            "chemistry_reduction_enabled": True,
            "chemistry_tabulation_enabled": True,
            "combustion_cmix": 0.03,
        }
    for key, value in proof_defaults.items():
        normalized.setdefault(key, value)

    if "deltaT" not in normalized:
        normalized["deltaT"] = (
            5.0e-6
            if mode == STAGED_REACTING_ENGINE_CALIBRATION_MODE
            else 2.0e-5
            if mode == REACTING_ENGINE_CALIBRATION_MODE
            else 1.0e-4
        )
    if mode == STAGED_REACTING_ENGINE_CALIBRATION_MODE:
        normalized.setdefault("engine_stage_profile", "closed_valve_ignition_v1")
        normalized.setdefault("maxCo", 0.2)
        normalized.setdefault("maxDeltaT", min(float(normalized.get("deltaT", 5.0e-6)), 5.0e-6))
        normalized.setdefault("engine_max_temperature_K", 1700.0)
        angle_tol_deg, time_tol_s = _engine_stage_profile_completion_defaults(
            str(normalized.get("engine_stage_profile", "closed_valve_ignition_v1"))
        )
        normalized.setdefault("engine_stage_completion_angle_tolerance_deg", angle_tol_deg)
        normalized.setdefault("engine_stage_completion_time_tolerance_s", time_tol_s)
    normalized.setdefault("maxDeltaT", float(normalized.get("deltaT", 1.0e-4)))
    if bool(normalized.get("derive_end_time_from_angles", True)):
        normalized["endTime"] = derived_end_time
    else:
        normalized.setdefault("endTime", derived_end_time)
    return normalized


def _engine_seed_bundle(
    params: dict[str, Any],
    *,
    handoff_bundle: dict[str, Any] | None,
    package_manifest: dict[str, Any],
) -> dict[str, Any]:
    incoming = dict(handoff_bundle or {})
    residual_fraction = float(
        incoming.get("residual_fraction", params.get("residual_fraction_seed", 0.08))
    )
    species_mole_fractions = dict(incoming.get("species_mole_fractions", {}) or {})
    if not species_mole_fractions:
        species_mole_fractions = _default_species_mole_fractions(
            float(params.get("lambda_af", 1.0)),
            residual_fraction,
        )
    species_mass_fractions = _to_mass_fractions(species_mole_fractions)
    pressure = float(incoming.get("pressure_Pa", params.get("p_manifold_Pa", 101325.0)))
    temperature = float(incoming.get("temperature_K", params.get("T_intake_K", 300.0)))
    cycle_coordinate = float(
        incoming.get("cycle_coordinate_deg", params.get("engine_start_angle_deg", -180.0))
    )
    mixture_homogeneity = float(
        incoming.get("mixture_homogeneity_index", params.get("mixture_homogeneity_index", 0.92))
    )
    velocity = float(incoming.get("velocity_m_s", params.get("handoff_velocity_m_s", 0.0)))
    return {
        "bundle_id": str(
            incoming.get(
                "bundle_id",
                f"engine_seed_rpm{int(round(float(params.get('rpm', 0.0))))}_tq{int(round(float(params.get('torque', 0.0))))}",
            )
        ),
        "mechanism_id": str(
            incoming.get(
                "mechanism_id", package_manifest.get("package_id", "chem323_reduced_v2512")
            )
        ),
        "fuel_name": str(incoming.get("fuel_name", "iso-octane")),
        "pressure_Pa": pressure,
        "temperature_K": temperature,
        "species_mole_fractions": species_mole_fractions,
        "species_mass_fractions": species_mass_fractions,
        "vapor_fraction": float(incoming.get("vapor_fraction", 1.0)),
        "mixture_homogeneity_index": mixture_homogeneity,
        "velocity_m_s": velocity,
        "turbulence_intensity": float(incoming.get("turbulence_intensity", 0.05)),
        "stage_marker": str(incoming.get("stage_marker", "engine_intake_start")),
        "cycle_coordinate_deg": cycle_coordinate,
        "total_mass_kg": float(
            incoming.get("total_mass_kg", params.get("trapped_mass_seed", 4.0e-4))
        ),
        "total_energy_J": float(
            incoming.get(
                "total_energy_J",
                params.get("handoff_total_energy_J", 4.0e-4 * 1000.0 * temperature),
            )
        ),
        "residual_fraction": residual_fraction,
    }


def _engine_case_placeholders(
    params: dict[str, Any],
    handoff_bundle: dict[str, Any],
) -> dict[str, Any]:
    mass_fractions = dict(handoff_bundle.get("species_mass_fractions", {}) or {})
    cycle_coordinate = float(handoff_bundle.get("cycle_coordinate_deg", -180.0))
    pressure = float(handoff_bundle.get("pressure_Pa", params.get("p_manifold_Pa", 101325.0)))
    temperature = float(handoff_bundle.get("temperature_K", params.get("T_intake_K", 300.0)))
    residual_fraction = float(
        handoff_bundle.get("residual_fraction", params.get("residual_fraction_seed", 0.08))
    )
    residual_temperature = float(
        params.get(
            "T_residual_K", max(float(params.get("T_intake_K", 300.0)) * 2.5, temperature * 1.3)
        )
    )
    wall_temperature = float(
        params.get(
            "engine_wall_temperature_K",
            max(float(params.get("T_intake_K", temperature)) * 1.2, residual_temperature * 0.5),
        )
    )
    bore_m = float(params.get("bore_mm", 80.0)) / 1000.0
    stroke_m = float(params.get("stroke_mm", 90.0)) / 1000.0
    cylinder_length = stroke_m * 1.5
    valve_length = bore_m * 0.5
    intake_valve_z = float(params.get("intake_valve_z", cylinder_length / 2.0))
    exhaust_left_z = float(params.get("exhaust_left_z", valve_length / 2.0))
    exhaust_right_z = float(params.get("exhaust_right_z", cylinder_length - valve_length / 2.0))
    omega_rad_s = float(
        params.get("omega_rad_s", float(params.get("rpm", 1500.0)) * 2.0 * math.pi / 60.0)
    )
    intake_valve_omega = float(params.get("intake_valve_omega_rad_s", omega_rad_s))
    exhaust_left_omega = float(params.get("exhaust_left_omega_rad_s", -omega_rad_s))
    exhaust_right_omega = float(params.get("exhaust_right_omega_rad_s", -omega_rad_s))
    mesh_x_half_extent = float(params.get("mesh_x_half_extent", max(bore_m * 1.5, 0.12)))
    mesh_y_half_extent = float(params.get("mesh_y_half_extent", max(bore_m * 2.0, 0.16)))
    mesh_z_max = float(params.get("mesh_z_max", cylinder_length + max(bore_m * 0.5625, 0.045)))

    def _point(x: float, y: float, z: float) -> str:
        return f"({x:.9g} {y:.9g} {z:.9g})"

    enriched = dict(params)
    enriched.update(
        {
            "solver_name": str(params.get("solver_name", "larrakEngineFoam")),
            "engine_rpm": float(params.get("rpm", 1500.0)),
            "engine_initialCrankAngleDeg": cycle_coordinate,
            "engine_end_angle_deg": float(params.get("engine_end_angle_deg", 180.0)),
            "engine_min_temperature_K": float(params.get("engine_min_temperature_K", 300.0)),
            "engine_max_temperature_K": float(params.get("engine_max_temperature_K", 3500.0)),
            "engine_max_thermo_delta_K": float(params.get("engine_max_thermo_delta_K", 250.0)),
            "engine_min_pressure_Pa": float(params.get("engine_min_pressure_Pa", 100.0)),
            "engine_min_density_kg_m3": float(params.get("engine_min_density_kg_m3", 1.0e-4)),
            "engine_energy_mode": str(params.get("engine_energy_mode", "sensibleInternalEnergy")),
            "runtime_chemistry_mode": str(
                params.get("runtime_chemistry_mode", "fullReducedKinetics")
            ),
            "runtime_chemistry_strict": "true"
            if bool(params.get("runtime_chemistry_strict", False))
            else "false",
            "runtime_chemistry_abort_on_authority_miss": "true"
            if bool(params.get("runtime_chemistry_abort_on_authority_miss", False))
            else "false",
            "runtime_chemistry_fallback_policy": str(
                params.get("runtime_chemistry_fallback_policy", "fullReducedKinetics")
            ),
            "runtime_chemistry_interpolation": str(
                params.get("runtime_chemistry_interpolation", "local_rbf")
            ),
            "runtime_chemistry_max_untracked_mass_fraction": float(
                params.get("runtime_chemistry_max_untracked_mass_fraction", 0.02)
            ),
            "runtime_chemistry_table_hash": str(
                _openfoam_word_token(
                    params.get("runtime_chemistry_table_hash", "none"),
                    default="none",
                )
            ),
            "runtime_chemistry_stage_name": str(
                _openfoam_word_token(
                    params.get("runtime_chemistry_stage_name", "unset"),
                    default="unset",
                )
            ),
            "engine_wall_temperature_K": wall_temperature,
            "chemistry_switch": "on" if bool(params.get("chemistry_enabled", True)) else "off",
            "combustion_switch": "yes" if bool(params.get("combustion_enabled", True)) else "no",
            "chemistry_initial_timestep_s": float(
                params.get("chemistry_initial_timestep_s", 1.0e-7)
            ),
            "chemistry_abs_tol": float(params.get("chemistry_abs_tol", 1.0e-12)),
            "chemistry_rel_tol": float(params.get("chemistry_rel_tol", 1.0e-7)),
            "chemistry_reduction_switch": "on"
            if bool(params.get("chemistry_reduction_enabled", True))
            else "off",
            "chemistry_tabulation_switch": "on"
            if bool(params.get("chemistry_tabulation_enabled", True))
            else "off",
            "combustion_cmix": float(params.get("combustion_cmix", 0.03)),
            "mesh_nx": int(params.get("mesh_nx", 20)),
            "mesh_ny": int(params.get("mesh_ny", 20)),
            "mesh_nz": int(params.get("mesh_nz", 30)),
            "surface_level": int(params.get("surface_level", 1)),
            "interface_level": int(params.get("interface_level", 1)),
            "maxCo": float(params.get("maxCo", 1.0)),
            "maxDeltaT": float(params.get("maxDeltaT", params.get("deltaT", 1.0e-4))),
            "T_intake_K": temperature,
            "T_residual_K": residual_temperature,
            "handoff_pressure_Pa": pressure,
            "handoff_temperature_K": temperature,
            "handoff_residual_pressure_Pa": float(params.get("p_back_Pa", pressure)),
            "handoff_residual_temperature_K": residual_temperature,
            "handoff_velocity_m_s": float(handoff_bundle.get("velocity_m_s", 0.0)),
            "handoff_residual_tracer": max(0.0, min(1.0, residual_fraction)),
            "handoff_species_IC8H18": mass_fractions.get("IC8H18", 0.0),
            "handoff_species_O2": mass_fractions.get("O2", 0.0),
            "handoff_species_N2": mass_fractions.get("N2", 0.0),
            "handoff_species_CO2": mass_fractions.get("CO2", 0.0),
            "handoff_species_H2O": mass_fractions.get("H2O", 0.0),
            "handoff_species_Ydefault": max(
                0.0,
                1.0 - sum(mass_fractions.get(species, 0.0) for species in TRACKED_ENGINE_SPECIES),
            ),
            "injection_pressure_bar": float(params.get("injection_pressure_bar", pressure / 1.0e5)),
            "ambient_pressure_bar": float(params.get("ambient_pressure_bar", pressure / 1.0e5)),
            "ambient_gas_temperature_K": temperature,
            "intake_valve_z": intake_valve_z,
            "exhaust_left_z": exhaust_left_z,
            "exhaust_right_z": exhaust_right_z,
            "intake_valve_origin": str(
                params.get("intake_valve_origin", _point(0.0, 0.0, intake_valve_z))
            ),
            "exhaust_left_origin": str(
                params.get("exhaust_left_origin", _point(0.0, 0.0, exhaust_left_z))
            ),
            "exhaust_right_origin": str(
                params.get("exhaust_right_origin", _point(0.0, 0.0, exhaust_right_z))
            ),
            "intake_interface_point": str(
                params.get("intake_interface_point", _point(0.0, 0.0, intake_valve_z))
            ),
            "exhaust_left_interface_point": str(
                params.get("exhaust_left_interface_point", _point(0.0, 0.0, exhaust_left_z))
            ),
            "exhaust_right_interface_point": str(
                params.get("exhaust_right_interface_point", _point(0.0, 0.0, exhaust_right_z))
            ),
            "location_in_mesh": str(
                params.get("location_in_mesh", _point(0.0, 0.0, cylinder_length * 0.5))
            ),
            "intake_valve_omega_rad_s": intake_valve_omega,
            "exhaust_left_omega_rad_s": exhaust_left_omega,
            "exhaust_right_omega_rad_s": exhaust_right_omega,
            "mesh_x_min": -mesh_x_half_extent,
            "mesh_x_max": mesh_x_half_extent,
            "mesh_y_min": -mesh_y_half_extent,
            "mesh_y_max": mesh_y_half_extent,
            "mesh_z_max": mesh_z_max,
        }
    )
    return enriched


class OpenFoamPipeline:
    """Orchestrates the end-to-end execution of an OpenFOAM case."""

    def __init__(
        self,
        template_dir: str | Path,
        solver_cmd: str = "pisoFoam",
        docker_timeout_s: int = 1800,
        docker_image: str | None = None,
        custom_solver_source_dir: str | Path | None = None,
        custom_solver_cache_root: str | Path | None = None,
        chemistry_package_dir: str | Path | None = None,
        runtime_chemistry_table_dir: str | Path | None = None,
    ):
        self.template_dir = Path(template_dir)
        self.solver_cmd = solver_cmd
        self.docker_timeout_s = docker_timeout_s
        self.custom_solver_source_dir = (
            Path(custom_solver_source_dir)
            if custom_solver_source_dir is not None
            else DEFAULT_ENGINE_SOLVER_SOURCE_DIR
        )
        self.custom_solver_cache_root = (
            Path(custom_solver_cache_root)
            if custom_solver_cache_root is not None
            else Path(DockerOpenFoamConfig().custom_solver_cache_root)
        )
        self.chemistry_package_dir = (
            Path(chemistry_package_dir)
            if chemistry_package_dir is not None
            else DEFAULT_ENGINE_PACKAGE_DIR
        )
        self.runtime_chemistry_table_dir = (
            Path(runtime_chemistry_table_dir)
            if runtime_chemistry_table_dir is not None
            else DEFAULT_ENGINE_RUNTIME_CHEMISTRY_TABLE_DIR
        )
        self.docker = (
            DockerOpenFoam(
                DockerOpenFoamConfig(
                    image=docker_image or DockerOpenFoamConfig().image,
                    custom_solver_cache_root=str(self.custom_solver_cache_root),
                )
            )
            if docker_image is not None
            else DockerOpenFoam(
                DockerOpenFoamConfig(custom_solver_cache_root=str(self.custom_solver_cache_root))
            )
        )
        self.runner = OpenFoamRunner(
            template_dir=template_dir,
            solver_cmd=solver_cmd,
            backend="docker",
            docker_image=docker_image,
        )

    def _is_default_engine_template(self) -> bool:
        return self.template_dir.name == DEFAULT_ENGINE_TEMPLATE_DIR.name

    def _engine_case_assets(
        self,
        params: dict[str, Any],
        *,
        handoff_bundle: dict[str, Any] | None = None,
        staged_inputs: list[dict[str, str]] | None = None,
    ) -> tuple[
        dict[str, Any], list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, Any]
    ]:
        if not self._is_default_engine_template():
            return dict(params), list(staged_inputs or []), dict(handoff_bundle or {}), {}, {}
        normalized_params = _apply_engine_runtime_profile(params)
        package_dir = _resolve_engine_package_dir(normalized_params, self.chemistry_package_dir)
        package_manifest = _package_manifest(package_dir)
        runtime_table_manifest: dict[str, Any] = {}
        runtime_table_dir = _resolve_engine_runtime_chemistry_table_dir(
            normalized_params,
            self.runtime_chemistry_table_dir,
        )
        merged_stage_inputs = list(staged_inputs or []) + _package_staged_inputs(package_dir)
        if runtime_table_dir is not None:
            runtime_table_manifest = _runtime_chemistry_table_manifest(runtime_table_dir)
            normalized_params.setdefault("runtime_chemistry_mode", "lookupTableStrict")
            runtime_mode = str(normalized_params.get("runtime_chemistry_mode", "lookupTableStrict"))
            normalized_params.setdefault(
                "runtime_chemistry_strict",
                runtime_mode == "lookupTableStrict",
            )
            normalized_params.setdefault(
                "runtime_chemistry_abort_on_authority_miss",
                bool(normalized_params.get("runtime_chemistry_strict", False)),
            )
            normalized_params.setdefault(
                "runtime_chemistry_fallback_policy",
                str(runtime_table_manifest.get("fallback_policy", "fullReducedKinetics")),
            )
            normalized_params.setdefault(
                "runtime_chemistry_interpolation",
                str(runtime_table_manifest.get("interpolation_method", "local_rbf")),
            )
            normalized_params.setdefault(
                "runtime_chemistry_max_untracked_mass_fraction",
                float(runtime_table_manifest.get("max_untracked_mass_fraction", 0.02)),
            )
            normalized_params["openfoam_runtime_chemistry_table_dir"] = str(runtime_table_dir)
            normalized_params["runtime_chemistry_table_hash"] = str(
                runtime_table_manifest.get("generated_file_hashes", {}).get(
                    "runtimeChemistryTable", ""
                )
            )
            merged_stage_inputs += _runtime_chemistry_table_staged_inputs(runtime_table_dir)
        else:
            normalized_params.setdefault("runtime_chemistry_mode", "fullReducedKinetics")
            normalized_params.setdefault("runtime_chemistry_strict", False)
            normalized_params.setdefault("runtime_chemistry_abort_on_authority_miss", False)
            normalized_params.setdefault("runtime_chemistry_fallback_policy", "fullReducedKinetics")
            normalized_params.setdefault("runtime_chemistry_interpolation", "local_rbf")
            normalized_params.setdefault("runtime_chemistry_max_untracked_mass_fraction", 0.02)
        engine_handoff = _engine_seed_bundle(
            normalized_params,
            handoff_bundle=handoff_bundle,
            package_manifest=package_manifest,
        )
        merged_params = _engine_case_placeholders(normalized_params, engine_handoff)
        merged_params["openfoam_chemistry_package_dir"] = str(package_dir)
        return (
            merged_params,
            merged_stage_inputs,
            engine_handoff,
            package_manifest,
            runtime_table_manifest,
        )

    def _ensure_custom_solver(self, *, log_file: Path | None = None) -> dict[str, str]:
        if self.solver_cmd != "larrakEngineFoam":
            return {}
        return self.docker.ensure_custom_solver(
            source_dir=self.custom_solver_source_dir,
            solver_name=self.solver_cmd,
            cache_root=self.custom_solver_cache_root,
            log_file=log_file,
        )

    def setup_case(
        self,
        run_dir: Path,
        params: dict[str, Any],
        *,
        staged_inputs: list[dict[str, str]] | None = None,
    ) -> None:
        """Clone template and substitute parameters."""
        self.runner.setup_case_with_assets(run_dir, params, staged_inputs=staged_inputs)

    def generate_geometry(
        self,
        run_dir: Path,
        bore_mm: float,
        stroke_mm: float,
        intake_port_area_m2: float,
        exhaust_port_area_m2: float,
        *,
        geometry_params: dict[str, Any] | None = None,
    ) -> None:
        """Generate STL geometry for the case."""
        tri_dir = run_dir / "constant" / "triSurface"
        tri_dir.mkdir(parents=True, exist_ok=True)

        from larrak2.geometry.generate_stl import generate_stl_workflow

        # Logic adapted from generate_stl.py main() to avoid shelling out
        # We can also just call the workflow function if we restructure it to accept args object or kwargs
        # But let's reuse the logic since we have access to it.
        # Actually, simpler: construct a dummy args object and call generate_stl_workflow?
        # Or better: refactor generate_stl.py to have a clean python API.

        # For now, let's implement the logic here using the imported functions
        # OR better yet, let's update generate_stl.py to have a `generate_geometry_files` function
        # and call that.

        # Let's assume we update generate_stl.py first.
        # But since I can't do that in this atomic step without multi-file edit (which is fine),
        # I'll just use the provided generate_stl_workflow but I need to mock args.

        class GeometryArgs:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        args = GeometryArgs(
            outdir=str(tri_dir),
            bore_mm=bore_mm,
            stroke_mm=stroke_mm,
            intake_port_area_m2=intake_port_area_m2,
            exhaust_port_area_m2=exhaust_port_area_m2,
            intake_open_deg=(geometry_params or {}).get("intake_open_deg"),
            intake_close_deg=(geometry_params or {}).get("intake_close_deg"),
            exhaust_open_deg=(geometry_params or {}).get("exhaust_open_deg"),
            exhaust_close_deg=(geometry_params or {}).get("exhaust_close_deg"),
            engine_start_angle_deg=(geometry_params or {}).get("engine_start_angle_deg", -180.0),
            intake_valve_rotation_sign=(geometry_params or {}).get(
                "intake_valve_rotation_sign", 1.0
            ),
            exhaust_valve_rotation_sign=(geometry_params or {}).get(
                "exhaust_valve_rotation_sign", -1.0
            ),
        )

        generate_stl_workflow(args)

    def _solver_completed_successfully(self, log_file: Path) -> bool:
        """Check if solver completed by looking for 'End' marker."""
        if not log_file.exists():
            return False
        try:
            with log_file.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                read_size = min(2048, size)
                f.seek(max(0, size - read_size))
                tail = f.read().decode("utf-8", errors="ignore")
            # Check for 'End' marker
            return "\nEnd\n" in tail or tail.strip().endswith("End")
        except Exception:
            return False

    @staticmethod
    def _rewrite_dictionary_entry(case_file: Path, key: str, value: str) -> None:
        text = case_file.read_text(encoding="utf-8")
        updated, count = re.subn(
            rf"(^\s*{re.escape(key)}\s+)([^;]+)(;)",
            rf"\g<1>{value}\g<3>",
            text,
            flags=re.MULTILINE,
        )
        if count == 0:
            raise ValueError(f"Entry '{key}' not found in {case_file}")
        case_file.write_text(updated, encoding="utf-8")

    @staticmethod
    def _upsert_dictionary_entry(case_file: Path, key: str, value: str) -> None:
        text = case_file.read_text(encoding="utf-8")
        updated, count = re.subn(
            rf"(^\s*{re.escape(key)}\s+)([^;]+)(;)",
            rf"\g<1>{value}\g<3>",
            text,
            flags=re.MULTILINE,
        )
        if count == 0:
            stripped = text.rstrip()
            updated = stripped + ("\n" if stripped else "") + f"{key} {value};\n"
        case_file.write_text(updated, encoding="utf-8")

    @staticmethod
    def _rewrite_block_entry(case_file: Path, block_name: str, key: str, value: str) -> None:
        text = case_file.read_text(encoding="utf-8")
        pattern = re.compile(
            rf"(^\s*{re.escape(block_name)}\s*\n\s*\{{.*?^\s*{re.escape(key)}\s+)([^;]+)(;)",
            flags=re.MULTILINE | re.DOTALL,
        )
        updated, count = pattern.subn(rf"\g<1>{value}\g<3>", text, count=1)
        if count == 0:
            raise ValueError(f"Entry '{block_name}.{key}' not found in {case_file}")
        case_file.write_text(updated, encoding="utf-8")

    def _default_engine_stage_sequence(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        profile_id = str(params.get("engine_stage_profile", "") or "").strip().lower()
        if profile_id not in {
            "closed_valve_ignition_v1",
            "closed_valve_ignition_fast_runtime_v1",
            "closed_valve_ignition_low_clamp_v1",
        }:
            return []
        start_angle = float(params.get("engine_start_angle_deg", -10.0))
        end_angle = float(params.get("engine_end_angle_deg", 12.0))
        stage_ends = [
            min(end_angle, start_angle + 0.75),
            min(end_angle, start_angle + 1.5),
            min(end_angle, start_angle + 3.0),
            min(end_angle, start_angle + 5.5),
            end_angle,
        ]
        base_stages = [
            {
                "name": "settle_flow",
                "end_angle_deg": stage_ends[0],
                "chemistry_enabled": False,
                "combustion_enabled": False,
                "runtime_chemistry_mode": "fullReducedKinetics",
                "deltaT": min(float(params.get("deltaT", 5.0e-6)), 2.0e-6),
                "maxDeltaT": min(float(params.get("deltaT", 5.0e-6)), 2.0e-6),
                "maxCo": 0.08,
                "engine_max_temperature_K": min(
                    float(params.get("engine_max_temperature_K", 1700.0)), 1500.0
                ),
                "engine_max_thermo_delta_K": 20.0,
                "engine_min_pressure_Pa": 2.0e4,
                "engine_min_density_kg_m3": 0.05,
                "chemistry_initial_timestep_s": 5.0e-8,
                "chemistry_abs_tol": 1.0e-13,
                "chemistry_rel_tol": 1.0e-8,
                "chemistry_tabulation_enabled": False,
                "chemistry_reduction_enabled": True,
                "combustion_cmix": 0.02,
            },
            {
                "name": "chemistry_seed",
                "end_angle_deg": stage_ends[1],
                "chemistry_enabled": True,
                "combustion_enabled": False,
                "runtime_chemistry_mode": "fullReducedKinetics",
                "deltaT": 5.0e-7,
                "maxDeltaT": 5.0e-7,
                "maxCo": 0.03,
                "engine_max_temperature_K": min(
                    float(params.get("engine_max_temperature_K", 1700.0)), 1425.0
                ),
                "engine_max_thermo_delta_K": 6.0,
                "engine_min_pressure_Pa": 2.5e4,
                "engine_min_density_kg_m3": 0.08,
                "chemistry_initial_timestep_s": 2.5e-8,
                "chemistry_abs_tol": 1.0e-14,
                "chemistry_rel_tol": 5.0e-9,
                "chemistry_tabulation_enabled": False,
                "chemistry_reduction_enabled": True,
                "combustion_cmix": 0.012,
            },
            {
                "name": "chemistry_spinup",
                "end_angle_deg": stage_ends[2],
                "chemistry_enabled": True,
                "combustion_enabled": False,
                "runtime_chemistry_mode": "fullReducedKinetics",
                "deltaT": 2.5e-7,
                "maxDeltaT": 2.5e-7,
                "maxCo": 0.02,
                "engine_max_temperature_K": min(
                    float(params.get("engine_max_temperature_K", 1700.0)), 1500.0
                ),
                "engine_max_thermo_delta_K": 4.0,
                "engine_min_pressure_Pa": 3.0e4,
                "engine_min_density_kg_m3": 0.10,
                "chemistry_initial_timestep_s": 1.0e-8,
                "chemistry_abs_tol": 1.0e-14,
                "chemistry_rel_tol": 2.5e-9,
                "chemistry_tabulation_enabled": False,
                "chemistry_reduction_enabled": True,
                "combustion_cmix": 0.01,
            },
            {
                "name": "ignition_release",
                "end_angle_deg": stage_ends[3],
                "chemistry_enabled": True,
                "combustion_enabled": True,
                "runtime_chemistry_mode": "fullReducedKinetics",
                "deltaT": 2.5e-7,
                "maxDeltaT": 2.5e-7,
                "maxCo": 0.03,
                "engine_max_temperature_K": min(
                    float(params.get("engine_max_temperature_K", 1700.0)), 1600.0
                ),
                "engine_max_thermo_delta_K": 5.0,
                "engine_min_pressure_Pa": 3.0e4,
                "engine_min_density_kg_m3": 0.10,
                "chemistry_initial_timestep_s": 2.5e-8,
                "chemistry_abs_tol": 1.0e-14,
                "chemistry_rel_tol": 2.5e-9,
                "chemistry_tabulation_enabled": False,
                "chemistry_reduction_enabled": True,
                "combustion_cmix": 0.008,
            },
            {
                "name": "early_burn",
                "end_angle_deg": stage_ends[4],
                "chemistry_enabled": True,
                "combustion_enabled": True,
                "runtime_chemistry_mode": "fullReducedKinetics",
                "deltaT": 5.0e-7,
                "maxDeltaT": 5.0e-7,
                "maxCo": 0.025,
                "engine_max_temperature_K": float(params.get("engine_max_temperature_K", 1700.0)),
                "engine_max_thermo_delta_K": 8.0,
                "engine_min_pressure_Pa": 2.5e4,
                "engine_min_density_kg_m3": 0.08,
                "chemistry_initial_timestep_s": 2.5e-8,
                "chemistry_abs_tol": 1.0e-13,
                "chemistry_rel_tol": 5.0e-9,
                "chemistry_tabulation_enabled": False,
                "chemistry_reduction_enabled": True,
                "combustion_cmix": 0.008,
            },
        ]
        if profile_id == "closed_valve_ignition_v1":
            return base_stages

        profile_overrides: dict[str, dict[str, dict[str, Any]]] = {
            "closed_valve_ignition_fast_runtime_v1": {
                "settle_flow": {
                    "deltaT": min(float(params.get("deltaT", 5.0e-6)), 4.0e-6),
                    "maxDeltaT": min(float(params.get("deltaT", 5.0e-6)), 4.0e-6),
                    "maxCo": 0.12,
                    "engine_max_thermo_delta_K": 30.0,
                },
                "chemistry_seed": {
                    "runtime_chemistry_mode": "lookupTableStrict",
                    "deltaT": 7.5e-7,
                    "maxDeltaT": 7.5e-7,
                    "maxCo": 0.045,
                    "engine_max_temperature_K": min(
                        float(params.get("engine_max_temperature_K", 1700.0)), 1475.0
                    ),
                    "engine_max_thermo_delta_K": 8.0,
                    "chemistry_initial_timestep_s": 5.0e-8,
                    "chemistry_abs_tol": 1.0e-13,
                    "chemistry_rel_tol": 1.0e-8,
                    "chemistry_tabulation_enabled": True,
                    "combustion_cmix": 0.014,
                },
                "chemistry_spinup": {
                    "runtime_chemistry_mode": "lookupTableStrict",
                    "deltaT": 4.0e-7,
                    "maxDeltaT": 4.0e-7,
                    "maxCo": 0.03,
                    "engine_max_temperature_K": min(
                        float(params.get("engine_max_temperature_K", 1700.0)), 1550.0
                    ),
                    "engine_max_thermo_delta_K": 6.0,
                    "chemistry_initial_timestep_s": 2.5e-8,
                    "chemistry_abs_tol": 1.0e-13,
                    "chemistry_rel_tol": 5.0e-9,
                    "chemistry_tabulation_enabled": True,
                },
                "ignition_release": {
                    "runtime_chemistry_mode": "lookupTableStrict",
                    "deltaT": 4.0e-7,
                    "maxDeltaT": 4.0e-7,
                    "maxCo": 0.04,
                    "engine_max_temperature_K": min(
                        float(params.get("engine_max_temperature_K", 1700.0)), 1675.0
                    ),
                    "engine_max_thermo_delta_K": 7.5,
                    "chemistry_initial_timestep_s": 5.0e-8,
                    "chemistry_abs_tol": 1.0e-13,
                    "chemistry_rel_tol": 5.0e-9,
                    "chemistry_tabulation_enabled": True,
                    "combustion_cmix": 0.010,
                    "writeInterval": 25,
                },
                "early_burn": {
                    "runtime_chemistry_mode": "lookupTableStrict",
                    "deltaT": 7.5e-7,
                    "maxDeltaT": 7.5e-7,
                    "maxCo": 0.035,
                    "engine_max_temperature_K": float(
                        params.get("engine_max_temperature_K", 1700.0)
                    ),
                    "engine_max_thermo_delta_K": 12.0,
                    "chemistry_initial_timestep_s": 5.0e-8,
                    "chemistry_abs_tol": 1.0e-12,
                    "chemistry_rel_tol": 1.0e-8,
                    "chemistry_tabulation_enabled": True,
                    "combustion_cmix": 0.010,
                    "writeInterval": 25,
                },
            },
            "closed_valve_ignition_low_clamp_v1": {
                "settle_flow": {
                    "deltaT": min(float(params.get("deltaT", 5.0e-6)), 1.5e-6),
                    "maxDeltaT": min(float(params.get("deltaT", 5.0e-6)), 1.5e-6),
                    "maxCo": 0.06,
                    "engine_max_thermo_delta_K": 18.0,
                },
                "chemistry_seed": {
                    "deltaT": 3.0e-7,
                    "maxDeltaT": 3.0e-7,
                    "maxCo": 0.02,
                    "engine_max_temperature_K": min(
                        float(params.get("engine_max_temperature_K", 1700.0)), 1500.0
                    ),
                    "engine_max_thermo_delta_K": 9.0,
                    "chemistry_initial_timestep_s": 1.5e-8,
                    "chemistry_rel_tol": 4.0e-9,
                    "combustion_cmix": 0.011,
                },
                "chemistry_spinup": {
                    "deltaT": 1.5e-7,
                    "maxDeltaT": 1.5e-7,
                    "maxCo": 0.0125,
                    "engine_max_temperature_K": min(
                        float(params.get("engine_max_temperature_K", 1700.0)), 1575.0
                    ),
                    "engine_max_thermo_delta_K": 7.0,
                    "chemistry_initial_timestep_s": 7.5e-9,
                    "chemistry_rel_tol": 2.0e-9,
                },
                "ignition_release": {
                    "deltaT": 1.5e-7,
                    "maxDeltaT": 1.5e-7,
                    "maxCo": 0.015,
                    "engine_max_temperature_K": min(
                        float(params.get("engine_max_temperature_K", 1700.0)), 1750.0
                    ),
                    "engine_max_thermo_delta_K": 8.0,
                    "chemistry_initial_timestep_s": 1.5e-8,
                    "chemistry_rel_tol": 2.0e-9,
                    "combustion_cmix": 0.007,
                },
                "early_burn": {
                    "deltaT": 3.5e-7,
                    "maxDeltaT": 3.5e-7,
                    "maxCo": 0.02,
                    "engine_max_temperature_K": max(
                        float(params.get("engine_max_temperature_K", 1700.0)), 1750.0
                    ),
                    "engine_max_thermo_delta_K": 10.0,
                    "chemistry_initial_timestep_s": 2.0e-8,
                    "chemistry_rel_tol": 4.0e-9,
                    "combustion_cmix": 0.007,
                },
            },
        }
        overrides = profile_overrides[profile_id]
        adjusted: list[dict[str, Any]] = []
        for stage in base_stages:
            merged = dict(stage)
            merged.update(dict(overrides.get(str(stage["name"]), {}) or {}))
            adjusted.append(merged)
        return adjusted

    def _apply_engine_stage_settings(
        self,
        run_dir: Path,
        *,
        base_params: dict[str, Any],
        stage: dict[str, Any],
    ) -> None:
        initial_angle = float(
            base_params.get(
                "engine_initialCrankAngleDeg", base_params.get("engine_start_angle_deg", -180.0)
            )
        )
        rpm = float(base_params.get("engine_rpm", base_params.get("rpm", 1800.0)))
        end_time = _end_time_for_target_angle(
            rpm=rpm,
            initial_angle_deg=initial_angle,
            target_angle_deg=float(stage["end_angle_deg"]),
        )
        control_dict = run_dir / "system" / "controlDict"
        engine_geometry = run_dir / "constant" / "engineGeometry"
        chemistry_properties = run_dir / "constant" / "chemistryProperties"
        combustion_properties = run_dir / "constant" / "combustionProperties"

        self._rewrite_dictionary_entry(control_dict, "startFrom", "latestTime")
        self._rewrite_dictionary_entry(control_dict, "endTime", f"{end_time:.12g}")
        self._rewrite_dictionary_entry(control_dict, "deltaT", f"{float(stage['deltaT']):.12g}")
        self._rewrite_dictionary_entry(control_dict, "maxCo", f"{float(stage['maxCo']):.12g}")
        self._rewrite_dictionary_entry(
            control_dict, "maxDeltaT", f"{float(stage['maxDeltaT']):.12g}"
        )
        self._rewrite_dictionary_entry(
            control_dict,
            "writeInterval",
            str(int(stage.get("writeInterval", base_params.get("writeInterval", 10)))),
        )

        self._rewrite_dictionary_entry(
            engine_geometry,
            "cycleEndAngleDeg",
            f"{float(stage['end_angle_deg']):.12g}",
        )
        self._rewrite_dictionary_entry(
            engine_geometry,
            "minTemperatureK",
            f"{float(stage.get('engine_min_temperature_K', base_params.get('engine_min_temperature_K', 300.0))):.12g}",
        )
        self._rewrite_dictionary_entry(
            engine_geometry,
            "maxTemperatureK",
            f"{float(stage.get('engine_max_temperature_K', base_params.get('engine_max_temperature_K', 1700.0))):.12g}",
        )
        self._rewrite_dictionary_entry(
            engine_geometry,
            "maxThermoDeltaTK",
            f"{float(stage.get('engine_max_thermo_delta_K', base_params.get('engine_max_thermo_delta_K', 250.0))):.12g}",
        )
        self._rewrite_dictionary_entry(
            engine_geometry,
            "minPressurePa",
            f"{float(stage.get('engine_min_pressure_Pa', base_params.get('engine_min_pressure_Pa', 100.0))):.12g}",
        )
        self._rewrite_dictionary_entry(
            engine_geometry,
            "minDensityKgM3",
            f"{float(stage.get('engine_min_density_kg_m3', base_params.get('engine_min_density_kg_m3', 1.0e-4))):.12g}",
        )
        self._upsert_dictionary_entry(
            engine_geometry,
            "runtimeChemistryMode",
            str(
                stage.get(
                    "runtime_chemistry_mode",
                    base_params.get("runtime_chemistry_mode", "fullReducedKinetics"),
                )
            ),
        )
        runtime_mode = str(
            stage.get(
                "runtime_chemistry_mode",
                base_params.get("runtime_chemistry_mode", "fullReducedKinetics"),
            )
        )
        runtime_strict = bool(
            stage.get(
                "runtime_chemistry_strict",
                base_params.get("runtime_chemistry_strict", runtime_mode == "lookupTableStrict"),
            )
        )
        self._upsert_dictionary_entry(
            engine_geometry,
            "runtimeChemistryStrict",
            "true" if runtime_strict else "false",
        )
        self._upsert_dictionary_entry(
            engine_geometry,
            "runtimeChemistryAbortOnAuthorityMiss",
            "true"
            if bool(
                stage.get(
                    "runtime_chemistry_abort_on_authority_miss",
                    base_params.get("runtime_chemistry_abort_on_authority_miss", runtime_strict),
                )
            )
            else "false",
        )
        self._upsert_dictionary_entry(
            engine_geometry,
            "runtimeChemistryFallbackPolicy",
            str(
                stage.get(
                    "runtime_chemistry_fallback_policy",
                    base_params.get("runtime_chemistry_fallback_policy", "fullReducedKinetics"),
                )
            ),
        )
        self._upsert_dictionary_entry(
            engine_geometry,
            "runtimeChemistryInterpolation",
            str(
                stage.get(
                    "runtime_chemistry_interpolation",
                    base_params.get("runtime_chemistry_interpolation", "local_rbf"),
                )
            ),
        )
        self._upsert_dictionary_entry(
            engine_geometry,
            "runtimeChemistryMaxUntrackedMassFraction",
            f"{float(stage.get('runtime_chemistry_max_untracked_mass_fraction', base_params.get('runtime_chemistry_max_untracked_mass_fraction', 0.02))):.12g}",
        )
        self._upsert_dictionary_entry(
            engine_geometry,
            "runtimeChemistryTableHash",
            _openfoam_word_token(
                stage.get(
                    "runtime_chemistry_table_hash",
                    base_params.get("runtime_chemistry_table_hash", "none"),
                ),
                default="none",
            ),
        )
        self._upsert_dictionary_entry(
            engine_geometry,
            "runtimeChemistryStageName",
            _openfoam_word_token(
                stage.get("name", base_params.get("runtime_chemistry_stage_name", "unset")),
                default="unset",
            ),
        )

        self._rewrite_dictionary_entry(
            chemistry_properties,
            "chemistry",
            "on" if bool(stage.get("chemistry_enabled", True)) else "off",
        )
        self._rewrite_dictionary_entry(
            chemistry_properties,
            "initialChemicalTimeStep",
            f"{float(stage.get('chemistry_initial_timestep_s', base_params.get('chemistry_initial_timestep_s', 1.0e-7))):.12g}",
        )
        self._rewrite_block_entry(
            chemistry_properties,
            "odeCoeffs",
            "absTol",
            f"{float(stage.get('chemistry_abs_tol', base_params.get('chemistry_abs_tol', 1.0e-12))):.12g}",
        )
        self._rewrite_block_entry(
            chemistry_properties,
            "odeCoeffs",
            "relTol",
            f"{float(stage.get('chemistry_rel_tol', base_params.get('chemistry_rel_tol', 1.0e-7))):.12g}",
        )
        self._rewrite_block_entry(
            chemistry_properties,
            "reduction",
            "active",
            "on" if bool(stage.get("chemistry_reduction_enabled", True)) else "off",
        )
        self._rewrite_block_entry(
            chemistry_properties,
            "tabulation",
            "active",
            "on" if bool(stage.get("chemistry_tabulation_enabled", True)) else "off",
        )

        self._rewrite_dictionary_entry(
            combustion_properties,
            "active",
            "yes" if bool(stage.get("combustion_enabled", True)) else "no",
        )
        self._rewrite_block_entry(
            combustion_properties,
            "PaSRCoeffs",
            "Cmix",
            f"{float(stage.get('combustion_cmix', base_params.get('combustion_cmix', 0.03))):.12g}",
        )

    def _run_solver_with_custom_dirs_log(
        self,
        run_dir: Path,
        *,
        custom_solver_dirs: list[str | Path] | None,
        log_name: str,
    ) -> tuple[bool, str]:
        solver_log_file = run_dir / log_name
        code, _, _ = self.docker.run_solver(
            solver=self.solver_cmd,
            case_dir=run_dir,
            timeout_s=self.docker_timeout_s,
            log_file=solver_log_file,
            custom_solver_dirs=list(custom_solver_dirs or []),
        )
        if code != 0:
            if self._solver_completed_successfully(solver_log_file):
                return True, ""
            return False, "solver"
        return True, ""

    @staticmethod
    def _latest_log_summary(run_dir: Path) -> dict[str, float]:
        summaries = sorted(
            run_dir.glob("logSummary.*.dat"),
            key=lambda path: float(path.name[len("logSummary.") : -len(".dat")]),
        )
        if not summaries:
            return {}
        latest = summaries[-1]
        rows = [
            line.strip()
            for line in latest.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not rows:
            return {}
        crank, pressure, temperature, velocity = rows[-1].split()
        return {
            "time_s": float(latest.name[len("logSummary.") : -len(".dat")]),
            "crank_angle_deg": float(crank),
            "mean_pressure_Pa": float(pressure),
            "mean_temperature_K": float(temperature),
            "mean_velocity_magnitude_m_s": float(velocity),
        }

    def _stage_completion_status(
        self,
        run_dir: Path,
        *,
        base_params: dict[str, Any],
        stage: dict[str, Any],
    ) -> dict[str, Any]:
        latest = self._latest_log_summary(run_dir)
        if not latest:
            return {}
        initial_angle = float(
            base_params.get(
                "engine_initialCrankAngleDeg",
                base_params.get("engine_start_angle_deg", -180.0),
            )
        )
        rpm = float(base_params.get("engine_rpm", base_params.get("rpm", 1800.0)))
        target_angle = float(stage["end_angle_deg"])
        target_time = _end_time_for_target_angle(
            rpm=rpm,
            initial_angle_deg=initial_angle,
            target_angle_deg=target_angle,
        )
        stage_dt = abs(float(stage.get("deltaT", base_params.get("deltaT", 0.0))))
        time_tolerance_s = float(
            stage.get(
                "completion_time_tolerance_s",
                base_params.get(
                    "engine_stage_completion_time_tolerance_s",
                    max(stage_dt * 0.1, 1.0e-8),
                ),
            )
        )
        angle_tolerance_deg = float(
            stage.get(
                "completion_angle_tolerance_deg",
                base_params.get("engine_stage_completion_angle_tolerance_deg", 0.0),
            )
        )
        latest_time = float(latest["time_s"])
        latest_angle = float(latest["crank_angle_deg"])
        time_gap = float(target_time - latest_time)
        angle_gap = float(target_angle - latest_angle)
        within_time = latest_time >= target_time or time_gap <= time_tolerance_s
        within_angle = angle_tolerance_deg > 0.0 and abs(angle_gap) <= angle_tolerance_deg
        return {
            "target_time_s": float(target_time),
            "latest_time_s": latest_time,
            "time_gap_s": time_gap,
            "time_tolerance_s": time_tolerance_s,
            "target_angle_deg": target_angle,
            "latest_angle_deg": latest_angle,
            "angle_gap_deg": angle_gap,
            "angle_tolerance_deg": angle_tolerance_deg,
            "within_tolerance": bool(within_time or within_angle),
        }

    def _load_engine_stage_manifest(self, run_dir: Path) -> dict[str, Any]:
        manifest_path = run_dir / "engine_stage_manifest.json"
        if not manifest_path.exists():
            return {}
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _remaining_engine_stages(
        self,
        *,
        base_params: dict[str, Any],
        manifest: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        stages = self._default_engine_stage_sequence(base_params)
        manifest_payload = manifest if isinstance(manifest, dict) else {}
        stage_lookup = {
            str(entry.get("name", "")).strip(): entry
            for entry in manifest_payload.get("stages", [])
            if isinstance(entry, dict) and str(entry.get("name", "")).strip()
        }
        remaining: list[dict[str, Any]] = []
        seen_incomplete = False
        for stage in stages:
            entry = stage_lookup.get(str(stage.get("name", "")).strip())
            if not seen_incomplete and entry is not None and entry.get("ok") is True:
                continue
            remaining.append(dict(stage))
            if entry is None or entry.get("ok") is not True:
                seen_incomplete = True
        return remaining

    def write_engine_stage_resume_summary(
        self,
        run_dir: Path,
        *,
        base_params: dict[str, Any],
        results: list[dict[str, Any]],
        docker_timeout_s: int | None = None,
    ) -> dict[str, Any]:
        manifest = self._load_engine_stage_manifest(run_dir)
        remaining = self._remaining_engine_stages(base_params=base_params, manifest=manifest)
        summary: dict[str, Any] = {
            "remaining_stages": [str(stage.get("name", "")) for stage in remaining],
            "current_stage": str(remaining[0].get("name", "")) if remaining else None,
            "results": list(results),
        }
        if docker_timeout_s is not None:
            summary["docker_timeout_s"] = int(docker_timeout_s)
        (run_dir / "engine_stage_resume_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    def _run_engine_stage_profile(
        self,
        run_dir: Path,
        *,
        base_params: dict[str, Any],
        custom_solver_dirs: list[str | Path] | None,
    ) -> tuple[bool, str]:
        stages = self._default_engine_stage_sequence(base_params)
        if not stages:
            return self._run_solver_with_custom_dirs_log(
                run_dir,
                custom_solver_dirs=custom_solver_dirs,
                log_name=f"{self.solver_cmd}.log",
            )
        manifest_entries: list[dict[str, Any]] = []
        for index, stage in enumerate(stages, start=1):
            stage_payload = dict(stage)
            self._apply_engine_stage_settings(run_dir, base_params=base_params, stage=stage_payload)
            ok, stage_result = self._run_solver_with_custom_dirs_log(
                run_dir,
                custom_solver_dirs=custom_solver_dirs,
                log_name=f"{self.solver_cmd}.stage_{index:02d}_{stage_payload['name']}.log",
            )
            completion_status = self._stage_completion_status(
                run_dir,
                base_params=base_params,
                stage=stage_payload,
            )
            completion_mode = "solver_end" if ok else "solver"
            if not ok and completion_status.get("within_tolerance"):
                ok = True
                stage_result = ""
                completion_mode = "near_target_tolerance"
            manifest_entries.append(
                {
                    **stage_payload,
                    "ok": bool(ok),
                    "stage_result": str(stage_result),
                    "completion_mode": completion_mode,
                    "completion_status": completion_status,
                }
            )
            (run_dir / "engine_stage_manifest.json").write_text(
                json.dumps(
                    {
                        "profile": str(base_params.get("engine_stage_profile", "")),
                        "stages": manifest_entries,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if not ok:
                return False, str(stage_payload["name"])
        latest_stage_log = (
            run_dir / f"{self.solver_cmd}.stage_{len(stages):02d}_{stages[-1]['name']}.log"
        )
        if latest_stage_log.exists():
            shutil.copy2(latest_stage_log, run_dir / f"{self.solver_cmd}.log")
        return True, ""

    def run_meshing(self, run_dir: Path) -> tuple[bool, str]:
        """Execute the meshing sequence (BlockMesh -> Snappy -> ...)."""
        is_overset = (run_dir / "system" / "snappyHexMeshDict.background").exists()
        is_sliding = (run_dir / "system" / "snappyHexMeshDict").exists() and not is_overset

        if is_overset:
            # 1. Background Mesh
            c, _, _ = self.docker.run_utility(
                utility="blockMesh",
                case_dir=run_dir,
                log_file=run_dir / "blockMesh.log",
                timeout_s=600,
            )
            if c != 0:
                return False, "blockMesh"

            c, _, _ = self.docker.run_utility(
                utility="snappyHexMesh",
                args=["-dict", "system/snappyHexMeshDict.background", "-overwrite"],
                case_dir=run_dir,
                log_file=run_dir / "snappyHexMesh.background.log",
                timeout_s=1200,
            )
            if c != 0:
                return False, "snappyHexMesh.background"

            # Save background
            if (run_dir / "constant/polyMesh.background").exists():
                shutil.rmtree(run_dir / "constant/polyMesh.background")
            shutil.move(run_dir / "constant/polyMesh", run_dir / "constant/polyMesh.background")

            # 2. Valve Mesh
            c, _, _ = self.docker.run_utility(
                utility="blockMesh", case_dir=run_dir, timeout_s=600
            )  # Re-run blockMesh for valve domain

            c, _, _ = self.docker.run_utility(
                utility="snappyHexMesh",
                args=["-dict", "system/snappyHexMeshDict.valve", "-overwrite"],
                case_dir=run_dir,
                log_file=run_dir / "snappyHexMesh.valve.log",
                timeout_s=1200,
            )
            if c != 0:
                return False, "snappyHexMesh.valve"

            # Save valve
            if (run_dir / "constant/polyMesh.valves").exists():
                shutil.rmtree(run_dir / "constant/polyMesh.valves")
            shutil.move(run_dir / "constant/polyMesh", run_dir / "constant/polyMesh.valves")

            # Restore background as master
            shutil.move(run_dir / "constant/polyMesh.background", run_dir / "constant/polyMesh")

            # 3. Merge
            valve_case_dir = run_dir / "valve_mesh_case"
            if valve_case_dir.exists():
                shutil.rmtree(valve_case_dir)
            valve_case_dir.mkdir()
            (valve_case_dir / "constant").mkdir()
            shutil.copytree(run_dir / "system", valve_case_dir / "system")
            shutil.copytree(
                run_dir / "constant/polyMesh.valves", valve_case_dir / "constant/polyMesh"
            )

            c, _, _ = self.docker.run_utility(
                utility="mergeMeshes",
                args=[".", "valve_mesh_case", "-overwrite"],
                case_dir=run_dir,
                log_file=run_dir / "mergeMeshes.log",
                timeout_s=600,
            )
            if c != 0:
                return False, "mergeMeshes"

            if valve_case_dir.exists():
                shutil.rmtree(valve_case_dir)

            # 4. Zones (TopoSet)
            c, _, _ = self.docker.run_utility(
                utility="topoSet",
                case_dir=run_dir,
                log_file=run_dir / "topoSet.log",
                timeout_s=600,
            )
            if c != 0:
                return False, "topoSet"

            # 5. Fields (SetFields)
            c, _, _ = self.docker.run_utility(
                utility="setFields",
                case_dir=run_dir,
                log_file=run_dir / "setFields.log",
                timeout_s=600,
            )
            if c != 0:
                return False, "setFields"

        else:
            # Standard / Sliding Mesh Pipeline
            c, _, _ = self.docker.run_utility(
                utility="blockMesh",
                case_dir=run_dir,
                timeout_s=300,
                log_file=run_dir / "blockMesh.log",
            )
            if c != 0:
                return False, "blockMesh"

            if is_sliding:
                c, _, _ = self.docker.run_utility(
                    utility="snappyHexMesh",
                    args=["-overwrite"],
                    case_dir=run_dir,
                    log_file=run_dir / "snappyHexMesh.log",
                    timeout_s=1200,
                )
                if c != 0:
                    return False, "snappyHexMesh"

                c, _, _ = self.docker.run_utility(
                    utility="topoSet",
                    case_dir=run_dir,
                    log_file=run_dir / "topoSet.log",
                )
                if c != 0:
                    return False, "topoSet"

                if (run_dir / "system/createBafflesDict").exists():
                    c, _, _ = self.docker.run_utility(
                        utility="createBaffles",
                        args=["-overwrite"],
                        case_dir=run_dir,
                        log_file=run_dir / "createBaffles.log",
                    )
                    if c != 0:
                        return False, "createBaffles"

                if (run_dir / "system/createPatchDict").exists():
                    c, _, _ = self.docker.run_utility(
                        utility="createPatch",
                        args=["-overwrite"],
                        case_dir=run_dir,
                        log_file=run_dir / "createPatch.log",
                    )
                    if c != 0:
                        return False, "createPatch"

                c, _, _ = self.docker.run_utility(
                    utility="setFields",
                    case_dir=run_dir,
                    log_file=run_dir / "setFields.log",
                )
                if c != 0:
                    return False, "setFields"
                self.runner.repair_ami_boundary_values(run_dir, time_dir="0")

        return True, ""

    def run_solver(self, run_dir: Path) -> tuple[bool, str]:
        """Execute the solver."""
        return self.run_solver_with_custom_dirs(run_dir, custom_solver_dirs=None)

    def run_solver_with_custom_dirs(
        self,
        run_dir: Path,
        *,
        custom_solver_dirs: list[str | Path] | None,
    ) -> tuple[bool, str]:
        """Execute the solver with optional repo-owned custom solver search paths."""
        return self._run_solver_with_custom_dirs_log(
            run_dir,
            custom_solver_dirs=custom_solver_dirs,
            log_name=f"{self.solver_cmd}.log",
        )

    def parse_results(self, run_dir: Path) -> dict[str, float]:
        """Parse results using the runner adapter."""
        return self.runner.parse_results(run_dir, log_name=f"{self.solver_cmd}.log")

    def _ensure_case_metrics(self, run_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
        metrics = self.parse_results(run_dir)
        required = {
            "trapped_mass",
            "scavenging_efficiency",
            "residual_fraction",
            "trapped_o2_mass",
        }
        if required.issubset(metrics):
            return metrics

        latest_dir = self.runner.latest_time_dir(run_dir)
        if latest_dir is None:
            return metrics

        if not self.runner.has_cell_volume_field(latest_dir):
            code, _, _ = self.docker.run_utility(
                utility="postProcess",
                args=["-func", "writeCellVolumes", "-time", latest_dir.name],
                case_dir=run_dir,
                log_file=run_dir / "postProcess.log",
                timeout_s=600,
            )
            if code != 0:
                return metrics

        field_metrics = self.runner.compute_field_metrics(
            run_dir,
            p_manifold_Pa=float(params.get("p_manifold_Pa", 101325.0)),
        )
        if not required.issubset(field_metrics):
            return metrics

        self.runner.emit_metrics(run_dir, field_metrics, log_name=f"{self.solver_cmd}.log")
        return self.parse_results(run_dir)

    def _emit_engine_results(
        self,
        run_dir: Path,
        *,
        case_params: dict[str, Any],
        engine_metrics: dict[str, Any] | None,
        engine_handoff: dict[str, Any],
        package_manifest: dict[str, Any],
        solver_metadata: dict[str, str],
        run_ok: bool,
        stage: str,
    ) -> dict[str, Any] | None:
        if not self._is_default_engine_template():
            return None
        if not any(run_dir.glob("logSummary.*.dat")):
            return None
        try:
            return emit_engine_results_artifact(
                engine_case_dir=run_dir,
                params=case_params,
                engine_metrics=engine_metrics,
                solver_name=self.solver_cmd,
                handoff_bundle_id=str(engine_handoff.get("bundle_id", "")),
                mechanism_id=str(engine_handoff.get("mechanism_id", "")),
                openfoam_chemistry_package_id=str(package_manifest.get("package_id", "")),
                openfoam_chemistry_package_hash=str(package_manifest.get("package_hash", "")),
                custom_solver_source_hash=str(solver_metadata.get("source_hash", "")),
                run_ok=run_ok,
                stage=stage,
            )
        except Exception:
            return None

    def execute(
        self,
        run_dir: Path,
        params: dict[str, Any],
        geometry_args: dict[str, float] | None = None,
        *,
        staged_inputs: list[dict[str, str]] | None = None,
        handoff_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute full pipeline for a single case."""
        solver_metadata = self._ensure_custom_solver(log_file=run_dir / "custom_solver_build.log")
        (
            case_params,
            case_staged_inputs,
            engine_handoff,
            package_manifest,
            runtime_table_manifest,
        ) = self._engine_case_assets(
            params,
            handoff_bundle=handoff_bundle,
            staged_inputs=staged_inputs,
        )

        # 1. Setup
        self.setup_case(run_dir, case_params, staged_inputs=case_staged_inputs)

        # 2. Geometry
        if geometry_args:
            self.generate_geometry(
                run_dir,
                bore_mm=geometry_args["bore_mm"],
                stroke_mm=geometry_args["stroke_mm"],
                intake_port_area_m2=geometry_args["intake_port_area_m2"],
                exhaust_port_area_m2=geometry_args["exhaust_port_area_m2"],
                geometry_params=case_params,
            )

        # 3. Mesh
        ok, stage = self.run_meshing(run_dir)
        if not ok:
            return {
                "error": 1.0,
                "stage": stage,
                "ok": False,
                "solver_name": self.solver_cmd,
                "handoff_bundle_id": str(engine_handoff.get("bundle_id", "")),
            }

        # 4. Solve
        custom_solver_dirs = (
            [Path(str(solver_metadata.get("binary_path", ""))).parent]
            if solver_metadata.get("binary_path")
            else None
        )
        if (
            self._is_default_engine_template()
            and str(case_params.get("engine_stage_profile", "")).strip()
        ):
            ok, stage = self._run_engine_stage_profile(
                run_dir,
                base_params=case_params,
                custom_solver_dirs=custom_solver_dirs,
            )
        else:
            ok, stage = self.run_solver_with_custom_dirs(
                run_dir,
                custom_solver_dirs=custom_solver_dirs,
            )
        if not ok:
            authority_miss_path = (
                str(run_dir / "runtimeChemistryAuthorityMiss.json")
                if (run_dir / "runtimeChemistryAuthorityMiss.json").exists()
                else ""
            )
            engine_results = self._emit_engine_results(
                run_dir,
                case_params=case_params,
                engine_metrics=self.parse_results(run_dir),
                engine_handoff=engine_handoff,
                package_manifest=package_manifest,
                solver_metadata=solver_metadata,
                run_ok=False,
                stage=stage,
            )
            return {
                "error": 1.0,
                "stage": stage,
                "ok": False,
                "solver_name": self.solver_cmd,
                "handoff_bundle_id": str(engine_handoff.get("bundle_id", "")),
                "runtime_chemistry_table_id": str(runtime_table_manifest.get("table_id", "")),
                "runtime_chemistry_interpolation_method": str(
                    runtime_table_manifest.get("interpolation_method", "")
                ),
                "runtime_chemistry_jacobian_mode": str(
                    runtime_table_manifest.get("jacobian_mode", "")
                ),
                "runtime_chemistry_authority_miss_path": authority_miss_path,
                "engine_results_path": (
                    str(run_dir / "engine_results.json") if engine_results is not None else ""
                ),
            }

        # 5. Parse
        metrics = self._ensure_case_metrics(run_dir, case_params)
        required = {
            "trapped_mass",
            "scavenging_efficiency",
            "residual_fraction",
            "trapped_o2_mass",
        }
        if not required.issubset(metrics):
            authority_miss_path = (
                str(run_dir / "runtimeChemistryAuthorityMiss.json")
                if (run_dir / "runtimeChemistryAuthorityMiss.json").exists()
                else ""
            )
            engine_results = self._emit_engine_results(
                run_dir,
                case_params=case_params,
                engine_metrics=metrics,
                engine_handoff=engine_handoff,
                package_manifest=package_manifest,
                solver_metadata=solver_metadata,
                run_ok=False,
                stage="metrics",
            )
            return {
                "error": 1.0,
                "stage": "metrics",
                "ok": False,
                "solver_name": self.solver_cmd,
                "handoff_bundle_id": str(engine_handoff.get("bundle_id", "")),
                "runtime_chemistry_table_id": str(runtime_table_manifest.get("table_id", "")),
                "runtime_chemistry_interpolation_method": str(
                    runtime_table_manifest.get("interpolation_method", "")
                ),
                "runtime_chemistry_jacobian_mode": str(
                    runtime_table_manifest.get("jacobian_mode", "")
                ),
                "runtime_chemistry_authority_miss_path": authority_miss_path,
                "engine_results_path": (
                    str(run_dir / "engine_results.json") if engine_results is not None else ""
                ),
            }
        engine_results = self._emit_engine_results(
            run_dir,
            case_params=case_params,
            engine_metrics=metrics,
            engine_handoff=engine_handoff,
            package_manifest=package_manifest,
            solver_metadata=solver_metadata,
            run_ok=True,
            stage="complete",
        )
        result = {
            **metrics,
            "ok": True,
            "solver_name": self.solver_cmd,
            "handoff_bundle_id": str(engine_handoff.get("bundle_id", "")),
            "mechanism_id": str(engine_handoff.get("mechanism_id", "")),
            "openfoam_chemistry_package_id": str(package_manifest.get("package_id", "")),
            "openfoam_chemistry_package_hash": str(package_manifest.get("package_hash", "")),
            "runtime_chemistry_table_id": str(runtime_table_manifest.get("table_id", "")),
            "runtime_chemistry_table_hash": str(
                runtime_table_manifest.get("generated_file_hashes", {}).get(
                    "runtimeChemistryTable", ""
                )
            ),
            "runtime_chemistry_interpolation_method": str(
                runtime_table_manifest.get("interpolation_method", "")
            ),
            "runtime_chemistry_jacobian_mode": str(runtime_table_manifest.get("jacobian_mode", "")),
            "runtime_chemistry_authority_miss_path": (
                str(run_dir / "runtimeChemistryAuthorityMiss.json")
                if (run_dir / "runtimeChemistryAuthorityMiss.json").exists()
                else ""
            ),
            "custom_solver_source_hash": str(solver_metadata.get("source_hash", "")),
            "custom_solver_binary": str(solver_metadata.get("binary_path", "")),
            "engine_results_path": (
                str(run_dir / "engine_results.json") if engine_results is not None else ""
            ),
        }
        if engine_results is not None:
            extracted_metrics = dict(engine_results.get("metrics", {}) or {})
            for key in (
                "peak_pressure_Pa",
                "peak_pressure_crank_angle_deg",
                "ca10_deg",
                "ca50_deg",
                "ca90_deg",
                "imep_Pa",
                "net_indicated_work_J",
            ):
                if extracted_metrics.get(key) is not None:
                    result[key] = extracted_metrics[key]
        return result

    def engine_smoke_gate(
        self,
        run_dir: Path,
        *,
        params: dict[str, Any],
        geometry_args: dict[str, float] | None = None,
        handoff_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a single repo-default engine proof case and require finite core metrics."""
        smoke_params = dict(params)
        if self._is_default_engine_template():
            smoke_params.setdefault("engine_proof_mode", DEFAULT_ENGINE_PROOF_MODE)
        result = self.execute(
            run_dir=run_dir,
            params=smoke_params,
            geometry_args=geometry_args,
            handoff_bundle=handoff_bundle,
        )
        required = {
            "trapped_mass",
            "scavenging_efficiency",
            "residual_fraction",
            "trapped_o2_mass",
        }
        if not bool(result.get("ok", False)) or not required.issubset(result):
            raise RuntimeError(
                f"Engine smoke gate failed for '{self.template_dir}': stage={result.get('stage', 'unknown')}"
            )
        return result
