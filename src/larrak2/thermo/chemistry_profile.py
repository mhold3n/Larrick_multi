"""Hybrid chemistry profile loader for staged thermo evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

SPARK_TIMING_VAR_NAME = "spark_timing_deg_from_compression_tdc"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_THERMO_CHEMISTRY_PROFILE_PATH = (
    _repo_root() / "data" / "thermo" / "hybrid_chemistry_profile_v1.json"
)


@dataclass(frozen=True)
class FuelChemistryProfile:
    fuel_name: str
    afr_stoich: float
    fuel_lhv: float
    o2_required_per_fuel: float
    evaporation_tau_scale: float
    wall_wetting_factor: float
    charge_cooling_factor: float
    mixture_homogeneity_factor: float
    ignition_delay_ref_s: float
    ignition_temp_ref_k: float
    ignition_pressure_ref_pa: float
    ignition_temp_activation_k: float
    ignition_pressure_exponent: float
    ignition_lambda_sensitivity: float
    spark_assist_factor: float
    ignitability_target_delay_s: float
    burn_duration_base_deg: float
    burn_duration_lambda_sensitivity: float
    burn_duration_inhomogeneity_sensitivity: float


@dataclass(frozen=True)
class ChemistryHardwareProfile:
    injector_type: str
    plenum_volume_factor: float
    wall_film_base_fraction: float
    mixture_homogeneity_base: float
    mixing_length_factor: float
    evaporation_tau_ref_s: float
    evaporation_temp_exponent: float
    evaporation_pressure_exponent: float
    charge_cooling_gain_k: float
    mixture_inhomogeneity_lambda_gain: float
    mixture_inhomogeneity_rpm_gain: float


@dataclass(frozen=True)
class ChemistryThresholds:
    delivered_vapor_fraction_min: float
    mixture_inhomogeneity_max: float
    wall_film_fraction_max: float
    ignitability_margin_min: float
    preignition_margin_min: float


@dataclass(frozen=True)
class WiebeHandoffProfile:
    anchor_mode: str
    handoff_burn_fraction: float
    burn_duration_min_deg: float
    burn_duration_max_deg: float
    chemistry_weight: float
    legacy_heat_release_weight: float


@dataclass(frozen=True)
class ThermoChemistryProfile:
    profile_id: str
    profile_version: str
    spark_timing_lower: float
    spark_timing_upper: float
    spark_timing_legacy_default: float
    hardware: ChemistryHardwareProfile
    thresholds: ChemistryThresholds
    wiebe_handoff: WiebeHandoffProfile
    fuel_profiles: dict[str, FuelChemistryProfile]
    source_references: tuple[str, ...]
    path: str


_DEF_FUEL_KEYS = (
    "afr_stoich",
    "fuel_lhv",
    "o2_required_per_fuel",
    "evaporation_tau_scale",
    "wall_wetting_factor",
    "charge_cooling_factor",
    "mixture_homogeneity_factor",
    "ignition_delay_ref_s",
    "ignition_temp_ref_k",
    "ignition_pressure_ref_pa",
    "ignition_temp_activation_k",
    "ignition_pressure_exponent",
    "ignition_lambda_sensitivity",
    "spark_assist_factor",
    "ignitability_target_delay_s",
    "burn_duration_base_deg",
    "burn_duration_lambda_sensitivity",
    "burn_duration_inhomogeneity_sensitivity",
)


def _as_path(raw: str | Path | None, default_path: Path) -> Path:
    if raw is None:
        return default_path
    text = str(raw).strip()
    if not text:
        return default_path
    path = Path(text)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _require_keys(payload: dict[str, Any], keys: tuple[str, ...], *, label: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


def _validate_profile(path: Path, payload: dict[str, Any]) -> ThermoChemistryProfile:
    _require_keys(
        payload,
        (
            "profile_id",
            "profile_version",
            "spark_timing_bounds",
            "hardware",
            "thresholds",
            "wiebe_handoff",
            "fuel_profiles",
            "source_references",
        ),
        label="Thermo chemistry profile",
    )

    spark = dict(payload.get("spark_timing_bounds", {}) or {})
    _require_keys(spark, ("lower", "upper", "legacy_default"), label="spark_timing_bounds")
    spark_lo = float(spark["lower"])
    spark_hi = float(spark["upper"])
    spark_default = float(spark["legacy_default"])
    if not np.isfinite(spark_lo) or not np.isfinite(spark_hi) or spark_lo >= spark_hi:
        raise ValueError("spark_timing_bounds must define finite lower < upper")
    if not spark_lo <= spark_default <= spark_hi:
        raise ValueError("spark_timing_bounds legacy_default must be within bounds")

    hw_payload = dict(payload.get("hardware", {}) or {})
    _require_keys(
        hw_payload,
        (
            "injector_type",
            "plenum_volume_factor",
            "wall_film_base_fraction",
            "mixture_homogeneity_base",
            "mixing_length_factor",
            "evaporation_tau_ref_s",
            "evaporation_temp_exponent",
            "evaporation_pressure_exponent",
            "charge_cooling_gain_k",
            "mixture_inhomogeneity_lambda_gain",
            "mixture_inhomogeneity_rpm_gain",
        ),
        label="hardware",
    )
    hardware = ChemistryHardwareProfile(
        injector_type=str(hw_payload["injector_type"]),
        plenum_volume_factor=float(hw_payload["plenum_volume_factor"]),
        wall_film_base_fraction=float(hw_payload["wall_film_base_fraction"]),
        mixture_homogeneity_base=float(hw_payload["mixture_homogeneity_base"]),
        mixing_length_factor=float(hw_payload["mixing_length_factor"]),
        evaporation_tau_ref_s=float(hw_payload["evaporation_tau_ref_s"]),
        evaporation_temp_exponent=float(hw_payload["evaporation_temp_exponent"]),
        evaporation_pressure_exponent=float(hw_payload["evaporation_pressure_exponent"]),
        charge_cooling_gain_k=float(hw_payload["charge_cooling_gain_k"]),
        mixture_inhomogeneity_lambda_gain=float(hw_payload["mixture_inhomogeneity_lambda_gain"]),
        mixture_inhomogeneity_rpm_gain=float(hw_payload["mixture_inhomogeneity_rpm_gain"]),
    )

    thr_payload = dict(payload.get("thresholds", {}) or {})
    _require_keys(
        thr_payload,
        (
            "delivered_vapor_fraction_min",
            "mixture_inhomogeneity_max",
            "wall_film_fraction_max",
            "ignitability_margin_min",
            "preignition_margin_min",
        ),
        label="thresholds",
    )
    thresholds = ChemistryThresholds(
        delivered_vapor_fraction_min=float(thr_payload["delivered_vapor_fraction_min"]),
        mixture_inhomogeneity_max=float(thr_payload["mixture_inhomogeneity_max"]),
        wall_film_fraction_max=float(thr_payload["wall_film_fraction_max"]),
        ignitability_margin_min=float(thr_payload["ignitability_margin_min"]),
        preignition_margin_min=float(thr_payload["preignition_margin_min"]),
    )

    wiebe_payload = dict(payload.get("wiebe_handoff", {}) or {})
    _require_keys(
        wiebe_payload,
        (
            "anchor_mode",
            "handoff_burn_fraction",
            "burn_duration_min_deg",
            "burn_duration_max_deg",
            "chemistry_weight",
            "legacy_heat_release_weight",
        ),
        label="wiebe_handoff",
    )
    wiebe_handoff = WiebeHandoffProfile(
        anchor_mode=str(wiebe_payload["anchor_mode"]),
        handoff_burn_fraction=float(wiebe_payload["handoff_burn_fraction"]),
        burn_duration_min_deg=float(wiebe_payload["burn_duration_min_deg"]),
        burn_duration_max_deg=float(wiebe_payload["burn_duration_max_deg"]),
        chemistry_weight=float(wiebe_payload["chemistry_weight"]),
        legacy_heat_release_weight=float(wiebe_payload["legacy_heat_release_weight"]),
    )
    if not 0.0 <= wiebe_handoff.handoff_burn_fraction <= 1.0:
        raise ValueError("wiebe_handoff.handoff_burn_fraction must be within [0,1]")
    if wiebe_handoff.burn_duration_min_deg <= 0.0 or (
        wiebe_handoff.burn_duration_max_deg <= wiebe_handoff.burn_duration_min_deg
    ):
        raise ValueError("wiebe_handoff burn duration bounds are invalid")

    fuel_profiles_raw = dict(payload.get("fuel_profiles", {}) or {})
    if not fuel_profiles_raw:
        raise ValueError("Thermo chemistry profile requires non-empty fuel_profiles")
    fuel_profiles: dict[str, FuelChemistryProfile] = {}
    for fuel_name, fuel_payload_any in fuel_profiles_raw.items():
        fuel_payload = dict(fuel_payload_any or {})
        _require_keys(fuel_payload, _DEF_FUEL_KEYS, label=f"fuel_profiles.{fuel_name}")
        fuel_profiles[str(fuel_name)] = FuelChemistryProfile(
            fuel_name=str(fuel_name),
            **{key: float(fuel_payload[key]) for key in _DEF_FUEL_KEYS},
        )

    refs = tuple(str(v) for v in list(payload.get("source_references", []) or []))
    if not refs:
        raise ValueError("Thermo chemistry profile requires non-empty source_references")

    return ThermoChemistryProfile(
        profile_id=str(payload["profile_id"]),
        profile_version=str(payload["profile_version"]),
        spark_timing_lower=spark_lo,
        spark_timing_upper=spark_hi,
        spark_timing_legacy_default=spark_default,
        hardware=hardware,
        thresholds=thresholds,
        wiebe_handoff=wiebe_handoff,
        fuel_profiles=fuel_profiles,
        source_references=refs,
        path=str(path),
    )


@lru_cache(maxsize=8)
def load_thermo_chemistry_profile(path: str | Path | None = None) -> ThermoChemistryProfile:
    profile_path = _as_path(path, DEFAULT_THERMO_CHEMISTRY_PROFILE_PATH)
    if not profile_path.exists():
        raise FileNotFoundError(f"Thermo chemistry profile not found: {profile_path}")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {profile_path}")
    return _validate_profile(profile_path, payload)


def spark_timing_bounds(profile_path: str | Path | None = None) -> tuple[float, float]:
    profile = load_thermo_chemistry_profile(profile_path)
    return float(profile.spark_timing_lower), float(profile.spark_timing_upper)


def legacy_spark_timing_default(profile_path: str | Path | None = None) -> float:
    profile = load_thermo_chemistry_profile(profile_path)
    return float(profile.spark_timing_legacy_default)


def fuel_profile_for_name(
    fuel_name: str | None,
    *,
    profile_path: str | Path | None = None,
) -> FuelChemistryProfile:
    profile = load_thermo_chemistry_profile(profile_path)
    key = str(fuel_name or "gasoline").strip().lower() or "gasoline"
    if key not in profile.fuel_profiles:
        raise ValueError(
            f"Unsupported fuel_name '{fuel_name}'. Expected one of {sorted(profile.fuel_profiles)}"
        )
    return profile.fuel_profiles[key]


__all__ = [
    "DEFAULT_THERMO_CHEMISTRY_PROFILE_PATH",
    "FuelChemistryProfile",
    "SPARK_TIMING_VAR_NAME",
    "ThermoChemistryProfile",
    "fuel_profile_for_name",
    "legacy_spark_timing_default",
    "load_thermo_chemistry_profile",
    "spark_timing_bounds",
]
