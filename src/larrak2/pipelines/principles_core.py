"""Reduced-order principles problem definitions and deterministic expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from larrak2.core.encoding import (
    N_TOTAL,
    bounds,
    decode_candidate,
    mid_bounds_candidate,
    variable_manifest,
)

PRINCIPLES_OBJECTIVE_NAMES: tuple[str, ...] = (
    "eta_comb_gap",
    "eta_exp_gap",
    "eta_gear_gap",
    "motion_law_penalty",
    "life_damage_penalty",
    "material_risk_penalty",
)

REDUCED_VARIABLE_NAMES: tuple[str, ...] = (
    "compression_duration",
    "expansion_duration",
    "heat_release_center",
    "heat_release_width",
    "lambda_af",
    "intake_open_offset_from_bdc",
    "intake_duration_deg",
    "exhaust_open_offset_from_expansion_tdc",
    "exhaust_duration_deg",
    "spark_timing_deg_from_compression_tdc",
    "base_radius",
    "pitch_coeff_0",
    "pitch_coeff_1",
    "pitch_coeff_2",
    "pitch_coeff_3",
    "pitch_coeff_4",
    "face_width_mm",
)

_FULL_INDEX_BY_NAME = {meta.name: int(meta.index) for meta in variable_manifest()}
REDUCED_FULL_INDICES: tuple[int, ...] = tuple(_FULL_INDEX_BY_NAME[name] for name in REDUCED_VARIABLE_NAMES)
REALWORLD_NAMES: tuple[str, ...] = (
    "surface_finish_level",
    "lube_mode_level",
    "material_quality_level",
    "coating_level",
    "hunting_level",
    "oil_flow_level",
    "oil_supply_temp_level",
    "evacuation_level",
)


@dataclass(frozen=True)
class PrinciplesReducedVector:
    """Named reduced-order decision vector used by the principles region search."""

    values: np.ndarray

    @classmethod
    def from_array(cls, values: np.ndarray | list[float] | tuple[float, ...]) -> "PrinciplesReducedVector":
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size != len(REDUCED_VARIABLE_NAMES):
            raise ValueError(
                f"Reduced vector must have length {len(REDUCED_VARIABLE_NAMES)}, got {arr.size}"
            )
        return cls(values=arr)

    def to_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64).copy()


@dataclass(frozen=True)
class PrinciplesExpansionPolicy:
    """Deterministic policy that expands a reduced vector into the full canonical encoding."""

    pitch_coeff_5: float
    pitch_coeff_6: float
    realworld_defaults: dict[str, float]
    lube_pitch_line_velocity_threshold_m_s: float
    lube_mode_level_min: float
    material_quality_level_min: float

    @classmethod
    def from_profile(cls, profile_payload: dict[str, Any]) -> "PrinciplesExpansionPolicy":
        payload = dict(profile_payload.get("expansion_policy", {}) or {})
        defaults = dict(payload.get("realworld_defaults", {}) or {})
        missing = [name for name in REALWORLD_NAMES if name not in defaults]
        if missing:
            raise ValueError(f"Principles expansion policy missing realworld defaults: {missing}")
        floors = dict(payload.get("realworld_floors", {}) or {})
        return cls(
            pitch_coeff_5=float(payload.get("pitch_coeff_5", 0.0)),
            pitch_coeff_6=float(payload.get("pitch_coeff_6", 0.0)),
            realworld_defaults={str(k): float(v) for k, v in defaults.items()},
            lube_pitch_line_velocity_threshold_m_s=float(
                floors.get("lube_pitch_line_velocity_threshold_m_s", 10.0)
            ),
            lube_mode_level_min=float(floors.get("lube_mode_level_min", 0.67)),
            material_quality_level_min=float(floors.get("material_quality_level_min", 0.5)),
        )


def reduced_bounds() -> tuple[np.ndarray, np.ndarray]:
    xl, xu = bounds()
    idx = np.asarray(REDUCED_FULL_INDICES, dtype=int)
    return xl[idx].copy(), xu[idx].copy()


def reduced_mid_bounds() -> np.ndarray:
    full_mid = np.asarray(mid_bounds_candidate(), dtype=np.float64)
    return full_mid[np.asarray(REDUCED_FULL_INDICES, dtype=int)].copy()


def reduced_seed_states(profile_payload: dict[str, Any]) -> dict[str, np.ndarray]:
    reduced_payload = dict(profile_payload.get("reduced_core", {}) or {})
    seeds = dict(reduced_payload.get("seed_states", {}) or {})
    if not seeds:
        raise ValueError("Principles profile requires non-empty reduced_core.seed_states")
    out: dict[str, np.ndarray] = {}
    xl, xu = reduced_bounds()
    for name, values in seeds.items():
        arr = PrinciplesReducedVector.from_array(values).to_array()
        out[str(name)] = np.clip(arr, xl, xu)
    return out


def reduced_release_stages(profile_payload: dict[str, Any]) -> list[dict[str, Any]]:
    reduced_payload = dict(profile_payload.get("reduced_core", {}) or {})
    stages = list(reduced_payload.get("release_stages", []) or [])
    if not stages:
        raise ValueError("Principles profile requires non-empty reduced_core.release_stages")
    valid_names = set(REDUCED_VARIABLE_NAMES)
    normalized: list[dict[str, Any]] = []
    for rec in stages:
        name = str(rec.get("name", "")).strip()
        variable_names = [str(v) for v in rec.get("variable_names", [])]
        if not name:
            raise ValueError("Principles release stage is missing 'name'")
        if not variable_names:
            raise ValueError(f"Principles release stage '{name}' has no variable_names")
        missing = [v for v in variable_names if v not in valid_names]
        if missing:
            raise ValueError(f"Principles release stage '{name}' uses unknown variables: {missing}")
        normalized.append({"name": name, "variable_names": variable_names})
    return normalized


def reduced_variable_names(profile_payload: dict[str, Any]) -> list[str]:
    reduced_payload = dict(profile_payload.get("reduced_core", {}) or {})
    names = [str(v) for v in reduced_payload.get("variable_names", [])]
    if names != list(REDUCED_VARIABLE_NAMES):
        raise ValueError(
            "Principles reduced_core.variable_names must exactly match the canonical reduced vector ordering"
        )
    return names


def objective_scales_from_profile(profile_payload: dict[str, Any]) -> np.ndarray:
    raw = dict(profile_payload.get("normalization_scales", {}) or {})
    missing = [name for name in PRINCIPLES_OBJECTIVE_NAMES if name not in raw]
    if missing:
        raise ValueError(f"Principles normalization_scales missing objectives: {missing}")
    out = np.asarray([float(raw[name]) for name in PRINCIPLES_OBJECTIVE_NAMES], dtype=np.float64)
    if np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError("Principles normalization_scales must be finite and > 0")
    return out


def weight_vectors_from_profile(profile_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = list(profile_payload.get("weight_vectors", []) or [])
    if not raw:
        raise ValueError("Principles profile requires non-empty weight_vectors")
    out: list[dict[str, Any]] = []
    for rec in raw:
        name = str(rec.get("name", "")).strip()
        weights = np.asarray(rec.get("weights", []), dtype=np.float64).reshape(-1)
        if not name:
            raise ValueError("Principles weight vector missing name")
        if weights.size != len(PRINCIPLES_OBJECTIVE_NAMES):
            raise ValueError(
                f"Principles weight vector '{name}' must have {len(PRINCIPLES_OBJECTIVE_NAMES)} entries"
            )
        if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError(f"Principles weight vector '{name}' must be non-negative with positive sum")
        out.append({"name": name, "weights": weights / np.sum(weights)})
    return out


def expand_reduced_vector(
    reduced: np.ndarray | PrinciplesReducedVector,
    *,
    profile_payload: dict[str, Any],
    rpm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = (
        reduced.to_array()
        if isinstance(reduced, PrinciplesReducedVector)
        else PrinciplesReducedVector.from_array(reduced).to_array()
    )
    policy = PrinciplesExpansionPolicy.from_profile(profile_payload)
    full_mid = np.asarray(mid_bounds_candidate(), dtype=np.float64)
    x_full = np.asarray(full_mid, dtype=np.float64).copy()
    for r_idx, f_idx in enumerate(REDUCED_FULL_INDICES):
        x_full[int(f_idx)] = float(arr[r_idx])

    x_full[_FULL_INDEX_BY_NAME["pitch_coeff_5"]] = float(policy.pitch_coeff_5)
    x_full[_FULL_INDEX_BY_NAME["pitch_coeff_6"]] = float(policy.pitch_coeff_6)
    for name in REALWORLD_NAMES:
        x_full[_FULL_INDEX_BY_NAME[name]] = float(policy.realworld_defaults[name])

    candidate = decode_candidate(x_full)
    pitch_line_velocity_m_s = float(2.0 * np.pi * (candidate.gear.base_radius / 1000.0) * float(rpm) / 60.0)
    overrides: dict[str, float] = {}
    if pitch_line_velocity_m_s >= float(policy.lube_pitch_line_velocity_threshold_m_s):
        idx = _FULL_INDEX_BY_NAME["lube_mode_level"]
        updated = max(float(x_full[idx]), float(policy.lube_mode_level_min))
        x_full[idx] = updated
        overrides["lube_mode_level"] = updated
    mat_idx = _FULL_INDEX_BY_NAME["material_quality_level"]
    updated_mat = max(float(x_full[mat_idx]), float(policy.material_quality_level_min))
    x_full[mat_idx] = updated_mat
    overrides["material_quality_level"] = updated_mat

    xl, xu = bounds()
    x_full = np.clip(x_full, xl, xu)
    return x_full, {
        "pitch_coeff_5": float(policy.pitch_coeff_5),
        "pitch_coeff_6": float(policy.pitch_coeff_6),
        "realworld_defaults": dict(policy.realworld_defaults),
        "deterministic_overrides": overrides,
        "pitch_line_velocity_m_s": pitch_line_velocity_m_s,
        "n_var_full": int(N_TOTAL),
    }


__all__ = [
    "PRINCIPLES_OBJECTIVE_NAMES",
    "PrinciplesExpansionPolicy",
    "PrinciplesReducedVector",
    "REDUCED_FULL_INDICES",
    "REDUCED_VARIABLE_NAMES",
    "expand_reduced_vector",
    "objective_scales_from_profile",
    "reduced_bounds",
    "reduced_mid_bounds",
    "reduced_release_stages",
    "reduced_seed_states",
    "reduced_variable_names",
    "weight_vectors_from_profile",
]
