"""Canonical valve-timing profile loader and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

THERMO_TIMING_VAR_NAMES: tuple[str, ...] = (
    "intake_open_offset_from_bdc",
    "intake_duration_deg",
    "exhaust_open_offset_from_expansion_tdc",
    "exhaust_duration_deg",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_THERMO_TIMING_PROFILE_PATH = (
    _repo_root() / "data" / "thermo" / "valve_timing_profile_v1.json"
)


@dataclass(frozen=True)
class ThermoTimingProfile:
    profile_id: str
    profile_version: str
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    legacy_defaults: np.ndarray
    hard_thresholds: dict[str, float]
    source_references: tuple[str, ...]
    path: str


def _validate_payload(path: Path, payload: dict[str, Any]) -> ThermoTimingProfile:
    required_top = ("profile_id", "profile_version", "timing_bounds", "legacy_defaults", "hard_thresholds")
    missing_top = [name for name in required_top if name not in payload]
    if missing_top:
        raise ValueError(f"Thermo timing profile missing required keys: {missing_top}")

    timing_bounds = dict(payload.get("timing_bounds", {}) or {})
    missing_bounds = [name for name in THERMO_TIMING_VAR_NAMES if name not in timing_bounds]
    if missing_bounds:
        raise ValueError(f"Thermo timing profile missing bounds for: {missing_bounds}")

    legacy_defaults = dict(payload.get("legacy_defaults", {}) or {})
    missing_defaults = [name for name in THERMO_TIMING_VAR_NAMES if name not in legacy_defaults]
    if missing_defaults:
        raise ValueError(f"Thermo timing profile missing legacy defaults for: {missing_defaults}")

    lower = []
    upper = []
    for name in THERMO_TIMING_VAR_NAMES:
        rec = dict(timing_bounds.get(name, {}) or {})
        lo = float(rec.get("lower", np.nan))
        hi = float(rec.get("upper", np.nan))
        if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
            raise ValueError(
                f"Thermo timing profile invalid bounds for '{name}': lower={lo}, upper={hi}"
            )
        default = float(legacy_defaults[name])
        if default < lo or default > hi:
            raise ValueError(
                f"Thermo timing default for '{name}' must be within bounds, got {default}"
            )
        lower.append(lo)
        upper.append(hi)

    thresholds = {str(k): float(v) for k, v in dict(payload.get("hard_thresholds", {}) or {}).items()}
    threshold_keys = (
        "burn_cap_min",
        "trapped_mass_min_kg",
        "scavenging_efficiency_min",
        "residual_fraction_max",
    )
    missing_thresholds = [name for name in threshold_keys if name not in thresholds]
    if missing_thresholds:
        raise ValueError(
            f"Thermo timing profile missing stable-combustion thresholds: {missing_thresholds}"
        )
    if not 0.0 <= thresholds["burn_cap_min"] <= 1.0:
        raise ValueError("burn_cap_min must be within [0, 1]")
    if thresholds["trapped_mass_min_kg"] <= 0.0:
        raise ValueError("trapped_mass_min_kg must be > 0")
    if not 0.0 <= thresholds["scavenging_efficiency_min"] <= 1.0:
        raise ValueError("scavenging_efficiency_min must be within [0, 1]")
    if not 0.0 <= thresholds["residual_fraction_max"] <= 1.0:
        raise ValueError("residual_fraction_max must be within [0, 1]")

    refs = tuple(str(v) for v in list(payload.get("source_references", []) or []))
    if not refs:
        raise ValueError("Thermo timing profile requires non-empty source_references")

    return ThermoTimingProfile(
        profile_id=str(payload["profile_id"]),
        profile_version=str(payload["profile_version"]),
        bounds_lower=np.asarray(lower, dtype=np.float64),
        bounds_upper=np.asarray(upper, dtype=np.float64),
        legacy_defaults=np.asarray(
            [float(legacy_defaults[name]) for name in THERMO_TIMING_VAR_NAMES],
            dtype=np.float64,
        ),
        hard_thresholds=thresholds,
        source_references=refs,
        path=str(path),
    )


@lru_cache(maxsize=8)
def load_thermo_timing_profile(path: str | Path | None = None) -> ThermoTimingProfile:
    raw = DEFAULT_THERMO_TIMING_PROFILE_PATH if path is None else Path(path)
    profile_path = raw if raw.is_absolute() else (_repo_root() / raw).resolve()
    if not profile_path.exists():
        raise FileNotFoundError(f"Thermo timing profile not found: {profile_path}")
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    return _validate_payload(profile_path, payload)


def thermo_timing_bounds(
    profile_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    profile = load_thermo_timing_profile(profile_path)
    return profile.bounds_lower.copy(), profile.bounds_upper.copy()


def legacy_timing_defaults(profile_path: str | Path | None = None) -> np.ndarray:
    profile = load_thermo_timing_profile(profile_path)
    return profile.legacy_defaults.copy()


def stable_combustion_thresholds(profile_path: str | Path | None = None) -> dict[str, float]:
    profile = load_thermo_timing_profile(profile_path)
    return dict(profile.hard_thresholds)

