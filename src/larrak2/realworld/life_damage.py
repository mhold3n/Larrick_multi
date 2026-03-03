"""Phase-binned service-life damage accumulation model."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CALIBRATION_PATH = Path("data/cem/life_damage_calibration_v1.json")
_CALIBRATION_CACHE: dict[str, Any] | None = None
_EMITTED_STRICT_DATA_DEPRECATION = False
_SIGMA_REF_MPA = 1500.0  # Backward-compatible alias; synchronized from calibration file.
_BASELINE_ROUTE_ID = "AISI_9310"  # Backward-compatible alias; synchronized from calibration file.

# Module-scope cache for limit stress numbers table
_LIMIT_STRESS_CACHE: dict[str, float] | None = None

# Pseudo-hunting set cardinality ladder
_HUNTING_LADDER: list[tuple[float, int]] = [
    (0.00, 1),
    (0.15, 2),
    (0.30, 3),
    (0.45, 4),
    (0.65, 6),
    (0.85, 8),
]


def _load_calibration(path: Path = _CALIBRATION_PATH) -> dict[str, Any]:
    global _BASELINE_ROUTE_ID, _CALIBRATION_CACHE, _SIGMA_REF_MPA
    if _CALIBRATION_CACHE is not None:
        return _CALIBRATION_CACHE

    if not path.exists():
        raise FileNotFoundError(f"Life-damage calibration file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Life-damage calibration file must be an object: {path}")

    required = {
        "version",
        "baseline_route_id",
        "sigma_ref_mpa",
        "stress_exponent",
        "lambda_exponent",
        "cleanliness_scale_min",
        "cleanliness_scale_max",
    }
    missing = sorted(k for k in required if k not in payload)
    if missing:
        raise ValueError(f"Life-damage calibration missing required fields: {missing}")

    calib = {
        "version": str(payload["version"]),
        "baseline_route_id": str(payload["baseline_route_id"]),
        "sigma_ref_mpa": float(payload["sigma_ref_mpa"]),
        "stress_exponent": float(payload["stress_exponent"]),
        "lambda_exponent": float(payload["lambda_exponent"]),
        "cleanliness_scale_min": float(payload["cleanliness_scale_min"]),
        "cleanliness_scale_max": float(payload["cleanliness_scale_max"]),
    }
    if calib["sigma_ref_mpa"] <= 0:
        raise ValueError("sigma_ref_mpa must be > 0 in life-damage calibration")
    if calib["stress_exponent"] <= 0 or calib["lambda_exponent"] <= 0:
        raise ValueError("stress_exponent/lambda_exponent must be > 0 in life-damage calibration")
    if calib["cleanliness_scale_max"] < calib["cleanliness_scale_min"]:
        raise ValueError("cleanliness scale bounds are invalid in life-damage calibration")

    _CALIBRATION_CACHE = calib
    _SIGMA_REF_MPA = float(calib["sigma_ref_mpa"])
    _BASELINE_ROUTE_ID = str(calib["baseline_route_id"])
    return _CALIBRATION_CACHE


def _strict_data_enabled(strict_data: bool | None) -> bool:
    global _EMITTED_STRICT_DATA_DEPRECATION
    if strict_data is not None:
        return bool(strict_data)

    env_raw = str(os.environ.get("LARRAK_STRICT_DATA", "")).strip()
    if not env_raw:
        return False

    if not _EMITTED_STRICT_DATA_DEPRECATION:
        logger.warning(
            "LARRAK_STRICT_DATA env fallback is deprecated. Set EvalContext.strict_data instead."
        )
        _EMITTED_STRICT_DATA_DEPRECATION = True
    return env_raw == "1"


def hunting_n_set(level: float) -> int:
    """Map continuous hunting_level (0–1) to discrete tooth-set count N_set."""
    level = float(np.clip(level, 0.0, 1.0))
    best = 1
    for threshold, n in _HUNTING_LADDER:
        if level >= threshold:
            best = n
    return best


def compute_life_damage_10k(
    hertz_stress_profile: np.ndarray,
    lambda_profile: np.ndarray,
    fn_profile: np.ndarray,
    rpm: float,
    hunting_level: float,
    service_hours: float = 10_000.0,
    sigma_ref_MPa: float | None = None,
) -> dict:
    """Compute accumulated Miner-style damage for target service life."""
    calib = _load_calibration()
    stress_exponent = float(calib["stress_exponent"])
    lambda_exponent = float(calib["lambda_exponent"])
    sigma_ref_default = float(calib["sigma_ref_mpa"])

    n_bins = len(hertz_stress_profile)
    if n_bins == 0:
        return {"D_total": 0.0, "D_ring": 0.0, "D_planet": 0.0, "N_set": 1, "revs_total": 0.0}

    N_set = hunting_n_set(hunting_level)
    revs_total = rpm * 60.0 * service_hours

    sigma_ref = sigma_ref_MPa if sigma_ref_MPa is not None else sigma_ref_default
    sigma_ratio = np.maximum(hertz_stress_profile, 0.0) / sigma_ref
    lambda_clamp = np.maximum(lambda_profile, 0.1)
    dD_per_bin = (sigma_ratio**stress_exponent) / (lambda_clamp**lambda_exponent)

    force_mean = float(np.mean(np.maximum(fn_profile, 0.0)))
    active_mask = fn_profile > 0.5 * force_mean if force_mean > 0 else np.ones(n_bins, dtype=bool)

    D_per_rev_planet = float(np.sum(dD_per_bin[active_mask])) / n_bins
    D_per_rev_ring = D_per_rev_planet / N_set

    D_planet = D_per_rev_planet * revs_total
    D_ring = D_per_rev_ring * revs_total
    D_total = max(D_planet, D_ring)

    return {
        "D_total": float(D_total),
        "D_ring": float(D_ring),
        "D_planet": float(D_planet),
        "N_set": N_set,
        "revs_total": float(revs_total),
        "sigma_ref_used": float(sigma_ref),
        "calibration_version": str(calib["version"]),
    }


def _load_limit_stress_table(*, strict_data: bool | None = None) -> dict[str, float]:
    """Load and cache route_id → sigma_Hlim_MPa mapping."""
    global _LIMIT_STRESS_CACHE
    if _LIMIT_STRESS_CACHE is not None:
        return _LIMIT_STRESS_CACHE

    strict = _strict_data_enabled(strict_data)

    from larrak2.cem.registry import get_registry

    reg = get_registry()
    table = reg.load_table("limit_stress_numbers")

    mapping: dict[str, float] = {}
    if table.get("route_id") and len(table["route_id"]) > 0:
        for i, rid in enumerate(table["route_id"]):
            key = str(rid).strip()
            if key in mapping:
                if strict:
                    raise ValueError(
                        f"Duplicate route_id '{key}' in limit_stress_numbers. "
                        f"Allowables tables must have unique route_ids."
                    )
                logger.warning(
                    "Duplicate route_id '%s' in limit_stress_numbers — last value wins",
                    key,
                )
            mapping[key] = float(table["sigma_Hlim_MPa"][i])

    _LIMIT_STRESS_CACHE = mapping
    return _LIMIT_STRESS_CACHE


def get_sigma_ref_for_route(
    route_id: str,
    cleanliness_proxy: float = 0.5,
    *,
    strict_data: bool | None = None,
) -> float:
    """Return calibration-preserving sigma_ref for a material route."""
    strict = _strict_data_enabled(strict_data)
    calib = _load_calibration()
    baseline_route_id = str(calib["baseline_route_id"])
    sigma_ref_default = float(calib["sigma_ref_mpa"])

    mapping = _load_limit_stress_table(strict_data=strict)
    if not mapping:
        if strict:
            raise ValueError(
                "strict_data=True but limit_stress_numbers dataset is empty. "
                "Cannot resolve sigma_ref for route."
            )
        logger.debug("limit_stress_numbers empty; using baseline sigma_ref")
        return sigma_ref_default

    if baseline_route_id not in mapping:
        raise ValueError(
            f"Baseline route '{baseline_route_id}' missing from limit_stress_numbers. "
            f"Available routes: {sorted(mapping.keys())}"
        )

    sigma_hlim_baseline = mapping[baseline_route_id]
    if route_id not in mapping:
        if strict:
            raise ValueError(
                f"Route '{route_id}' missing from limit_stress_numbers and strict_data=True."
            )
        logger.warning("Route '%s' not in limit_stress_numbers; using baseline", route_id)
        return sigma_ref_default

    sigma_hlim = mapping[route_id]
    cmin = float(calib["cleanliness_scale_min"])
    cmax = float(calib["cleanliness_scale_max"])
    f_cleanliness = cmin + (cmax - cmin) * float(np.clip(cleanliness_proxy, 0.0, 1.0))
    return sigma_ref_default * (sigma_hlim / sigma_hlim_baseline) * f_cleanliness


def invalidate_limit_stress_cache() -> None:
    """Clear module-scope caches (useful for testing)."""
    global _LIMIT_STRESS_CACHE, _CALIBRATION_CACHE
    _LIMIT_STRESS_CACHE = None
    _CALIBRATION_CACHE = None
