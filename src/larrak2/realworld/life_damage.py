"""Phase-binned service-life damage accumulation model.

Implements a simplified Miner-rule proxy for 10,000 h gear service life.
Each phase bin accumulates damage based on contact stress, lubricant film
quality (λ), and the pseudo-hunting exposure reduction factor.

Convention:  D_total ≤ 1.0  →  feasible for the target service life.
             D_total > 1.0  →  expected failure before service life.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Wöhler exponent for contact fatigue (typical carburized steel)
_STRESS_EXPONENT = 8.0

# Reference stress for unit damage rate (MPa) — calibrated to AISI 9310 baseline
_SIGMA_REF_MPA = 1500.0

# Lambda influence exponent: lower λ → faster damage accumulation
_LAMBDA_EXPONENT = 2.0

# Pseudo-hunting set cardinality ladder
_HUNTING_LADDER: list[tuple[float, int]] = [
    (0.00, 1),
    (0.15, 2),
    (0.30, 3),
    (0.45, 4),
    (0.65, 6),
    (0.85, 8),
]

# Module-scope cache for limit stress numbers table
_LIMIT_STRESS_CACHE: dict[str, float] | None = None
_BASELINE_ROUTE_ID = "AISI_9310"


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
    """Compute accumulated Miner-style damage for target service life.

    Args:
        hertz_stress_profile: Hertz contact stress per phase bin (MPa).
        lambda_profile: Specific film thickness per phase bin (dimensionless).
        fn_profile: Normal force per phase bin (N), used for force-gating.
        rpm: Operating speed (rev/min).
        hunting_level: Continuous [0,1] pseudo-hunting level.
        service_hours: Target service life (hours), default 10,000.
        sigma_ref_MPa: Optional calibrated reference stress (MPa).

    Returns:
        Dictionary with:
            D_total: Total accumulated damage (≤1.0 = feasible).
            D_ring: Ring-side accumulated damage (includes hunting reduction).
            D_planet: Planet-side accumulated damage (conservative, no reduction).
            N_set: Decoded hunting set count.
            revs_total: Total revolutions in service life.
            sigma_ref_used: Actual sigma_ref used for this computation.
    """
    n_bins = len(hertz_stress_profile)
    if n_bins == 0:
        return {"D_total": 0.0, "D_ring": 0.0, "D_planet": 0.0, "N_set": 1, "revs_total": 0.0}

    N_set = hunting_n_set(hunting_level)
    revs_total = rpm * 60.0 * service_hours

    # Per-bin damage rate:
    #   dD(θ) = (σ_H / σ_ref)^n / max(λ, 0.1)^m
    # Summed over bins per revolution, then multiplied by total revolutions.

    sigma_ref = sigma_ref_MPa if sigma_ref_MPa is not None else _SIGMA_REF_MPA
    sigma_ratio = np.maximum(hertz_stress_profile, 0.0) / sigma_ref
    lambda_clamp = np.maximum(lambda_profile, 0.1)

    dD_per_bin = (sigma_ratio**_STRESS_EXPONENT) / (lambda_clamp**_LAMBDA_EXPONENT)

    # Force-gate: only count bins where normal force > 50% of mean
    force_mean = float(np.mean(np.maximum(fn_profile, 0.0)))
    if force_mean > 0:
        active_mask = fn_profile > 0.5 * force_mean
    else:
        active_mask = np.ones(n_bins, dtype=bool)

    D_per_rev_planet = float(np.sum(dD_per_bin[active_mask])) / n_bins
    D_per_rev_ring = D_per_rev_planet / N_set  # hunting reduces ring exposure

    D_planet = D_per_rev_planet * revs_total
    D_ring = D_per_rev_ring * revs_total

    # Total damage is the maximum of ring and planet (whichever fails first)
    D_total = max(D_planet, D_ring)

    return {
        "D_total": float(D_total),
        "D_ring": float(D_ring),
        "D_planet": float(D_planet),
        "N_set": N_set,
        "revs_total": float(revs_total),
        "sigma_ref_used": float(sigma_ref),
    }


def _load_limit_stress_table() -> dict[str, float]:
    """Load and cache route_id → sigma_Hlim_MPa mapping."""
    global _LIMIT_STRESS_CACHE
    if _LIMIT_STRESS_CACHE is not None:
        return _LIMIT_STRESS_CACHE

    from larrak2.cem.registry import get_registry

    reg = get_registry()
    table = reg.load_table("limit_stress_numbers")

    mapping: dict[str, float] = {}
    if table.get("route_id") and len(table["route_id"]) > 0:
        for i, rid in enumerate(table["route_id"]):
            key = str(rid).strip()
            if key in mapping:
                if os.environ.get("LARRAK_STRICT_DATA", "0") == "1":
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


def get_sigma_ref_for_route(route_id: str, cleanliness_proxy: float = 0.5) -> float:
    """Return calibration-preserving sigma_ref for a material route.

    Scaling rule:
        sigma_ref = _SIGMA_REF_MPA * (sigma_Hlim / sigma_Hlim_baseline) * f_cleanliness

    where sigma_Hlim_baseline is the AISI_9310 value.
    f_cleanliness is a linear interpolation: 0.0 (air melt) → 0.8x, 1.0 (VIM-VAR) → 1.2x.
    Default cleanliness is 0.5 (neutral factor 1.0x).

    Falls back to _SIGMA_REF_MPA if the dataset is empty (placeholder mode)
    and LARRAK_STRICT_DATA is not set.  Raises ValueError in strict mode.
    """
    mapping = _load_limit_stress_table()

    if not mapping:
        if os.environ.get("LARRAK_STRICT_DATA", "0") == "1":
            raise ValueError(
                "LARRAK_STRICT_DATA=1 but limit_stress_numbers dataset is empty. "
                "Cannot resolve sigma_ref for route."
            )
        logger.debug("limit_stress_numbers empty; using baseline _SIGMA_REF_MPA")
        return _SIGMA_REF_MPA

    # Baseline must always exist
    if _BASELINE_ROUTE_ID not in mapping:
        raise ValueError(
            f"Baseline route '{_BASELINE_ROUTE_ID}' missing from "
            f"limit_stress_numbers. Cannot calibrate sigma_ref. "
            f"Available routes: {sorted(mapping.keys())}"
        )

    sigma_hlim_baseline = mapping[_BASELINE_ROUTE_ID]

    if route_id not in mapping:
        if os.environ.get("LARRAK_STRICT_DATA", "0") == "1":
            raise ValueError(
                f"Route '{route_id}' missing from limit_stress_numbers and LARRAK_STRICT_DATA=1."
            )
        logger.warning("Route '%s' not in limit_stress_numbers; using baseline", route_id)
        return _SIGMA_REF_MPA

    sigma_hlim = mapping[route_id]

    # Apply cleanliness scaling (0.0 -> 0.8x, 1.0 -> 1.2x)
    f_cleanliness = 0.8 + 0.4 * float(np.clip(cleanliness_proxy, 0.0, 1.0))

    return _SIGMA_REF_MPA * (sigma_hlim / sigma_hlim_baseline) * f_cleanliness


def invalidate_limit_stress_cache() -> None:
    """Clear the module-scope cache (useful for testing)."""
    global _LIMIT_STRESS_CACHE
    _LIMIT_STRESS_CACHE = None
