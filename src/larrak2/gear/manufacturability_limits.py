"""Manufacturability-derived limits for instantaneous ratio-rate changes.

This module computes conservative, geometry-driven bounds on ratio-profile slope
before gear optimization. Limits are derived from Litvin synthesis feasibility and
simple process-aware manufacturability checks (WEDM/laser style).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.encoding import GearParams
from ..ports.larrak_v1.gear_forward import compute_gear_geometry, litvin_synthesize

N_POINTS = 360
DEFAULT_DURATION_GRID_DEG = np.array([20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0])
RING_RADIUS_MM = 80.0


@dataclass(frozen=True)
class ManufacturingProcessParams:
    """Process-dependent geometric feasibility limits."""

    kerf_mm: float = 0.2
    overcut_mm: float = 0.05
    min_ligament_mm: float = 0.35
    min_feature_radius_mm: float = 0.2
    max_pressure_angle_deg: float = 35.0

    @property
    def min_clearance_mm(self) -> float:
        return float(self.kerf_mm + self.overcut_mm + self.min_ligament_mm)


@dataclass
class RatioRateLimitEnvelope:
    """Envelope of feasible instantaneous ratio-rate limits over durations."""

    duration_deg: np.ndarray
    max_delta_ratio: np.ndarray
    max_ratio_slope: np.ndarray
    process: ManufacturingProcessParams
    metadata: dict[str, Any] = field(default_factory=dict)

    def slope_limit_for_duration(self, duration_deg: float) -> float:
        """Interpolate slope bound for an arbitrary angular duration."""
        duration = float(duration_deg)
        if not np.isfinite(duration) or duration <= 0:
            return 0.0

        x = np.asarray(self.duration_deg, dtype=float)
        y = np.asarray(self.max_ratio_slope, dtype=float)
        if x.size == 0 or y.size == 0:
            return 0.0

        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        return float(np.interp(duration, x_sorted, y_sorted, left=y_sorted[0], right=y_sorted[-1]))


_LIMIT_CACHE: dict[tuple[Any, ...], RatioRateLimitEnvelope] = {}


def _cache_key(
    gear: GearParams,
    process: ManufacturingProcessParams,
    durations_deg: np.ndarray,
    amplitude_scan: np.ndarray,
) -> tuple[Any, ...]:
    return (
        round(float(gear.base_radius), 6),
        tuple(np.round(np.asarray(gear.pitch_coeffs, dtype=float), 6)),
        round(process.kerf_mm, 6),
        round(process.overcut_mm, 6),
        round(process.min_ligament_mm, 6),
        round(process.min_feature_radius_mm, 6),
        round(process.max_pressure_angle_deg, 6),
        tuple(np.round(np.asarray(durations_deg, dtype=float), 6)),
        tuple(np.round(np.asarray(amplitude_scan, dtype=float), 6)),
    )


def _ratio_law_candidate(theta: np.ndarray, duration_deg: float, amplitude: float) -> np.ndarray:
    """Build a smooth candidate ratio law with localized change over a duration."""
    duration_deg = max(float(duration_deg), 1e-3)
    duration_rad = np.deg2rad(duration_deg)

    # half-sine perturbation over [0, duration] around baseline ratio 2.0
    phase = np.mod(theta, 2.0 * np.pi)
    ratio = np.full_like(theta, 2.0, dtype=float)
    active = phase <= duration_rad
    if np.any(active):
        s = np.clip(phase[active] / duration_rad, 0.0, 1.0)
        ratio[active] = ratio[active] + amplitude * np.sin(np.pi * s)

    return np.maximum(ratio, 0.1)


def _manufacturability_check(
    theta: np.ndarray,
    ratio_profile: np.ndarray,
    process: ManufacturingProcessParams,
) -> bool:
    """Conservative manufacturability + Litvin feasibility checks."""
    try:
        ratio_safe = np.maximum(np.asarray(ratio_profile, dtype=float), 1e-6)
        r_planet = RING_RADIUS_MM / ratio_safe

        if not np.all(np.isfinite(r_planet)):
            return False

        synth = litvin_synthesize(theta, r_planet, target_ratio=RING_RADIUS_MM / np.mean(r_planet))
        psi = synth["psi"]
        R_psi = synth["R_psi"]
        rho_c = np.asarray(synth["rho_c"], dtype=float)

        if not (np.all(np.isfinite(psi)) and np.all(np.isfinite(R_psi)) and np.all(np.isfinite(rho_c))):
            return False
        if np.any(np.diff(psi) <= 0.0):
            return False

        thickness = np.asarray(R_psi, dtype=float) - np.asarray(r_planet, dtype=float)
        min_thickness = float(np.min(thickness)) if thickness.size else -np.inf
        if min_thickness < process.min_clearance_mm:
            return False

        min_osculating_radius = float(np.min(np.abs(rho_c))) if rho_c.size else 0.0
        if min_osculating_radius < (process.min_feature_radius_mm + process.kerf_mm * 0.5 + process.overcut_mm):
            return False

        geom = compute_gear_geometry(
            theta,
            r_planet,
            psi,
            R_psi,
            target_average_radius=float(np.mean(R_psi)),
            max_pressure_angle_deg=process.max_pressure_angle_deg,
        )

        if bool(geom.get("interference_flag", True)):
            return False
        if float(geom.get("contact_ratio", 0.0)) < 1.0:
            return False

        return True
    except Exception:
        # Hard-fail closed: any synthesis/geometry issue is infeasible.
        return False


def compute_manufacturable_ratio_rate_limits(
    gear: GearParams,
    process: ManufacturingProcessParams | None = None,
    durations_deg: np.ndarray | None = None,
    amplitude_scan: np.ndarray | None = None,
) -> RatioRateLimitEnvelope:
    """Compute manufacturability-derived ratio-rate envelope.

    Returns conservative bounds to be used as an upstream optimizer constraint.
    """
    process = process or ManufacturingProcessParams()
    theta = np.linspace(0.0, 2.0 * np.pi, N_POINTS, endpoint=False)

    durations = (
        np.asarray(durations_deg, dtype=float)
        if durations_deg is not None
        else DEFAULT_DURATION_GRID_DEG.astype(float)
    )
    durations = durations[np.isfinite(durations) & (durations > 0.0)]
    if durations.size == 0:
        durations = DEFAULT_DURATION_GRID_DEG.astype(float)

    amps = (
        np.asarray(amplitude_scan, dtype=float)
        if amplitude_scan is not None
        else np.linspace(0.0, 1.25, 31)
    )
    amps = amps[np.isfinite(amps) & (amps >= 0.0)]
    if amps.size == 0:
        amps = np.linspace(0.0, 1.25, 31)

    key = _cache_key(gear, process, durations, amps)
    cached = _LIMIT_CACHE.get(key)
    if cached is not None:
        return cached

    feasible_delta = np.zeros_like(durations, dtype=float)
    feasible_slope = np.zeros_like(durations, dtype=float)
    pass_counts = np.zeros_like(durations, dtype=int)

    for i, duration_deg in enumerate(durations):
        best_amp = 0.0
        n_pass = 0
        for amp in amps:
            ratio_profile = _ratio_law_candidate(theta, float(duration_deg), float(amp))
            if _manufacturability_check(theta, ratio_profile, process):
                best_amp = max(best_amp, float(amp))
                n_pass += 1

        duration_rad = np.deg2rad(max(float(duration_deg), 1e-6))
        feasible_delta[i] = best_amp
        feasible_slope[i] = best_amp * np.pi / duration_rad
        pass_counts[i] = n_pass

    envelope = RatioRateLimitEnvelope(
        duration_deg=durations,
        max_delta_ratio=feasible_delta,
        max_ratio_slope=feasible_slope,
        process=process,
        metadata={
            "cached": False,
            "candidate_amplitudes": amps,
            "pass_counts": pass_counts,
        },
    )

    _LIMIT_CACHE[key] = envelope
    return envelope
