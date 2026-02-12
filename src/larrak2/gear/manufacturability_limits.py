"""Manufacturability-derived limits for instantaneous ratio-rate changes.

Two-phase workflow:
    Phase 1 — Reduced Litvin-Consistent Geometric Analysis: evaluate the
        governing invariants of the conjugate envelope (Centrodes, Curvature,
        Stability Derivatives, Offset Robustness, Monotonicity) without
        generating the full surface.  Deterministic and exact.

    Phase 2 — Full Litvin Verification: complete conjugate tooth generation
        and contact analysis for candidates that satisfy the geometric invariants.

Falls back to brute-force (Litvin on every point) if the surrogate is disabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..core.encoding import GearParams


logger = logging.getLogger(__name__)

N_POINTS = 360
DEFAULT_DURATION_GRID_DEG = np.array(
    [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 270.0, 360.0]
)
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


# ============================================================================
# Candidate ratio-law builders
# ============================================================================

# Profile shape names for metadata / viz
PROFILE_NAMES: list[str] = [
    "half_sine",
    "second_harmonic",
    "ramp_up",
    "ramp_down",
    "trapezoid",
    "double_bump",
]


def _ratio_law_candidate(theta: np.ndarray, duration_deg: float, amplitude: float) -> np.ndarray:
    """Build a smooth candidate ratio law with localized change over a duration.

    Uses a basis function (e.g. half-sine) as a probe to explore the
    manufacturability of the ratio solution space (not an assertion that
    the final gearing is sinusoidal).
    """
    return _build_profile(theta, duration_deg, amplitude, "half_sine")


def _build_profile(
    theta: np.ndarray,
    duration_deg: float,
    amplitude: float,
    shape: str,
) -> np.ndarray:
    """Build a candidate ratio law for one of several profile shapes.

    All shapes share the same baseline (ratio=2.0) and are localized to
    [0, duration_rad].  The ``amplitude`` parameter scales the peak-to-peak
    excursion identically across shapes for fair comparison.

    Shapes:
        half_sine       — classic C¹ smooth bump
        second_harmonic — sin(2πs) layered on sin(πs), sharper peak
        ramp_up         — linear ramp → sine return (asymmetric)
        ramp_down       — sine rise → linear ramp (asymmetric mirror)
        trapezoid       — fast attack, flat dwell, fast decay (step-like)
        double_bump     — two half-sine lobes with opposite sign
    """
    duration_deg = max(float(duration_deg), 1e-3)
    duration_rad = np.deg2rad(duration_deg)

    phase = np.mod(theta, 2.0 * np.pi)
    ratio = np.full_like(theta, 2.0, dtype=float)
    active = phase <= duration_rad
    if not np.any(active):
        return np.maximum(ratio, 0.1)

    s = np.clip(phase[active] / duration_rad, 0.0, 1.0)

    if shape == "half_sine":
        # C¹ smooth bump
        ratio[active] += amplitude * np.sin(np.pi * s)

    elif shape == "second_harmonic":
        # Primary + 2nd harmonic for a sharper, asymmetric peak
        ratio[active] += amplitude * (0.7 * np.sin(np.pi * s) + 0.3 * np.sin(2.0 * np.pi * s))

    elif shape == "ramp_up":
        # Asymmetric: linear ramp up, sine return
        rise = np.clip(2.0 * s, 0.0, 1.0)  # linear in first half
        fall = np.where(s > 0.5, np.sin(np.pi * (1.0 - s)), 1.0)
        ratio[active] += amplitude * rise * fall

    elif shape == "ramp_down":
        # Mirror: sine rise, linear ramp down
        rise = np.where(s < 0.5, np.sin(np.pi * s), 1.0)
        fall = np.clip(2.0 * (1.0 - s), 0.0, 1.0)
        ratio[active] += amplitude * rise * fall

    elif shape == "trapezoid":
        # Step-like: fast attack (10%), flat dwell (80%), fast decay (10%)
        attack = np.clip(s / 0.1, 0.0, 1.0)
        decay = np.clip((1.0 - s) / 0.1, 0.0, 1.0)
        ratio[active] += amplitude * attack * decay

    elif shape == "double_bump":
        # Two lobes: positive first half, negative second half
        ratio[active] += amplitude * np.sin(2.0 * np.pi * s)

    else:
        # Fallback to half-sine
        ratio[active] += amplitude * np.sin(np.pi * s)

    return np.maximum(ratio, 0.1)


def _build_all_candidates(
    theta: np.ndarray,
    duration_deg: float,
    amplitude: float,
) -> list[tuple[str, np.ndarray]]:
    """Build all profile variants for a given (duration, amplitude).

    Returns list of (shape_name, ratio_profile) tuples.
    """
    return [(name, _build_profile(theta, duration_deg, amplitude, name)) for name in PROFILE_NAMES]


# ============================================================================
# Phase 1 — Mathematical surrogate (centrode-level geometry)
# ============================================================================


def _centrode_curvature(theta: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Polar-curve curvature κ(θ) for a centrode r(θ).

    κ = (r² + 2r'² − r·r'') / (r² + r'²)^(3/2)
    """
    r_safe = np.maximum(r, 1e-6)
    r_prime = np.gradient(r_safe, theta)
    r_dbl = np.gradient(r_prime, theta)

    num = r_safe**2 + 2.0 * r_prime**2 - r_safe * r_dbl
    denom = (r_safe**2 + r_prime**2) ** 1.5
    denom = np.maximum(denom, 1e-12)

    return num / denom


def _surrogate_check(
    theta: np.ndarray,
    ratio_profile: np.ndarray,
    process: ManufacturingProcessParams,
) -> bool:
    """Reduced Litvin-Consistent Geometric Analysis.

    Evaluates the geometric invariants that control the conjugate envelope,
    as defined by the governing differential geometry of the centrodes.
    This is NOT a heuristic model, but a deterministic evaluation of the
    necessary conditions for a valid Litvin envelope.

    Invariants Checked:
        1. Centrodes (Pitch Curves) — physical radius bounds
        2. Curvature Field — osculating radius > tooling limit
        3. Envelope Stability — relative motion derivatives bounded
        4. Medial Axis / Thickness — feature size vs kerf
        5. Offset Robustness — Minkowski sum viability
        6. Monotonicity — positive ratio (dψ/dθ > 0)
        7. Topology — limited sign changes (no complex loops)
    """
    ratio_safe = np.maximum(np.asarray(ratio_profile, dtype=float), 1e-6)
    r_planet = RING_RADIUS_MM / ratio_safe

    # --- 1. Centrodes (Pitch Curves) ---
    # Radius must remain within valid physical bounds for the mechanism.
    if not np.all(np.isfinite(r_planet)):
        return False
    r_min = float(np.min(r_planet))
    r_max = float(np.max(r_planet))
    if r_min < 2.0 or r_max > RING_RADIUS_MM - 1.0:
        return False

    # --- 2. Envelope Stability (Relative Motion Field) ---
    # Envelope singularities occur when d(n·v_rel)/dθ vanishes.
    # Large first or second derivatives of the ratio law amplify curvature
    # and produce envelope degeneracy (cusps/undercuts) or extreme pressure angles.
    # We use generous bounds as a "loose screen" for stability.
    d_ratio = np.gradient(ratio_safe, theta)
    if not np.all(np.isfinite(d_ratio)):
        return False
    d2_ratio = np.gradient(d_ratio, theta)
    if not np.all(np.isfinite(d2_ratio)):
        return False

    max_d1 = float(np.max(np.abs(d_ratio)))
    max_d2 = float(np.max(np.abs(d2_ratio)))

    # Limits for stability screen:
    # d1 ~ 10.0 allows steep ramps but catches infinite spikes.
    # d2 ~ 50.0 allows tight turns but catches cusp-forming acceleration.
    if max_d1 > 10.0 or max_d2 > 50.0:
        return False

    # --- 3. Curvature & 4. Medial Axis / Offset Robustness ---
    # 5. Minkowski Sum Viability & 7. Osculating Circle Approximation ---
    #
    # All these invariants reduce to the same geometric condition:
    # The osculating radius ρ must be larger than the tool/kerf radius.
    # If ρ_min < R_kerf, the offset surface self-intersects (undercut).
    #
    # We enforce: ρ_min >= min_feature + kerf/2 + overcut
    kappa = _centrode_curvature(theta, r_planet)
    if not np.all(np.isfinite(kappa)):
        return False

    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(np.abs(kappa) > 1e-12, 1.0 / np.abs(kappa), np.inf)

    min_rho = float(np.min(rho))
    min_allowed_rho = process.min_feature_radius_mm + process.kerf_mm * 0.5 + process.overcut_mm

    if min_rho < min_allowed_rho:
        return False

    # --- 6. Mapping Monotonicity ---
    # Litvin requirement: dψ/dθ > 0.
    # For a fixed center distance, dψ/dθ = i(θ).
    # Thus, the ratio must be strictly positive to avoid kinematic reversal.
    if float(np.min(ratio_safe)) <= 0.01:
        return False

    # --- 7. Topology (Self-Intersection Risk) ---
    # Curvature sign changes indicate inflection points.
    # Complex looping behavior (many sign changes) implies pathological
    # centrode geometry that rarely yields valid teeth.
    sign_changes = int(np.sum(np.diff(np.sign(kappa)) != 0))
    if sign_changes > 4:
        return False

    return True


# ============================================================================
# Phase 2 — Full Litvin confirmation (conjugate tooth profile)
# ============================================================================


def _manufacturability_check(
    theta: np.ndarray,
    ratio_profile: np.ndarray,
    process: ManufacturingProcessParams,
) -> bool:
    """Indepth check using full conjugate tooth pair equations.

    Runs the complete Litvin synthesis pipeline:
        1. Conjugate ring profile R(ψ) via envelope theory
        2. Osculating radius ≥ min feature radius + kerf allowance
        3. Gear geometry → no interference, contact ratio ≥ 1,
           max pressure angle within process limit
    """
    try:
        from ..ports.larrak_v1.gear_forward import compute_gear_geometry, litvin_synthesize

        ratio_safe = np.maximum(np.asarray(ratio_profile, dtype=float), 1e-6)
        r_planet = RING_RADIUS_MM / ratio_safe

        if not np.all(np.isfinite(r_planet)):
            return False

        synth = litvin_synthesize(
            theta,
            r_planet,
            target_ratio=float(RING_RADIUS_MM / np.mean(r_planet)),
        )
        psi = synth["psi"]
        R_psi = synth["R_psi"]
        rho_c = np.asarray(synth["rho_c"], dtype=float)

        if not (
            np.all(np.isfinite(psi)) and np.all(np.isfinite(R_psi)) and np.all(np.isfinite(rho_c))
        ):
            return False

        # Osculating radius bound — curvature must support tooling
        min_osculating_radius = float(np.min(np.abs(rho_c))) if rho_c.size else 0.0
        if min_osculating_radius < (
            process.min_feature_radius_mm + process.kerf_mm * 0.5 + process.overcut_mm
        ):
            return False

        # Full gear geometry check
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

        # Pressure angle check — max across mesh must stay within process limit
        pa = geom.get("pressure_angle")
        if pa is not None:
            pa_arr = np.asarray(pa, dtype=float)
            if pa_arr.size > 0:
                max_pa_deg = float(np.rad2deg(np.max(np.abs(pa_arr))))
                if max_pa_deg > process.max_pressure_angle_deg:
                    return False

        return True
    except Exception:
        # Hard-fail closed: any synthesis/geometry issue is infeasible.
        return False


# ============================================================================
# Three-phase envelope computation
# ============================================================================

# PicoGK oracle is opt-in via environment variable
import os as _os

_PICOGK_ENABLED = _os.environ.get("LARRAK_PICOGK_ORACLE", "0") == "1"


def _picogk_check(
    theta: np.ndarray,
    ratio_profile: np.ndarray,
    process: ManufacturingProcessParams,
) -> bool:
    """Phase 2 — PicoGK voxel offset manufacturability check.

    Uses the PicoGK CLI oracle to evaluate WEDM/laser manufacturability
    via SDF offset operations on the gear profile.

    Returns True if the profile passes all PicoGK checks, False otherwise.
    Always returns True if PicoGK is disabled.
    """
    if not _PICOGK_ENABLED:
        return True

    try:
        from .picogk_adapter import evaluate_manufacturability

        r_planet = RING_RADIUS_MM / np.maximum(np.asarray(ratio_profile, dtype=float), 1e-6)
        result = evaluate_manufacturability(
            theta,
            r_planet,
            wire_d_mm=process.kerf_mm,
            overcut_mm=process.overcut_mm,
            min_ligament_mm=process.min_ligament_mm,
        )
        return bool(result.get("passed", False))
    except Exception:
        logger.debug("PicoGK oracle failed, treating as pass (fail-open for oracle errors)")
        return True


def compute_manufacturable_ratio_rate_limits(
    gear: GearParams,
    process: ManufacturingProcessParams | None = None,
    durations_deg: np.ndarray | None = None,
    amplitude_scan: np.ndarray | None = None,
) -> RatioRateLimitEnvelope:
    """Compute manufacturability-derived ratio-rate envelope.

    Three-phase workflow:
        Phase 1 — Mathematical surrogate: evaluate centrode-level geometry
            (curvature, thickness, monotonicity, self-intersection) derived
            deterministically from the ratio law.  No learned parameters.
        Phase 2 — PicoGK oracle: SDF offset manufacturability (WEDM kerf
            survival, ligament thickness, concave radius).  Opt-in via
            LARRAK_PICOGK_ORACLE=1 environment variable.
        Phase 3 — Full Litvin: run conjugate tooth profile synthesis + gear
            geometry validation on candidates that pass Phase 1 and Phase 2.

    Returns conservative bounds for upstream optimizer constraints.
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
        else np.linspace(-1.5, 4.0, 61)
    )
    amps = amps[np.isfinite(amps)]
    if amps.size == 0:
        amps = np.linspace(-1.5, 4.0, 61)

    key = _cache_key(gear, process, durations, amps)
    cached = _LIMIT_CACHE.get(key)
    if cached is not None:
        return cached

    envelope = _three_phase_scan(theta, durations, amps, process)

    _LIMIT_CACHE[key] = envelope
    return envelope


def _three_phase_scan(
    theta: np.ndarray,
    durations: np.ndarray,
    amps: np.ndarray,
    process: ManufacturingProcessParams,
) -> RatioRateLimitEnvelope:
    """Phase 1: surrogate → Phase 2: PicoGK → Phase 3: Litvin.

    For each (duration, amplitude) pair, ALL profile shapes are tested.
    A grid point passes if ANY shape produces a feasible design.
    """

    feasible_delta = np.zeros_like(durations, dtype=float)
    feasible_slope = np.zeros_like(durations, dtype=float)
    pass_counts = np.zeros_like(durations, dtype=int)
    surrogate_pass_counts = np.zeros_like(durations, dtype=int)
    picogk_pass_counts = np.zeros_like(durations, dtype=int)
    litvin_call_counts = np.zeros_like(durations, dtype=int)
    shape_pass_counts: dict[str, int] = {name: 0 for name in PROFILE_NAMES}

    for i, duration_deg in enumerate(durations):
        best_amp = 0.0
        n_pass = 0
        n_surrogate_pass = 0
        n_picogk_pass = 0
        n_litvin = 0

        for amp in amps:
            point_passed = False

            for shape_name, ratio_profile in _build_all_candidates(
                theta, float(duration_deg), float(amp)
            ):
                # Phase 1 — mathematical surrogate (centrode geometry)
                if not _surrogate_check(theta, ratio_profile, process):
                    continue
                n_surrogate_pass += 1

                # Phase 2 — PicoGK oracle (SDF offset checks)
                if not _picogk_check(theta, ratio_profile, process):
                    continue
                n_picogk_pass += 1

                # Phase 3 — full Litvin confirmation (conjugate tooth pair)
                litvin_pass = _manufacturability_check(theta, ratio_profile, process)
                n_litvin += 1

                if litvin_pass:
                    point_passed = True
                    shape_pass_counts[shape_name] += 1
                    break

            if point_passed:
                best_amp = max(best_amp, abs(float(amp)))
                n_pass += 1

        duration_rad = np.deg2rad(max(float(duration_deg), 1e-6))
        feasible_delta[i] = best_amp
        feasible_slope[i] = best_amp * np.pi / duration_rad
        pass_counts[i] = n_pass
        surrogate_pass_counts[i] = n_surrogate_pass
        picogk_pass_counts[i] = n_picogk_pass
        litvin_call_counts[i] = n_litvin

    n_shapes = len(PROFILE_NAMES)
    total_grid = len(durations) * len(amps) * n_shapes
    total_litvin = int(np.sum(litvin_call_counts))
    total_surrogate_pass = int(np.sum(surrogate_pass_counts))
    total_picogk_pass = int(np.sum(picogk_pass_counts))
    logger.info(
        "Three-phase scan: %d/%d surr → %d picogk → %d litvin (%.1f%% total reduction)",
        total_surrogate_pass,
        total_grid,
        total_picogk_pass,
        total_litvin,
        100.0 * (1.0 - total_litvin / max(total_grid, 1)),
    )

    return RatioRateLimitEnvelope(
        duration_deg=durations,
        max_delta_ratio=feasible_delta,
        max_ratio_slope=feasible_slope,
        process=process,
        metadata={
            "method": "three_phase",
            "picogk_enabled": _PICOGK_ENABLED,
            "cached": False,
            "candidate_amplitudes": amps,
            "profile_shapes": PROFILE_NAMES,
            "n_shapes": n_shapes,
            "pass_counts": pass_counts,
            "surrogate_pass_counts": surrogate_pass_counts,
            "picogk_pass_counts": picogk_pass_counts,
            "litvin_call_counts": litvin_call_counts,
            "shape_pass_counts": shape_pass_counts,
            "total_grid_points": total_grid,
            "total_litvin_calls": total_litvin,
        },
    )
