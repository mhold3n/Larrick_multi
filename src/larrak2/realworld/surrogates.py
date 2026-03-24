"""Surrogate decision representations for the optimizer.

Maps continuous 0–1 levels to CEM enum tiers and produces lightweight
feasibility estimates without calling the full CEM evaluation.

Design rationale: scalar placeholder methods for material/surface/lube
decisions produce poor optimization signal.  Instead, continuous levels
let the optimizer explore the *ordering* of importance (e.g., "this
design needs jets more than it needs superfinish") and the *feasibility
range* (e.g., "λ requires at least splash-bath level oil delivery").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from larrak2.cem.lubrication import (
    LubricationParams,
    cooling_effectiveness,
    effective_viscosity,
    mode_from_level,
)
from larrak2.cem.material_db import MaterialClass, get_material
from larrak2.cem.material_snapping import get_soft_selected_routes
from larrak2.cem.post_processing import (
    apply_coating_modifiers,
    coating_from_level,
    get_coating,
)
from larrak2.cem.surface_finish import (
    effective_composite_roughness,
    get_finish_properties,
    tier_from_level,
)
from larrak2.cem.tribology import (
    TribologyParams,
    classify_regime,
    compute_micropitting_safety,
    evaluate_tribology,
)


@dataclass
class RealWorldSurrogateParams:
    """Continuous-level surrogate parameters for the optimization loop.

    All levels are in [0, 1] and map to CEM enum tiers:
        0.0 = minimum quality / cheapest option
        1.0 = maximum quality / most expensive option

    Attributes:
        surface_finish_level: 0→AS_GROUND, 0.5→FINE_GROUND, 1→SUPERFINISHED
        lube_mode_level: 0→DRY, 0.33→SPLASH, 0.67→JET, 1→PHASE_GATED_JET
        material_quality_level: 0→9310, 0.5→CBS50NiL, 1→M50NiL
        coating_level: 0→NONE, 0.5→ta-C, 1→W-DLC/CrN duplex
        hunting_level: 0→1-set, 0.5→4-set, 1→8-set pseudo-hunting
        oil_flow_level: 0→0.5 L/min, 1→10 L/min
        oil_supply_temp_level: 0→40°C, 1→120°C
        evacuation_level: 0→passive drain, 1→active scavenge
        material_state: Optional 4D properties [case_HRC, core_KIC, temp_C, clean]
    """

    surface_finish_level: float = 0.7
    lube_mode_level: float = 0.7
    material_quality_level: float | None = None
    coating_level: float = 0.0
    hunting_level: float = 0.0
    oil_flow_level: float = 0.5
    oil_supply_temp_level: float = 0.5
    evacuation_level: float = 0.5
    material_state: np.ndarray | None = None


# Sensible defaults: mid-tier options that represent "generic good practice"
DEFAULT_REALWORLD_PARAMS = RealWorldSurrogateParams()


# Material level → MaterialClass mapping (ordered by quality)
_MATERIAL_LADDER: list[tuple[float, MaterialClass]] = [
    (0.0, MaterialClass.AISI_9310),
    (0.25, MaterialClass.PYROWEAR_53),
    (0.50, MaterialClass.CBS50_NIL),
    (0.75, MaterialClass.M50_NIL),
    (1.0, MaterialClass.FERRIUM_C64),
]


def _material_from_level(level: float | None) -> MaterialClass:
    """Map continuous level (0–1) to material class."""
    if level is None:
        level = 0.5
    level = float(np.clip(level, 0.0, 1.0))
    best = _MATERIAL_LADDER[0][1]
    for threshold, mat in _MATERIAL_LADDER:
        if level >= threshold:
            best = mat
    return best


def _resolve_tribology_validation_mode(
    *,
    tribology_validation_mode: str,
    strict_tribology_data: bool | None,
) -> str:
    mode = str(tribology_validation_mode).strip().lower()
    if strict_tribology_data is True:
        return "strict"
    if strict_tribology_data is False:
        return mode if mode in {"warn", "off"} else "warn"
    return mode if mode in {"strict", "warn", "off"} else "strict"


def _tribology_identifiers(params: RealWorldSurrogateParams) -> tuple[str, str]:
    oil_type = (
        "high_ep"
        if (params.lube_mode_level >= 0.67 or params.oil_flow_level >= 0.70)
        else "generic_ep"
    )
    additive_package = "high_ep" if oil_type == "high_ep" else "standard_ep"
    return oil_type, additive_package


def _merge_data_status(statuses: list[str]) -> str:
    normalized = [str(s).strip().lower() for s in statuses if str(s).strip()]
    if not normalized:
        return "ok"
    if any("degraded_off" in s for s in normalized):
        return "degraded_off"
    if any("degraded_warn" in s for s in normalized):
        return "degraded_warn"
    return "ok"


@dataclass
class RealWorldSurrogateResult:
    """Output of the surrogate evaluation.

    Contains feasibility metrics and a feature-importance ranking
    that communicates what changes matter most for this design.
    """

    lambda_min: float
    scuff_margin_flash_C: float
    scuff_margin_integral_C: float
    scuff_margin_C: float
    micropitting_safety: float
    lube_regime: str
    tribology_method_used: str
    tribology_data_status: str
    material_temp_margin_C: float
    total_cost_index: float
    feature_importance: list[tuple[str, float]]
    min_snap_distance: float = 0.0
    tribology_data_messages: tuple[str, ...] = ()
    tribology_provenance: dict[str, dict[str, str]] = field(default_factory=dict)


def evaluate_realworld_surrogates(
    params: RealWorldSurrogateParams,
    operating_temp_C: float = 200.0,
    hertz_stress_MPa: float = 1200.0,
    sliding_velocity_m_s: float = 5.0,
    entrainment_velocity_m_s: float = 15.0,
    pitch_line_vel_m_s: float = 20.0,
    tribology_scuff_method: str = "auto",
    tribology_validation_mode: str = "strict",
    strict_tribology_data: bool | None = None,
) -> RealWorldSurrogateResult:
    """Evaluate real-world surrogates for the optimization loop.

    This is a lightweight approximation of the full CEM, designed to run
    fast enough for every pymoo iteration.

    Args:
        params: Continuous-level surrogate parameters.
        operating_temp_C: Bulk gear temperature (°C).
        hertz_stress_MPa: Representative Hertzian contact stress.
        sliding_velocity_m_s: Representative sliding velocity.
        entrainment_velocity_m_s: Representative entrainment velocity.
        pitch_line_vel_m_s: Pitch-line velocity for lube calculations.

    Returns:
        RealWorldSurrogateResult with all metrics and importance ranking.
    """
    # 1. Decode levels → CEM tiers
    finish_tier = tier_from_level(params.surface_finish_level)
    lube_mode = mode_from_level(params.lube_mode_level)
    coating = coating_from_level(params.coating_level)

    # 2. Properties from CEM modules (fast lookups, no heavy computation)
    finish_props = get_finish_properties(finish_tier)
    coating_props = get_coating(coating)

    # Composite roughness
    composite_roughness = effective_composite_roughness(finish_tier)

    # Lubrication viscosity (using oil params from decision vector)
    flow_rate = 0.5 + params.oil_flow_level * 9.5  # 0.5 to 10.0 L/min
    supply_temp = 40.0 + params.oil_supply_temp_level * 80.0  # 40 to 120 °C
    lube_params = LubricationParams(
        mode=lube_mode,
        supply_temp_C=supply_temp,
        flow_rate_L_min=flow_rate,
    )
    visc = effective_viscosity(lube_params, operating_temp_C)
    cool_eff = cooling_effectiveness(lube_params, pitch_line_vel_m_s)

    # Evacuation level modifies cooling effectiveness (active scavenge improves it)
    evac_boost = 1.0 + 0.3 * params.evacuation_level  # up to 30% boost
    cool_eff = float(np.clip(cool_eff * evac_boost, 0.0, 1.0))

    scuff_mult, _ = apply_coating_modifiers(1.0, 0.06, coating)
    oil_type, additive_package = _tribology_identifiers(params)
    effective_bulk_temp_C = max(20.0, float(operating_temp_C) - 35.0 * float(cool_eff))
    validation_mode = _resolve_tribology_validation_mode(
        tribology_validation_mode=tribology_validation_mode,
        strict_tribology_data=strict_tribology_data,
    )

    # 3. Tribology (data-backed)
    trib_eval = evaluate_tribology(
        TribologyParams(
            hertz_stress_MPa=hertz_stress_MPa,
            sliding_velocity_m_s=sliding_velocity_m_s,
            entrainment_velocity_m_s=entrainment_velocity_m_s,
            oil_viscosity_cSt=visc,
            composite_roughness_um=composite_roughness,
            bulk_temp_C=effective_bulk_temp_C,
            oil_inlet_temp_C=lube_params.supply_temp_C,
            oil_type=oil_type,
            additive_package=additive_package,
            finish_tier=finish_tier.value,
        ),
        scuff_method=tribology_scuff_method,
        validation_mode=validation_mode,
    )

    lambda_min = float(trib_eval.lambda_min)
    scuff_margin_flash = float(trib_eval.scuff_margin_flash_C * scuff_mult)
    scuff_margin_integral = float(trib_eval.scuff_margin_integral_C * scuff_mult)
    if str(tribology_scuff_method).strip().lower() == "flash":
        scuff_margin = scuff_margin_flash
        scuff_method_used = "flash"
    elif str(tribology_scuff_method).strip().lower() == "integral":
        scuff_margin = scuff_margin_integral
        scuff_method_used = "integral"
    else:
        scuff_margin = min(scuff_margin_flash, scuff_margin_integral)
        scuff_method_used = "flash" if scuff_margin_flash <= scuff_margin_integral else "integral"

    micropitting_sf = compute_micropitting_safety(lambda_min, lambda_perm=trib_eval.lambda_perm)
    regime = classify_regime(lambda_min)  # noqa: F841

    # --- Soft Material Selection & Safe Constraint Aggregation ---
    if params.material_state is not None:
        min_snap_dist, routes_weights = get_soft_selected_routes(
            params.material_state, operating_temp_C
        )
    else:
        # Legacy fallback
        mat_class = _material_from_level(params.material_quality_level)
        min_snap_dist, routes_weights = 0.0, [(mat_class.value, 1.0)]

    _lambda_mins = []
    _scuff_margins = []
    _micropitting_sfs = []
    _temp_margins = []
    _costs = []
    _rankings = []

    for rid, alpha in routes_weights:
        # Evaluate for this specific physical route
        mat_props = get_material(MaterialClass(rid))

        material_temp_margin = mat_props.max_service_temp_C - operating_temp_C
        total_cost = mat_props.cost_tier + finish_props.cost_multiplier + coating_props.cost_tier

        # Feature importance per route
        importance: dict[str, float] = {}
        importance["lubrication"] = max(0.0, (1.0 - lambda_min) * 10.0) if lambda_min < 1.5 else 0.3
        importance["surface_finish"] = max(0.0, 3.0 - lambda_min * 2.0) if lambda_min < 2.0 else 0.2
        importance["material"] = (
            max(0.0, (50.0 - material_temp_margin) / 10.0) if material_temp_margin < 80.0 else 0.3
        )
        importance["coating"] = (
            max(0.0, (100.0 - scuff_margin) / 20.0) if scuff_margin < 150.0 else 0.2
        )
        ranking = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)

        _lambda_mins.append(lambda_min)
        _scuff_margins.append(scuff_margin)
        _micropitting_sfs.append(micropitting_sf)
        _temp_margins.append(material_temp_margin)
        _costs.append(alpha * total_cost)
        if alpha == max(w for _, w in routes_weights):
            _rankings = ranking  # Use ranking from dominant route

    safe_lambda_min = min(_lambda_mins)
    safe_scuff_margin = min(_scuff_margins)
    safe_micropitting = min(_micropitting_sfs)
    safe_temp_margin = min(_temp_margins)
    expected_cost = sum(_costs)
    safe_regime = classify_regime(safe_lambda_min)

    return RealWorldSurrogateResult(
        lambda_min=safe_lambda_min,
        scuff_margin_flash_C=scuff_margin_flash,
        scuff_margin_integral_C=scuff_margin_integral,
        scuff_margin_C=safe_scuff_margin,
        micropitting_safety=safe_micropitting,
        lube_regime=safe_regime.value,
        tribology_method_used=scuff_method_used,
        tribology_data_status=str(trib_eval.tribology_data_status),
        material_temp_margin_C=safe_temp_margin,
        total_cost_index=expected_cost,
        feature_importance=_rankings,
        min_snap_distance=min_snap_dist,
        tribology_data_messages=tuple(trib_eval.tribology_data_messages),
        tribology_provenance=dict(trib_eval.tribology_provenance),
    )


@dataclass
class PhaseResolvedResult:
    """Output of phase-resolved tribology evaluation.

    Only high-load bins are analyzed — where F_n > force_threshold.
    """

    lambda_min: float
    scuff_margin_flash_C: float
    scuff_margin_integral_C: float
    scuff_margin_C: float
    micropitting_safety: float
    lube_regime: str
    tribology_method_used: str
    tribology_data_status: str
    material_temp_margin_C: float
    total_cost_index: float
    feature_importance: list[tuple[str, float]]
    # Phase diagnostics
    worst_phase_deg: float
    n_bins_analyzed: int
    force_threshold_N: float
    lambda_profile: np.ndarray  # Full 360-point λ(θ) — NaN for non-analyzed bins
    min_snap_distance: float = 0.0
    tribology_data_messages: tuple[str, ...] = ()
    tribology_provenance: dict[str, dict[str, str]] = field(default_factory=dict)


def evaluate_realworld_phase_resolved(
    params: RealWorldSurrogateParams,
    hertz_stress_profile: np.ndarray,
    sliding_velocity_profile: np.ndarray,
    entrainment_velocity_profile: np.ndarray,
    fn_profile: np.ndarray,
    operating_temp_C: float = 200.0,
    pitch_line_vel_m_s: float = 20.0,
    force_threshold_fraction: float = 0.8,
    tribology_scuff_method: str = "auto",
    tribology_validation_mode: str = "strict",
    strict_tribology_data: bool | None = None,
) -> PhaseResolvedResult:
    """Phase-resolved tribology: only analyze high-load bins.

    High-load sections are determined by the motion law's force profile.
    Only bins where F_n > fraction * F_n_mean are evaluated for tribology.
    Assumes constant worst-case oiling (jets are fixed geometry).

    Args:
        params: Continuous-level surrogate parameters.
        hertz_stress_profile: 360-point Hertz stress array (MPa).
        sliding_velocity_profile: 360-point sliding velocity (m/s).
        entrainment_velocity_profile: 360-point entrainment velocity (m/s).
        fn_profile: 360-point normal force array (N) for gating.
        operating_temp_C: Bulk gear temperature (°C).
        pitch_line_vel_m_s: For lubrication cooling calculation.
        force_threshold_fraction: Fraction of mean F_n above which to evaluate.

    Returns:
        PhaseResolvedResult with worst-case metrics from high-load bins.
    """
    n_bins = len(hertz_stress_profile)

    # 1. Decode levels → CEM tiers (same as scalar version)
    finish_tier = tier_from_level(params.surface_finish_level)
    lube_mode = mode_from_level(params.lube_mode_level)
    coating = coating_from_level(params.coating_level)

    finish_props = get_finish_properties(finish_tier)
    coating_props = get_coating(coating)

    composite_roughness = effective_composite_roughness(finish_tier)

    lube_params = LubricationParams(mode=lube_mode)
    visc = effective_viscosity(lube_params, operating_temp_C)
    cool_eff = cooling_effectiveness(lube_params, pitch_line_vel_m_s)
    evac_boost = 1.0 + 0.3 * params.evacuation_level
    cool_eff = float(np.clip(cool_eff * evac_boost, 0.0, 1.0))
    effective_bulk_temp_C = max(20.0, float(operating_temp_C) - 35.0 * float(cool_eff))
    scuff_mult, _ = apply_coating_modifiers(1.0, 0.06, coating)
    oil_type, additive_package = _tribology_identifiers(params)
    validation_mode = _resolve_tribology_validation_mode(
        tribology_validation_mode=tribology_validation_mode,
        strict_tribology_data=strict_tribology_data,
    )
    method = str(tribology_scuff_method).strip().lower()

    # 2. Force-gate: identify high-load bins
    fn_mean = float(np.mean(np.abs(fn_profile)))
    force_threshold = force_threshold_fraction * fn_mean
    high_load_mask = np.abs(fn_profile) > force_threshold

    n_analyzed = int(np.sum(high_load_mask))
    if n_analyzed == 0:
        # Fallback: analyze all bins if no bin exceeds threshold
        high_load_mask = np.ones(n_bins, dtype=bool)
        n_analyzed = n_bins

    # 3. Vectorized tribology over high-load bins
    lambda_profile = np.full(n_bins, np.nan)
    scuff_flash_profile = np.full(n_bins, np.nan)
    scuff_integral_profile = np.full(n_bins, np.nan)
    data_statuses: list[str] = []
    data_messages: list[str] = []
    provenance_ref: dict[str, dict[str, str]] = {}
    lambda_perm_ref: float | None = None

    # Extract high-load values
    hl_hertz = hertz_stress_profile[high_load_mask]
    hl_sliding = sliding_velocity_profile[high_load_mask]
    hl_entrainment = entrainment_velocity_profile[high_load_mask]
    hl_indices = np.where(high_load_mask)[0]

    for i, idx in enumerate(hl_indices):
        trib_eval = evaluate_tribology(
            TribologyParams(
                hertz_stress_MPa=float(hl_hertz[i]),
                sliding_velocity_m_s=float(hl_sliding[i]),
                entrainment_velocity_m_s=float(hl_entrainment[i]),
                oil_viscosity_cSt=visc,
                composite_roughness_um=composite_roughness,
                bulk_temp_C=effective_bulk_temp_C,
                oil_inlet_temp_C=lube_params.supply_temp_C,
                oil_type=oil_type,
                additive_package=additive_package,
                finish_tier=finish_tier.value,
            ),
            scuff_method=tribology_scuff_method,
            validation_mode=validation_mode,
        )
        lambda_profile[idx] = float(trib_eval.lambda_min)
        scuff_flash_profile[idx] = float(trib_eval.scuff_margin_flash_C * scuff_mult)
        scuff_integral_profile[idx] = float(trib_eval.scuff_margin_integral_C * scuff_mult)
        lambda_perm_ref = float(trib_eval.lambda_perm)
        data_statuses.append(str(trib_eval.tribology_data_status))
        data_messages.extend(list(trib_eval.tribology_data_messages))
        if not provenance_ref and trib_eval.tribology_provenance:
            provenance_ref = dict(trib_eval.tribology_provenance)

    # 4. Worst-case values from analyzed bins
    lambda_analyzed = lambda_profile[high_load_mask]
    scuff_flash_analyzed = scuff_flash_profile[high_load_mask]
    scuff_integral_analyzed = scuff_integral_profile[high_load_mask]

    lambda_min = float(np.nanmin(lambda_analyzed))
    scuff_margin_flash = float(np.nanmin(scuff_flash_analyzed))
    scuff_margin_integral = float(np.nanmin(scuff_integral_analyzed))
    if method == "flash":
        scuff_margin = scuff_margin_flash
        method_used = "flash"
    elif method == "integral":
        scuff_margin = scuff_margin_integral
        method_used = "integral"
    else:
        scuff_margin = min(scuff_margin_flash, scuff_margin_integral)
        method_used = "flash" if scuff_margin_flash <= scuff_margin_integral else "integral"

    micropitting_sf = compute_micropitting_safety(
        lambda_min,
        lambda_perm=lambda_perm_ref,
        validation_mode=validation_mode,
        params=TribologyParams(
            oil_type=oil_type,
            finish_tier=finish_tier.value,
            additive_package=additive_package,
        ),
    )
    regime = classify_regime(lambda_min)

    # Worst-case phase angle (minimum lambda)
    worst_idx = int(np.nanargmin(lambda_profile))
    worst_phase_deg = float(worst_idx) / n_bins * 360.0

    # 5. Soft Material Selection & Safe Constraint Aggregation
    if params.material_state is not None:
        min_snap_dist, routes_weights = get_soft_selected_routes(
            params.material_state, operating_temp_C
        )
    else:
        mat_class = _material_from_level(params.material_quality_level)
        min_snap_dist, routes_weights = 0.0, [(mat_class.value, 1.0)]

    _temp_margins = []
    _costs = []
    _rankings = []

    for rid, alpha in routes_weights:
        mat_props = get_material(MaterialClass(rid))
        material_temp_margin = mat_props.max_service_temp_C - operating_temp_C
        total_cost = mat_props.cost_tier + finish_props.cost_multiplier + coating_props.cost_tier

        importance: dict[str, float] = {}
        importance["lubrication"] = max(0.0, (1.0 - lambda_min) * 10.0) if lambda_min < 1.5 else 0.3
        importance["surface_finish"] = max(0.0, 3.0 - lambda_min * 2.0) if lambda_min < 2.0 else 0.2
        importance["material"] = (
            max(0.0, (50.0 - material_temp_margin) / 10.0) if material_temp_margin < 80.0 else 0.3
        )
        importance["coating"] = (
            max(0.0, (100.0 - scuff_margin) / 20.0) if scuff_margin < 150.0 else 0.2
        )
        ranking = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)

        _temp_margins.append(material_temp_margin)
        _costs.append(alpha * total_cost)
        if alpha == max(w for _, w in routes_weights):
            _rankings = ranking

    safe_temp_margin = min(_temp_margins)
    expected_cost = sum(_costs)
    data_status = _merge_data_status(data_statuses)
    data_messages_dedup = tuple(dict.fromkeys(msg for msg in data_messages if str(msg).strip()))

    return PhaseResolvedResult(
        lambda_min=lambda_min,
        scuff_margin_flash_C=scuff_margin_flash,
        scuff_margin_integral_C=scuff_margin_integral,
        scuff_margin_C=scuff_margin,
        micropitting_safety=micropitting_sf,
        lube_regime=regime.value,
        tribology_method_used=method_used,
        tribology_data_status=data_status,
        material_temp_margin_C=safe_temp_margin,
        total_cost_index=expected_cost,
        feature_importance=_rankings,
        min_snap_distance=min_snap_dist,
        worst_phase_deg=worst_phase_deg,
        n_bins_analyzed=n_analyzed,
        force_threshold_N=force_threshold,
        lambda_profile=lambda_profile,
        tribology_data_messages=data_messages_dedup,
        tribology_provenance=provenance_ref,
    )
