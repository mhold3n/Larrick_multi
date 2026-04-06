"""Full CEM evaluation — post-optimization validation without surrogates.

Composes all CEM domain modules (material, tribology, surface finish,
lubrication, post-processing) into a single evaluation that produces
a comprehensive feasibility assessment and feature-importance ranking.

This is NOT called during the optimization loop (too expensive).
It validates the final optimized output using the authoritative models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from larrak_runtime.cem.lubrication import (
    LubricationMode,
    LubricationParams,
    churning_loss_factor,
    cooling_effectiveness,
    effective_viscosity,
)
from larrak_runtime.cem.material_db import MaterialClass, get_material
from larrak_runtime.cem.post_processing import (
    CoatingType,
    HeatTreatment,
    apply_coating_modifiers,
    apply_heat_treat_modifiers,
    get_coating,
    get_heat_treat,
)
from larrak_runtime.cem.surface_finish import (
    SurfaceFinishTier,
    effective_composite_roughness,
    get_finish_properties,
)
from larrak_runtime.cem.tribology import (
    TribologyParams,
    classify_regime,
    compute_micropitting_safety,
    evaluate_tribology,
)


@dataclass(frozen=True)
class CEMEvalParams:
    """Inputs for full CEM evaluation.

    Combines design choices (material, surface, lube, coating, heat treat)
    with operating conditions and gear geometry proxies.
    """

    # Design choices
    material: MaterialClass = MaterialClass.CBS50_NIL
    surface_finish: SurfaceFinishTier = SurfaceFinishTier.SUPERFINISHED
    lubrication: LubricationParams = LubricationParams()
    coating: CoatingType = CoatingType.NONE
    heat_treatment: HeatTreatment = HeatTreatment.CARBURIZED_HOT_HARD

    # Operating conditions
    operating_temp_C: float = 200.0
    hertz_stress_MPa: float = 1200.0
    sliding_velocity_m_s: float = 5.0
    entrainment_velocity_m_s: float = 15.0
    pitch_line_vel_m_s: float = 20.0
    base_friction_coeff: float = 0.06
    tribology_scuff_method: str = "auto"
    tribology_validation_mode: str = "strict"
    strict_tribology_data: bool | None = None


@dataclass
class CEMResult:
    """Output of full CEM evaluation.

    Attributes:
        lambda_min: Minimum specific film thickness.
        scuff_margin_C: Scuffing temperature margin (°C), positive = safe.
        micropitting_safety: S_λ safety factor (≥1.0 acceptable).
        lube_regime: Lubrication regime classification.
        cooling_eff: Cooling effectiveness (0–1).
        fatigue_life_multiplier: Combined fatigue life multiplier.
        total_cost_index: Combined cost index (material + finish + coating).
        churning_loss: Relative churning/windage loss factor.
        material_temp_margin_C: Material max temp − operating temp (°C).
        recommendation_ranking: Ordered list of (feature, importance)
            tuples, from most to least important for feasibility.
        details: Full diagnostic dictionary.
    """

    lambda_min: float
    scuff_margin_flash_C: float
    scuff_margin_integral_C: float
    scuff_margin_C: float
    micropitting_safety: float
    lube_regime: str
    tribology_method_used: str
    tribology_data_status: str
    cooling_eff: float
    fatigue_life_multiplier: float
    total_cost_index: float
    churning_loss: float
    material_temp_margin_C: float
    recommendation_ranking: list[tuple[str, float]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_cem(params: CEMEvalParams) -> CEMResult:
    """Run full CEM evaluation.

    Composes all domain modules without surrogates.  Produces a
    feasibility assessment and feature-importance ranking.

    Args:
        params: Complete CEM evaluation inputs.

    Returns:
        CEMResult with all computed metrics and recommendations.
    """
    # 1. Material properties
    mat_props = get_material(params.material)
    material_temp_margin = mat_props.max_service_temp_C - params.operating_temp_C

    # 2. Surface finish → composite roughness
    finish_props = get_finish_properties(params.surface_finish)
    composite_roughness = effective_composite_roughness(params.surface_finish)

    # 3. Lubrication → viscosity at contact, cooling effectiveness
    contact_temp = params.operating_temp_C  # Simplified: contact ≈ bulk
    visc = effective_viscosity(params.lubrication, contact_temp)
    cool_eff = cooling_effectiveness(params.lubrication, params.pitch_line_vel_m_s)
    churn = churning_loss_factor(params.lubrication, params.pitch_line_vel_m_s)

    # 4. Coating effects on friction and scuff resistance
    modified_scuff_mult, modified_friction = apply_coating_modifiers(
        base_scuff_margin=1.0,  # Multiplier, not absolute
        base_friction=params.base_friction_coeff,
        coating=params.coating,
    )

    # 5. Heat treatment effects on fatigue
    fatigue_mult = apply_heat_treat_modifiers(
        base_fatigue_life_mult=mat_props.fatigue_life_multiplier,
        heat_treat=params.heat_treatment,
        operating_temp_C=params.operating_temp_C,
    )
    # Surface finish also contributes to fatigue life
    fatigue_mult *= finish_props.micropitting_life_multiplier

    # 6. Tribology calculations (ISO data-backed)
    oil_type = (
        "high_ep"
        if params.lubrication.mode
        in {LubricationMode.PRESSURIZED_JET, LubricationMode.PHASE_GATED_JET}
        else "generic_ep"
    )
    additive_package = "high_ep" if oil_type == "high_ep" else "standard_ep"
    effective_bulk_temp_C = max(20.0, float(params.operating_temp_C) - 35.0 * float(cool_eff))
    validation_mode = str(params.tribology_validation_mode).strip().lower()
    if params.strict_tribology_data is True:
        validation_mode = "strict"
    elif params.strict_tribology_data is False and validation_mode == "strict":
        validation_mode = "warn"

    trib_eval = evaluate_tribology(
        TribologyParams(
            hertz_stress_MPa=params.hertz_stress_MPa,
            sliding_velocity_m_s=params.sliding_velocity_m_s,
            entrainment_velocity_m_s=params.entrainment_velocity_m_s,
            oil_viscosity_cSt=visc,
            composite_roughness_um=composite_roughness,
            bulk_temp_C=effective_bulk_temp_C,
            oil_inlet_temp_C=params.lubrication.supply_temp_C,
            oil_type=oil_type,
            additive_package=additive_package,
            finish_tier=params.surface_finish.value,
            friction_coeff=modified_friction,
        ),
        scuff_method=params.tribology_scuff_method,
        validation_mode=validation_mode,
    )

    lambda_min = float(trib_eval.lambda_min)
    scuff_margin_flash = float(trib_eval.scuff_margin_flash_C * modified_scuff_mult)
    scuff_margin_integral = float(trib_eval.scuff_margin_integral_C * modified_scuff_mult)
    method_pref = str(params.tribology_scuff_method).strip().lower()
    if method_pref == "flash":
        scuff_margin = scuff_margin_flash
        scuff_method_used = "flash"
    elif method_pref == "integral":
        scuff_margin = scuff_margin_integral
        scuff_method_used = "integral"
    else:
        scuff_margin = min(scuff_margin_flash, scuff_margin_integral)
        scuff_method_used = "flash" if scuff_margin_flash <= scuff_margin_integral else "integral"

    micropitting_sf = compute_micropitting_safety(lambda_min, lambda_perm=trib_eval.lambda_perm)
    regime = classify_regime(lambda_min)

    # 7. Cost index
    coating_props = get_coating(params.coating)
    ht_props = get_heat_treat(params.heat_treatment)
    total_cost = (
        mat_props.cost_tier
        + finish_props.cost_multiplier
        + coating_props.cost_tier
        + ht_props.cost_multiplier
    )

    # 8. Feature importance ranking
    # Score each feature by its contribution to feasibility.
    # Higher score = more important for making this gear work.
    importance: dict[str, float] = {}

    # Lubrication importance: how far from full EHL boundary
    if lambda_min < 1.0:
        importance["lubrication"] = (1.0 - lambda_min) * 10.0
    else:
        importance["lubrication"] = 0.5

    # Surface finish: how much roughness is limiting λ
    importance["surface_finish"] = max(0.0, 3.0 - lambda_min * 2.0)

    # Material: temperature margin importance
    if material_temp_margin < 50.0:
        importance["material"] = max(0.0, (50.0 - material_temp_margin) / 10.0)
    else:
        importance["material"] = 0.5

    # Coating: scuffing margin importance
    if scuff_margin < 100.0:
        importance["coating"] = max(0.0, (100.0 - scuff_margin) / 20.0)
    else:
        importance["coating"] = 0.3

    # Sort by importance (descending)
    ranking = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)

    # 9. Details
    details = {
        "material": {
            "class": params.material.value,
            "name": mat_props.name,
            "temp_margin_C": material_temp_margin,
        },
        "surface": {
            "tier": params.surface_finish.value,
            "Ra_um": finish_props.Ra_um,
            "composite_roughness_um": composite_roughness,
        },
        "lubrication": {
            "mode": params.lubrication.mode.value,
            "viscosity_at_contact_cSt": visc,
            "cooling_effectiveness": cool_eff,
            "churning_loss": churn,
        },
        "coating": {
            "type": params.coating.value,
            "friction_reduction": coating_props.friction_coeff_reduction,
            "scuff_multiplier": modified_scuff_mult,
        },
        "heat_treatment": {
            "type": params.heat_treatment.value,
            "hardness_retention_300C": ht_props.hardness_retention_300C,
        },
        "tribology": {
            "lambda_min": lambda_min,
            "regime": regime.value,
            "scuff_margin_flash_C": scuff_margin_flash,
            "scuff_margin_integral_C": scuff_margin_integral,
            "scuff_margin_C": scuff_margin,
            "micropitting_safety": micropitting_sf,
            "tribology_method_used": scuff_method_used,
            "tribology_data_status": trib_eval.tribology_data_status,
            "tribology_data_messages": list(trib_eval.tribology_data_messages),
            "tribology_provenance": trib_eval.tribology_provenance,
        },
    }

    return CEMResult(
        lambda_min=lambda_min,
        scuff_margin_flash_C=scuff_margin_flash,
        scuff_margin_integral_C=scuff_margin_integral,
        scuff_margin_C=scuff_margin,
        micropitting_safety=micropitting_sf,
        lube_regime=regime.value,
        tribology_method_used=scuff_method_used,
        tribology_data_status=trib_eval.tribology_data_status,
        cooling_eff=cool_eff,
        fatigue_life_multiplier=fatigue_mult,
        total_cost_index=total_cost,
        churning_loss=churn,
        material_temp_margin_C=material_temp_margin,
        recommendation_ranking=ranking,
        details=details,
    )
