"""Candidate evaluation — THE canonical interface.

This is the ONLY interface between physics and optimizers.
No optimizer-specific code should exist in physics modules.

Interface:
    evaluate_candidate(x, ctx) -> EvalResult(F, G, diag)

Flow:
    1. decode_candidate(x) -> Candidate
    2. eval_thermo(params.thermo, ctx) -> ThermoResult
    3. eval_gear(params.gear, i_req_profile, ctx) -> GearResult
    4. Machining surrogates -> tooling & tolerance constraints
    5. Real-world surrogates -> λ, scuffing, micropitting, material, cost
    6. Assemble F (objectives) and G (constraints)
    7. Return EvalResult with diagnostics
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import numpy as np

from ..architecture.contracts import log_contract_edge
from ..core.constants import MODEL_VERSION_GEAR_V1, MODEL_VERSION_THERMO_V1
from ..core.constraints import (
    GEAR_CONSTRAINTS,
    THERMO_CONSTRAINTS_FID0,
    THERMO_CONSTRAINTS_FID1,
    combine_constraints,
    get_constraint_kinds_for_phase,
)
from ..gear.litvin_core import eval_gear
from ..gear.manufacturability_limits import (
    ManufacturingProcessParams,
    compute_manufacturable_ratio_rate_limits,
)
from ..thermo.motionlaw import eval_thermo
from .encoding import decode_candidate
from .types import EvalContext, EvalResult


def _default_engine_mode_for_fidelity(fidelity: int) -> str:
    if int(fidelity) == 0:
        return "placeholder"
    if int(fidelity) == 1:
        return "hybrid"
    return "production"


def evaluate_candidate(x: np.ndarray, ctx: EvalContext) -> EvalResult:
    """Evaluate candidate solution.

    THE canonical interface between physics and optimizers.

    Args:
        x: Flat decision vector (length N_TOTAL).
        ctx: Evaluation context with rpm, torque, fidelity, seed.

    Returns:
        EvalResult with:
            F: Objectives [
                1-η_comb,
                1-η_exp,
                1-η_gear,
                motion_law_penalty,
                life_damage_penalty,
                material_risk_penalty,
            ] (minimize)
            G: Constraints (G <= 0 feasible)
            diag: Diagnostics dict
    """
    t0 = time.perf_counter()

    # Decode candidate
    try:
        candidate = decode_candidate(x)
        log_contract_edge(
            edge_id="edge.decode_candidate",
            engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
            status="ok",
            request_payload={
                "x": np.asarray(x, dtype=np.float64),
                "ctx": {"fidelity": int(ctx.fidelity)},
            },
            response_payload={"candidate": {"decoded": True}},
        )
    except Exception as exc:
        log_contract_edge(
            edge_id="edge.decode_candidate",
            engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
            status="error",
            request_payload={
                "x": np.asarray(x, dtype=np.float64),
                "ctx": {"fidelity": int(ctx.fidelity)},
            },
            response_payload={},
            error_signature=str(exc),
        )
        raise

    # Manufacturability-derived ratio-rate limit (cached per gear/process setup)
    process_cfg = ManufacturingProcessParams(**(ctx.gear_process_params or {}))
    duration_grid = np.array(
        [candidate.thermo.compression_duration, candidate.thermo.expansion_duration], dtype=float
    )
    rate_env = compute_manufacturable_ratio_rate_limits(
        candidate.gear,
        process=process_cfg,
        durations_deg=duration_grid,
    )
    dynamic_slope_limit = float(
        min(
            rate_env.slope_limit_for_duration(candidate.thermo.compression_duration),
            rate_env.slope_limit_for_duration(candidate.thermo.expansion_duration),
        )
    )
    if not np.isfinite(dynamic_slope_limit) or dynamic_slope_limit < 0.0:
        dynamic_slope_limit = 0.0

    # Apply the same static cap that thermo applies internally, so the
    # recorded ``applied_ratio_slope_limit`` matches ``ratio_slope_limit_used``.
    from ..core.constants import RATIO_SLOPE_LIMIT_FID0, RATIO_SLOPE_LIMIT_FID1

    static_limit = RATIO_SLOPE_LIMIT_FID1 if ctx.fidelity >= 1 else RATIO_SLOPE_LIMIT_FID0
    dynamic_slope_limit = min(dynamic_slope_limit, static_limit)

    # Thermo evaluation
    t_thermo_start = time.perf_counter()
    try:
        thermo_result = eval_thermo(candidate.thermo, ctx, ratio_slope_limit=dynamic_slope_limit)
        log_contract_edge(
            edge_id="edge.thermo.forward",
            engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
            status="ok",
            request_payload={
                "ctx": {
                    "fidelity": int(ctx.fidelity),
                    "rpm": float(ctx.rpm),
                    "torque": float(ctx.torque),
                    "fuel_name": str(
                        getattr(getattr(ctx, "breathing", None), "fuel_name", "gasoline")
                    ),
                },
                "thermo_params": {
                    "compression_duration": float(candidate.thermo.compression_duration),
                    "expansion_duration": float(candidate.thermo.expansion_duration),
                    "heat_release_center": float(candidate.thermo.heat_release_center),
                    "heat_release_width": float(candidate.thermo.heat_release_width),
                    "lambda_af": float(candidate.thermo.lambda_af),
                    "intake_open_offset_from_bdc": float(
                        candidate.thermo.intake_open_offset_from_bdc
                    ),
                    "intake_duration_deg": float(candidate.thermo.intake_duration_deg),
                    "exhaust_open_offset_from_expansion_tdc": float(
                        candidate.thermo.exhaust_open_offset_from_expansion_tdc
                    ),
                    "exhaust_duration_deg": float(candidate.thermo.exhaust_duration_deg),
                    "spark_timing_deg_from_compression_tdc": float(
                        candidate.thermo.spark_timing_deg_from_compression_tdc
                    ),
                },
            },
            response_payload={
                "efficiency": float(thermo_result.efficiency),
                "diag": {
                    "thermo_solver_status": str(
                        (thermo_result.diag or {}).get("thermo_solver_status", "")
                    ),
                    "thermo_model_version": str(
                        (thermo_result.diag or {}).get("thermo_model_version", "")
                    ),
                    "valve_timing": dict((thermo_result.diag or {}).get("valve_timing", {})),
                    "mixture_preparation": dict(
                        (thermo_result.diag or {}).get("mixture_preparation", {})
                    ),
                    "ignition_stage": dict((thermo_result.diag or {}).get("ignition_stage", {})),
                    "chemistry_handoff": dict(
                        (thermo_result.diag or {}).get("chemistry_handoff", {})
                    ),
                },
            },
        )
    except Exception as exc:
        log_contract_edge(
            edge_id="edge.thermo.forward",
            engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
            status="error",
            request_payload={
                "ctx": {
                    "fidelity": int(ctx.fidelity),
                    "rpm": float(ctx.rpm),
                    "torque": float(ctx.torque),
                    "fuel_name": str(
                        getattr(getattr(ctx, "breathing", None), "fuel_name", "gasoline")
                    ),
                },
                "thermo_params": {
                    "compression_duration": float(candidate.thermo.compression_duration),
                    "expansion_duration": float(candidate.thermo.expansion_duration),
                    "heat_release_center": float(candidate.thermo.heat_release_center),
                    "heat_release_width": float(candidate.thermo.heat_release_width),
                    "lambda_af": float(candidate.thermo.lambda_af),
                    "intake_open_offset_from_bdc": float(
                        candidate.thermo.intake_open_offset_from_bdc
                    ),
                    "intake_duration_deg": float(candidate.thermo.intake_duration_deg),
                    "exhaust_open_offset_from_expansion_tdc": float(
                        candidate.thermo.exhaust_open_offset_from_expansion_tdc
                    ),
                    "exhaust_duration_deg": float(candidate.thermo.exhaust_duration_deg),
                    "spark_timing_deg_from_compression_tdc": float(
                        candidate.thermo.spark_timing_deg_from_compression_tdc
                    ),
                },
            },
            response_payload={},
            error_signature=str(exc),
        )
        raise
    t_thermo = time.perf_counter() - t_thermo_start

    # Resolve material properties for gear eval (E' requires material DB)
    from dataclasses import replace

    from ..cem.material_db import MaterialClass, get_material
    from ..cem.material_snapping import get_soft_selected_routes
    from ..realworld.surrogates import _material_from_level

    # Default to 200C gear bulk filter temp unless we have thermo
    gear_bulk_filter_C = (
        float(thermo_result.diag.get("T_wall_C", 200.0)) if thermo_result.diag else 200.0
    )

    if candidate.realworld.material_state is not None:
        min_snap_dist, routes_weights = get_soft_selected_routes(
            candidate.realworld.material_state, gear_bulk_filter_C
        )
    else:
        # Legacy fallback
        mat_class = _material_from_level(candidate.realworld.material_quality_level)
        min_snap_dist, routes_weights = 0.0, [(mat_class.value, 1.0)]

    # We evaluate gear kinematics using the dominant route's properties
    dominant_rid = max(routes_weights, key=lambda rw: rw[1])[0]
    dominant_mat = get_material(MaterialClass(dominant_rid))
    ctx_gear = replace(ctx, material_properties=dominant_mat)

    # Gear evaluation (uses ratio profile from thermo)
    t_gear_start = time.perf_counter()
    try:
        gear_result = eval_gear(candidate.gear, thermo_result.requested_ratio_profile, ctx_gear)
        log_contract_edge(
            edge_id="edge.gear.forward",
            engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
            status="ok",
            request_payload={
                "ctx": {"fidelity": int(ctx.fidelity)},
                "i_req_profile": np.asarray(
                    thermo_result.requested_ratio_profile, dtype=np.float64
                ),
                "gear_params": {
                    "base_radius": float(candidate.gear.base_radius),
                    "face_width_mm": float(candidate.gear.face_width_mm),
                },
            },
            response_payload={
                "loss_total": float(gear_result.loss_total),
                "diag": {
                    "hertz_stress_max": float(
                        (gear_result.diag or {}).get("hertz_stress_max", np.nan)
                    )
                },
            },
        )
    except Exception as exc:
        log_contract_edge(
            edge_id="edge.gear.forward",
            engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
            status="error",
            request_payload={
                "ctx": {"fidelity": int(ctx.fidelity)},
                "i_req_profile": np.asarray(
                    thermo_result.requested_ratio_profile, dtype=np.float64
                ),
                "gear_params": {
                    "base_radius": float(candidate.gear.base_radius),
                    "face_width_mm": float(candidate.gear.face_width_mm),
                },
            },
            response_payload={},
            error_signature=str(exc),
        )
        raise
    t_gear = time.perf_counter() - t_gear_start

    # ===================================================================
    # Objective foundations
    # ===================================================================
    # First three axes remain the efficiency gaps:
    #   F[0] = 1 - η_comb  (chemical -> released heat)
    #   F[1] = 1 - η_exp   (released heat -> work)
    #   F[2] = 1 - η_gear  (piston/mech power -> output power)

    thermo_diag = thermo_result.diag or {}
    Q_chem = float(thermo_diag.get("Q_chem", 0.0))
    Q_rel = float(thermo_diag.get("Q_rel", 0.0))
    W = float(thermo_diag.get("W", 0.0))

    eta_comb = Q_rel / Q_chem if Q_chem > 0 else 0.0
    eta_exp = W / Q_rel if Q_rel > 0 else float(thermo_result.efficiency)

    eta_comb = float(np.clip(eta_comb, 0.0, 1.0))
    eta_exp = float(np.clip(eta_exp, 0.0, 1.0))

    loss_corrected = gear_result.loss_total
    surrogate_meta: dict[str, object] = {
        "surrogate_used": False,
        "residual_correction_used": False,
        "residual_correction_engine": "disabled",
        "openfoam_model_path": ctx.openfoam_model_path
        or os.environ.get("LARRAK2_OPENFOAM_NN_PATH", ""),
        "calculix_stress_mode": ctx.calculix_stress_mode,
        "gear_loss_mode": ctx.gear_loss_mode,
        "machining_mode": str(getattr(ctx, "machining_mode", "nn")),
        "machining_model_path": str(getattr(ctx, "machining_model_path", "") or ""),
        "surrogate_validation_mode": str(getattr(ctx, "surrogate_validation_mode", "strict")),
    }

    omega = float(ctx.rpm) * 2.0 * np.pi / 60.0
    P_out = float(ctx.torque) * omega
    denom = max(P_out + loss_corrected, 1e-12)
    eta_gear = float(np.clip(P_out / denom, 0.0, 1.0)) if P_out > 0 else 1.0

    # ===================================================================
    # Machining Cost & Constraints
    # ===================================================================
    from larrak2.machining.cost_model import calculate_tolerance_budget, calculate_tooling_cost
    from larrak2.surrogate.machining_inference import get_machining_engine

    machining_eng = get_machining_engine(
        mode=str(getattr(ctx, "machining_mode", "nn")),
        model_path=getattr(ctx, "machining_model_path", None),
    )

    # Extract surrogate inputs from ratio profile
    rr = thermo_result.requested_ratio_profile
    dev = rr - 1.0
    max_pos = np.max(dev)
    min_neg = np.min(dev)
    amp_surr = float(max_pos) if abs(max_pos) > abs(min_neg) else float(min_neg)

    shape_name = str(thermo_diag.get("motion_law", "Sine"))
    dur_comp = float(candidate.thermo.compression_duration)
    dur_exp = float(candidate.thermo.expansion_duration)

    # Predict for both flanks, take worst case
    tm_c, bm_c, hd_c, cr_c = machining_eng.predict(dur_comp, amp_surr, shape_name)
    tm_e, bm_e, hd_e, cr_e = machining_eng.predict(dur_exp, amp_surr, shape_name)

    t_min_proxy = min(tm_c, tm_e)
    b_max_survivable = min(bm_c, bm_e)
    min_hole_diam = min(hd_c, hd_e)
    min_curvature = min(cr_c, cr_e)

    # Tooling cost
    tooling_cost_val, is_standard_tool = calculate_tooling_cost(
        min_feature_size_outer_mm=2.0 * b_max_survivable, min_feature_size_inner_mm=min_hole_diam
    )

    # Tolerance budget
    tol_req, tol_penalty = calculate_tolerance_budget(
        min_ligament_mm=t_min_proxy,
        min_curvature_mm=min_curvature,
        # Dynamic proxy instead of fixed constant to keep machining logic coupled
        # to candidate geometry. This can be replaced by a richer geometric AR.
        aspect_ratio=float(candidate.gear.face_width_mm / max(candidate.gear.base_radius, 1e-6)),
        torque_nm=float(ctx.torque),
        budget_mm=float(ctx.tolerance_threshold_mm),
        mode=ctx.tolerance_constraint_mode,
    )

    # Tooling penalty (soft constraint for non-standard / expensive tools)
    tooling_penalty = max(0.0, tooling_cost_val - 2.0) * 0.1

    machining_G = [tol_penalty, tooling_penalty]
    machining_names = ["tol_budget", "tooling_cost"]
    log_contract_edge(
        edge_id="edge.machining.forward",
        engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
        status="ok",
        request_payload={
            "ctx": {
                "machining_mode": str(getattr(ctx, "machining_mode", "nn")),
                "tolerance_threshold_mm": float(ctx.tolerance_threshold_mm),
            }
        },
        response_payload={
            "tooling_cost": float(tooling_cost_val),
            "tol_penalty": float(tol_penalty),
        },
    )

    # ===================================================================
    # Real-World Checks (tribology, material, surface, lubrication)
    # ===================================================================
    from larrak2.realworld.constraints import compute_realworld_constraints
    from larrak2.realworld.surrogates import (
        RealWorldSurrogateParams,
        evaluate_realworld_phase_resolved,
        evaluate_realworld_surrogates,
    )

    # Decode real-world params from candidate (part of the decision vector)
    rw_params = RealWorldSurrogateParams(
        surface_finish_level=candidate.realworld.surface_finish_level,
        lube_mode_level=candidate.realworld.lube_mode_level,
        material_quality_level=candidate.realworld.material_quality_level,
        coating_level=candidate.realworld.coating_level,
        hunting_level=candidate.realworld.hunting_level,
        oil_flow_level=candidate.realworld.oil_flow_level,
        oil_supply_temp_level=candidate.realworld.oil_supply_temp_level,
        evacuation_level=candidate.realworld.evacuation_level,
    )

    # Operating conditions from gear evaluation
    gear_diag = gear_result.diag or {}
    operating_temp_C = float(thermo_diag.get("T_mean_wall", 200.0))
    pitch_line_vel = omega * candidate.gear.base_radius / 1000.0  # m/s

    strict_tribology_flag = (
        bool(ctx.strict_data)
        if ctx.strict_tribology_data is None
        else bool(ctx.strict_tribology_data)
    )
    if strict_tribology_flag:
        tribology_validation_mode = "strict"
    else:
        mode = str(getattr(ctx, "surrogate_validation_mode", "warn")).strip().lower()
        tribology_validation_mode = mode if mode in {"warn", "off"} else "warn"
    tribology_scuff_method = str(getattr(ctx, "tribology_scuff_method", "auto"))

    t_rw_start = time.perf_counter()

    # Phase-resolved evaluation if gear provides profile arrays
    hertz_prof = gear_diag.get("hertz_stress_profile")
    fn_prof = gear_diag.get("fn_profile")
    has_profiles = (
        hertz_prof is not None
        and fn_prof is not None
        and hasattr(hertz_prof, "__len__")
        and len(hertz_prof) > 1
    )

    strict_lifetime_data = bool(getattr(ctx, "strict_data", True))
    lifetime_mode_hint = str(getattr(ctx, "surrogate_validation_mode", "warn")).strip().lower()
    lifetime_validation_mode = (
        "strict" if strict_lifetime_data else ("off" if lifetime_mode_hint == "off" else "warn")
    )
    degraded_lifetime_token = (
        "degraded_off" if lifetime_validation_mode == "off" else "degraded_warn"
    )
    lifetime_data_messages: list[str] = []

    from larrak2.realworld.life_damage import (
        compute_life_damage_10k,
        compute_life_damage_scalar_proxy_10k,
        get_route_cleanliness_proxy,
        get_sigma_ref_for_route,
    )

    rpm_for_life = float(ctx.rpm)
    if not np.isfinite(rpm_for_life):
        if strict_lifetime_data:
            raise ValueError("strict_data=True but rpm is non-finite for lifetime evaluation.")
        rpm_for_life = 0.0
        lifetime_data_messages.append("Non-finite rpm in lifetime path; defaulting to 0.0.")

    hunting_level_for_life = float(rw_params.hunting_level)
    if not np.isfinite(hunting_level_for_life):
        if strict_lifetime_data:
            raise ValueError(
                "strict_data=True but hunting_level is non-finite for lifetime evaluation."
            )
        hunting_level_for_life = 0.5
        lifetime_data_messages.append(
            "Non-finite hunting_level in lifetime path; defaulting to 0.5."
        )

    life_damage_total = 0.0
    life_damage_diag: dict[str, object] = {
        "D_total": 0.0,
        "D_ring": 0.0,
        "D_planet": 0.0,
        "N_set": 1,
        "life_damage_status": "ok",
        "life_damage_input_mode": "phase_profile" if has_profiles else "scalar_proxy",
        "sigma_ref_used": float("nan"),
        "route_worst_case": "",
        "calibration_version": "",
        "messages": [],
    }

    if has_profiles:
        # Force-gated phase-resolved tribology (high-load bins only)
        import numpy as _np

        # Prefer gear model's sliding speed profile when available
        sliding_prof = gear_diag.get("sliding_speed_profile")
        if sliding_prof is not None and hasattr(sliding_prof, "__len__") and len(sliding_prof) > 1:
            sliding_v_prof = _np.asarray(sliding_prof, dtype=_np.float64)
        else:
            # Fallback: derive from radius gradient (omega is already rad/s)
            r_planet = gear_diag.get("r_planet")
            if r_planet is not None and hasattr(r_planet, "__len__"):
                sliding_v_prof = omega * _np.abs(_np.gradient(r_planet / 1000.0))
            else:
                sliding_v_prof = _np.full_like(hertz_prof, 5.0)

        entrainment_prof = gear_diag.get("entrainment_velocity_profile")
        if entrainment_prof is None or not hasattr(entrainment_prof, "__len__"):
            entrainment_prof = _np.full_like(hertz_prof, 15.0)

        _rw_results = []
        _life_damages = []
        route_iter = list(routes_weights)
        if not route_iter:
            if strict_lifetime_data:
                raise ValueError(
                    "strict_data=True but no material routes were resolved for life damage."
                )
            route_iter = [(str(dominant_rid), 1.0)]
            lifetime_data_messages.append(
                "No route weights available; using dominant route fallback for life-damage evaluation."
            )

        for rid, alpha in route_iter:
            cleanliness, life_status, clean_messages = get_route_cleanliness_proxy(
                rid,
                strict_data=strict_lifetime_data,
                validation_mode=lifetime_validation_mode,
            )
            if life_status != "ok":
                lifetime_data_messages.extend([f"[{rid}] {m}" for m in clean_messages])

            # Override parameters scalar representation for internal physics call
            from dataclasses import replace

            rw_params_override = replace(
                rw_params,
                material_state=None,
                material_quality_level=float(
                    np.clip(
                        dict(
                            [
                                ("AISI_9310", 0.0),
                                ("Pyrowear_53", 0.25),
                                ("CBS50_NiL", 0.5),
                                ("M50NiL", 0.75),
                                ("Ferrium_C64", 1.0),
                            ]
                        ).get(rid, 0.5),
                        0.0,
                        1.0,
                    )
                ),
            )

            phase_res_val = evaluate_realworld_phase_resolved(
                rw_params_override,
                hertz_stress_profile=_np.asarray(hertz_prof, dtype=_np.float64),
                sliding_velocity_profile=_np.asarray(sliding_v_prof, dtype=_np.float64),
                entrainment_velocity_profile=_np.asarray(entrainment_prof, dtype=_np.float64),
                fn_profile=_np.asarray(fn_prof, dtype=_np.float64),
                operating_temp_C=operating_temp_C,
                pitch_line_vel_m_s=pitch_line_vel,
                tribology_scuff_method=tribology_scuff_method,
                tribology_validation_mode=tribology_validation_mode,
                strict_tribology_data=strict_tribology_flag,
            )

            _sigma_ref = get_sigma_ref_for_route(
                rid,
                cleanliness_proxy=cleanliness,
                strict_data=bool(getattr(ctx, "strict_data", True)),
            )

            lambda_prof = getattr(phase_res_val, "lambda_profile", None)
            if lambda_prof is None:
                lambda_prof = _np.full_like(hertz_prof, max(phase_res_val.lambda_min, 0.1))

            lr = compute_life_damage_10k(
                hertz_stress_profile=_np.asarray(hertz_prof, dtype=_np.float64),
                lambda_profile=lambda_prof,
                fn_profile=_np.asarray(fn_prof, dtype=_np.float64),
                rpm=rpm_for_life,
                hunting_level=hunting_level_for_life,
                service_hours=10_000.0,
                sigma_ref_MPa=_sigma_ref,
            )

            lr["_route_id"] = str(rid)
            _rw_results.append((alpha, phase_res_val))
            _life_damages.append((alpha, lr))

        # Safe Constraint Aggregation over explicit top-k routes
        from larrak2.realworld.surrogates import PhaseResolvedResult

        rw_result = PhaseResolvedResult(
            lambda_min=min(m.lambda_min for a, m in _rw_results),
            scuff_margin_flash_C=min(m.scuff_margin_flash_C for a, m in _rw_results),
            scuff_margin_integral_C=min(m.scuff_margin_integral_C for a, m in _rw_results),
            scuff_margin_C=min(m.scuff_margin_C for a, m in _rw_results),
            micropitting_safety=min(m.micropitting_safety for a, m in _rw_results),
            lube_regime=_rw_results[0][1].lube_regime,
            tribology_method_used=_rw_results[0][1].tribology_method_used,
            tribology_data_status=(
                "degraded_off"
                if any(
                    str(m.tribology_data_status).strip().lower() == "degraded_off"
                    for _, m in _rw_results
                )
                else (
                    "degraded_warn"
                    if any(
                        str(m.tribology_data_status).strip().lower() == "degraded_warn"
                        for _, m in _rw_results
                    )
                    else "ok"
                )
            ),
            material_temp_margin_C=min(m.material_temp_margin_C for a, m in _rw_results),
            total_cost_index=sum(a * m.total_cost_index for a, m in _rw_results),
            feature_importance=_rw_results[0][1].feature_importance,
            min_snap_distance=min_snap_dist,
            worst_phase_deg=_rw_results[0][1].worst_phase_deg,
            n_bins_analyzed=_rw_results[0][1].n_bins_analyzed,
            force_threshold_N=_rw_results[0][1].force_threshold_N,
            lambda_profile=_rw_results[0][1].lambda_profile,
            tribology_data_messages=tuple(_rw_results[0][1].tribology_data_messages),
            tribology_provenance=dict(_rw_results[0][1].tribology_provenance),
        )

        worst_lr = max(
            (lr for _, lr in _life_damages),
            key=lambda rec: float(rec.get("D_total", 0.0)),
        )
        life_damage_total = float(worst_lr.get("D_total", 0.0))
        life_damage_diag = {
            "D_total": life_damage_total,
            "D_ring": float(worst_lr.get("D_ring", 0.0)),
            "D_planet": float(worst_lr.get("D_planet", 0.0)),
            "N_set": int(worst_lr.get("N_set", 1)),
            "life_damage_status": "ok" if not lifetime_data_messages else degraded_lifetime_token,
            "life_damage_input_mode": "phase_profile",
            "sigma_ref_used": float(worst_lr.get("sigma_ref_used", np.nan)),
            "route_worst_case": str(worst_lr.get("_route_id", "")),
            "calibration_version": str(worst_lr.get("calibration_version", "")),
            "messages": list(lifetime_data_messages),
        }

        phase_diag = {
            "worst_phase_deg": rw_result.worst_phase_deg,
            "n_bins_analyzed": rw_result.n_bins_analyzed,
            "force_threshold_N": rw_result.force_threshold_N,
            "phase_resolved": True,
        }
    else:
        # Scalar fallback (no profiles available)
        hertz_raw = gear_diag.get("hertz_stress_max")
        if hertz_raw is None:
            if strict_lifetime_data:
                raise ValueError(
                    "strict_data=True but gear diagnostics missing required 'hertz_stress_max' "
                    "for scalar life-damage proxy."
                )
            lifetime_data_messages.append(
                "Missing hertz_stress_max in scalar fallback; using conservative 1400 MPa."
            )
            hertz_stress = 1400.0
        else:
            hertz_stress = float(hertz_raw)
            if not np.isfinite(hertz_stress):
                if strict_lifetime_data:
                    raise ValueError(
                        "strict_data=True but hertz_stress_max is non-finite in scalar fallback."
                    )
                lifetime_data_messages.append(
                    "Non-finite hertz_stress_max in scalar fallback; using conservative 1400 MPa."
                )
                hertz_stress = 1400.0

        sliding_vel = float(gear_diag.get("sliding_speed_max", 5.0))
        if not np.isfinite(sliding_vel):
            sliding_vel = 5.0
        entrainment_vel = float(gear_diag.get("entrainment_velocity_mean", 15.0))
        if not np.isfinite(entrainment_vel):
            entrainment_vel = 15.0

        rw_result = evaluate_realworld_surrogates(
            rw_params,
            operating_temp_C=operating_temp_C,
            hertz_stress_MPa=hertz_stress,
            sliding_velocity_m_s=sliding_vel,
            entrainment_velocity_m_s=entrainment_vel,
            pitch_line_vel_m_s=pitch_line_vel,
            tribology_scuff_method=tribology_scuff_method,
            tribology_validation_mode=tribology_validation_mode,
            strict_tribology_data=strict_tribology_flag,
        )
        lambda_scalar = float(getattr(rw_result, "lambda_min", np.nan))
        if not np.isfinite(lambda_scalar):
            if strict_lifetime_data:
                raise ValueError(
                    "strict_data=True but scalar lifetime fallback received non-finite lambda_min."
                )
            lifetime_data_messages.append(
                "Non-finite lambda_min in scalar fallback; using conservative lambda=0.8."
            )
            lambda_scalar = 0.8

        hunting_level_scalar = float(hunting_level_for_life)
        if not np.isfinite(hunting_level_scalar):
            if strict_lifetime_data:
                raise ValueError(
                    "strict_data=True but hunting_level is non-finite for scalar lifetime proxy."
                )
            lifetime_data_messages.append(
                "Non-finite hunting_level in scalar fallback; defaulting to 0.5."
            )
            hunting_level_scalar = 0.5

        route_iter = list(routes_weights)
        if not route_iter:
            if strict_lifetime_data:
                raise ValueError(
                    "strict_data=True but no material routes were resolved for life damage."
                )
            route_iter = [(str(dominant_rid), 1.0)]
            lifetime_data_messages.append(
                "No route weights available; using dominant route fallback for life-damage evaluation."
            )

        _life_damages = []
        for rid, _alpha in route_iter:
            cleanliness, life_status, clean_messages = get_route_cleanliness_proxy(
                rid,
                strict_data=strict_lifetime_data,
                validation_mode=lifetime_validation_mode,
            )
            if life_status != "ok":
                lifetime_data_messages.extend([f"[{rid}] {m}" for m in clean_messages])

            sigma_ref = get_sigma_ref_for_route(
                rid,
                cleanliness_proxy=cleanliness,
                strict_data=strict_lifetime_data,
            )
            lr = compute_life_damage_scalar_proxy_10k(
                hertz_stress_MPa=float(hertz_stress),
                lambda_min=float(lambda_scalar),
                rpm=float(rpm_for_life),
                hunting_level=float(hunting_level_scalar),
                service_hours=10_000.0,
                sigma_ref_MPa=float(sigma_ref),
            )
            lr["_route_id"] = str(rid)
            _life_damages.append(lr)

        worst_lr = max(
            _life_damages,
            key=lambda rec: float(rec.get("D_total", 0.0)),
        )
        life_damage_total = float(worst_lr.get("D_total", 0.0))
        life_damage_diag = {
            "D_total": life_damage_total,
            "D_ring": float(worst_lr.get("D_ring", 0.0)),
            "D_planet": float(worst_lr.get("D_planet", 0.0)),
            "N_set": int(worst_lr.get("N_set", 1)),
            "life_damage_status": "ok" if not lifetime_data_messages else degraded_lifetime_token,
            "life_damage_input_mode": "scalar_proxy",
            "sigma_ref_used": float(worst_lr.get("sigma_ref_used", np.nan)),
            "route_worst_case": str(worst_lr.get("_route_id", "")),
            "calibration_version": str(worst_lr.get("calibration_version", "")),
            "messages": list(lifetime_data_messages),
        }
        phase_diag = {"phase_resolved": False}

    rw_G, rw_names = compute_realworld_constraints(
        rw_result,
        operating_temp_C=operating_temp_C,
        life_damage_total=life_damage_total,
        min_snap_distance=min_snap_dist,
    )
    t_rw = time.perf_counter() - t_rw_start
    log_contract_edge(
        edge_id="edge.realworld.forward",
        engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
        status="ok",
        request_payload={
            "tribology_scuff_method": tribology_scuff_method,
            "strict_tribology_data": bool(strict_tribology_flag),
        },
        response_payload={
            "lambda_min": float(getattr(rw_result, "lambda_min", np.nan)),
            "scuff_margin_flash_C": float(getattr(rw_result, "scuff_margin_flash_C", np.nan)),
            "scuff_margin_integral_C": float(getattr(rw_result, "scuff_margin_integral_C", np.nan)),
            "micropitting_safety": float(getattr(rw_result, "micropitting_safety", np.nan)),
            "material_temp_margin_C": float(getattr(rw_result, "material_temp_margin_C", np.nan)),
        },
    )
    log_contract_edge(
        edge_id="edge.lifetime.extract",
        engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
        status="ok",
        request_payload={"diag": {"realworld": {"life_damage": dict(life_damage_diag)}}},
        response_payload={
            "life_damage_total": float(life_damage_diag.get("D_total", np.nan)),
            "life_damage_status": str(life_damage_diag.get("life_damage_status", "")),
            "life_damage_input_mode": str(life_damage_diag.get("life_damage_input_mode", "")),
        },
    )

    # ===================================================================
    # Objectives (6D): efficiency + motion-law + lifetime + material risk
    # ===================================================================
    ratio_stats = thermo_diag.get("ratio_profile_stats", {})
    ratio_slope = float(ratio_stats.get("max_slope", 0.0))
    ratio_jerk = float(ratio_stats.get("max_jerk", 0.0))
    slope_limit_used = float(thermo_diag.get("ratio_slope_limit_used", dynamic_slope_limit))
    slope_util = ratio_slope / max(slope_limit_used, 1e-9)
    jerk_util = ratio_jerk / 1e6
    motion_law_penalty = float(0.6 * slope_util + 0.4 * jerk_util)

    life_damage_penalty = float(np.log10(1.0 + max(life_damage_total, 0.0)))

    temp_shortfall = max(0.0, -float(rw_result.material_temp_margin_C))
    snap_norm = max(0.0, float(min_snap_dist)) / 0.4
    cost_norm = max(0.0, float(rw_result.total_cost_index) - 1.0) / 5.0
    material_risk_penalty = float(temp_shortfall / 100.0 + 0.5 * snap_norm + 0.25 * cost_norm)

    F = np.array(
        [
            1.0 - eta_comb,
            1.0 - eta_exp,
            1.0 - eta_gear,
            motion_law_penalty,
            life_damage_penalty,
            material_risk_penalty,
        ],
        dtype=np.float64,
    )
    log_contract_edge(
        edge_id="edge.objectives.assemble",
        engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
        status="ok",
        request_payload={
            "eta_comb": float(eta_comb),
            "eta_exp": float(eta_exp),
            "eta_gear": float(eta_gear),
            "life_damage_total": float(life_damage_total),
        },
        response_payload={
            "F": np.asarray(F, dtype=np.float64),
            "diag": {
                "objectives": {
                    "names": [
                        "eta_comb_gap",
                        "eta_exp_gap",
                        "eta_gear_gap",
                        "motion_law_penalty",
                        "life_damage_penalty",
                        "material_risk_penalty",
                    ]
                }
            },
        },
    )

    # ===================================================================
    # Combine all constraints
    # ===================================================================
    # Thermo constraint names from centralized registry
    thermo_cnames = THERMO_CONSTRAINTS_FID1 if ctx.fidelity >= 1 else THERMO_CONSTRAINTS_FID0
    thermo_G = list(thermo_result.G)

    # Gear constraints (8 from eval_gear) + machining (2)
    gear_G_list = list(gear_result.G) + machining_G
    gear_names_list = list(GEAR_CONSTRAINTS) + machining_names

    # Combine thermo + gear (incl. machining) + realworld constraints
    constraint_kinds = get_constraint_kinds_for_phase(ctx.constraint_phase)
    G, constraint_diag = combine_constraints(
        thermo_G=thermo_G,
        gear_G=gear_G_list,
        thermo_names=thermo_cnames,
        gear_names=gear_names_list,
        kind_overrides=constraint_kinds,
        realworld_G=rw_G,
        realworld_names=rw_names,
    )
    log_contract_edge(
        edge_id="edge.constraints.combine",
        engine_mode=_default_engine_mode_for_fidelity(ctx.fidelity),
        status="ok",
        request_payload={
            "thermo_G": list(thermo_G),
            "gear_G": list(gear_G_list),
            "constraint_phase": str(ctx.constraint_phase),
        },
        response_payload={
            "G": np.asarray(G, dtype=np.float64),
            "diag": {"constraints": constraint_diag},
        },
    )

    # ===================================================================
    # Diagnostics
    # ===================================================================
    t_total = time.perf_counter() - t0

    diag: dict = {
        "thermo": thermo_diag,
        "thermo_validation": {
            "thermo_solver_status": str(thermo_diag.get("thermo_solver_status", "")),
            "thermo_model_version": str(thermo_diag.get("thermo_model_version", "")),
            "thermo_constants_version": str(thermo_diag.get("thermo_constants_version", "")),
            "thermo_timing_profile_id": str(
                (thermo_diag.get("valve_timing", {}) or {}).get("timing_profile_id", "")
            ),
            "thermo_timing_profile_version": str(
                (thermo_diag.get("valve_timing", {}) or {}).get("timing_profile_version", "")
            ),
            "thermo_mass_residual": float(thermo_diag.get("thermo_mass_residual", np.nan)),
            "thermo_energy_residual": float(thermo_diag.get("thermo_energy_residual", np.nan)),
            "thermo_benchmark_status": str(thermo_diag.get("thermo_benchmark_status", "")),
            "thermo_nn_disagreement": thermo_diag.get("thermo_nn_disagreement", {}),
        },
        "gear": gear_result.diag or {},
        "machining": {
            "t_min_proxy_mm": t_min_proxy,
            "b_max_survivable_mm": b_max_survivable,
            "min_hole_diam_mm": min_hole_diam,
            "tooling_cost": tooling_cost_val,
            "is_standard_tool": is_standard_tool,
            "tol_required_mm": tol_req,
            "tol_penalty": tol_penalty,
            "inputs": {"dur_comp": dur_comp, "amp": amp_surr, "shape": shape_name},
        },
        "realworld": {
            "lambda_min": rw_result.lambda_min,
            "scuff_margin_flash_C": rw_result.scuff_margin_flash_C,
            "scuff_margin_integral_C": rw_result.scuff_margin_integral_C,
            "scuff_margin_C": rw_result.scuff_margin_C,
            "micropitting_safety": rw_result.micropitting_safety,
            "lube_regime": rw_result.lube_regime,
            "tribology_method_used": rw_result.tribology_method_used,
            "tribology_data_status": rw_result.tribology_data_status,
            "tribology_data_messages": list(getattr(rw_result, "tribology_data_messages", ())),
            "tribology_provenance": dict(getattr(rw_result, "tribology_provenance", {})),
            "material_temp_margin_C": rw_result.material_temp_margin_C,
            "cost_index": rw_result.total_cost_index,
            "feature_importance": rw_result.feature_importance,
            "life_damage_status": str(life_damage_diag.get("life_damage_status", "ok")),
            "life_damage_input_mode": str(life_damage_diag.get("life_damage_input_mode", "")),
            "lifetime_validation_mode": str(lifetime_validation_mode),
            "params": {
                "surface_finish_level": rw_params.surface_finish_level,
                "lube_mode_level": rw_params.lube_mode_level,
                "material_quality_level": rw_params.material_quality_level,
                "coating_level": rw_params.coating_level,
                "oil_flow_level": rw_params.oil_flow_level,
                "oil_supply_temp_level": rw_params.oil_supply_temp_level,
                "evacuation_level": rw_params.evacuation_level,
                "hunting_level": candidate.realworld.hunting_level,
                "tribology_scuff_method": tribology_scuff_method,
                "tribology_validation_mode": tribology_validation_mode,
                "strict_tribology_data": strict_tribology_flag,
            },
            "life_damage": life_damage_diag,
            **phase_diag,  # worst_phase_deg, n_bins_analyzed, etc.
        },
        "constraints": constraint_diag,
        "surrogate": surrogate_meta,
        "timings": {
            "thermo_ms": t_thermo * 1000,
            "gear_ms": t_gear * 1000,
            "realworld_ms": t_rw * 1000,
            "total_ms": t_total * 1000,
        },
        "metrics": {
            "eta_comb": eta_comb,
            "eta_exp": eta_exp,
            "eta_gear": eta_gear,
            "loss_total": loss_corrected,
            "dynamic_slope_limit": dynamic_slope_limit,
        },
        "objectives": {
            "names": [
                "eta_comb_gap",
                "eta_exp_gap",
                "eta_gear_gap",
                "motion_law_penalty",
                "life_damage_penalty",
                "material_risk_penalty",
            ],
            "values": F.tolist(),
            "motion_law": {
                "max_slope": ratio_slope,
                "max_jerk": ratio_jerk,
                "slope_limit_used": slope_limit_used,
                "slope_utilization": slope_util,
                "jerk_utilization": jerk_util,
            },
            "life_damage_total": life_damage_total,
            "material": {
                "temp_shortfall_C": temp_shortfall,
                "snap_distance_norm": snap_norm,
                "cost_index_norm": cost_norm,
            },
        },
        "versions": {
            "thermo": MODEL_VERSION_THERMO_V1,
            "gear": MODEL_VERSION_GEAR_V1,
        },
    }

    return EvalResult(F=F, G=G, diag=diag)


def evaluate_candidate_batch(
    X: np.ndarray | Sequence[np.ndarray], ctx: EvalContext
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Evaluate a batch of candidates.

    Args:
        X: Array-like of shape (n, N_TOTAL) or iterable of 1-D arrays.
        ctx: Shared evaluation context.

    Returns:
        F_all: (n, n_obj) objective array
        G_all: (n, n_constr) constraint array
        diags: list of diagnostics dicts
    """
    X_arr = np.atleast_2d(np.asarray(list(X)) if not isinstance(X, np.ndarray) else X)
    results = [evaluate_candidate(x, ctx) for x in X_arr]
    F_all = np.stack([r.F for r in results], axis=0)
    G_all = np.stack([r.G for r in results], axis=0)
    diags = [r.diag for r in results]
    return F_all, G_all, diags
