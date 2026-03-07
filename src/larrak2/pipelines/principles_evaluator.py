"""Reduced-order principles evaluator and canonical alignment bridge."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from larrak2.core.constants import RATIO_SLOPE_LIMIT_FID1
from larrak2.core.constraints import (
    GEAR_CONSTRAINTS,
    THERMO_CONSTRAINTS_FID1,
    combine_constraints,
    get_constraint_kinds_for_phase,
)
from larrak2.core.encoding import decode_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext
from larrak2.gear.litvin_core import eval_gear
from larrak2.gear.manufacturability_limits import (
    ManufacturingProcessParams,
    compute_manufacturable_ratio_rate_limits,
)
from larrak2.machining.cost_model import calculate_tolerance_budget, calculate_tooling_cost
from larrak2.pipelines.principles_core import PRINCIPLES_OBJECTIVE_NAMES, expand_reduced_vector
from larrak2.realworld.constraints import compute_realworld_constraints
from larrak2.realworld.life_damage import (
    compute_life_damage_scalar_proxy_10k,
    get_route_cleanliness_proxy,
    get_sigma_ref_for_route,
)
from larrak2.realworld.surrogates import (
    RealWorldSurrogateParams,
    _material_from_level,
    evaluate_realworld_surrogates,
)
from larrak2.surrogate.machining_inference import get_machining_engine
from larrak2.thermo.motionlaw import eval_thermo


@dataclass(frozen=True)
class PrinciplesProxyResult:
    F: np.ndarray
    G: np.ndarray
    diag: dict[str, Any]
    objective_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    x_full: np.ndarray
    expansion_policy: dict[str, Any]
    error_signature: str = ""


@dataclass(frozen=True)
class PrinciplesAlignmentResult:
    F: np.ndarray
    G: np.ndarray
    diag: dict[str, Any]
    objective_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    error_signature: str = ""


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _proxy_ctx(base_ctx: EvalContext, *, rpm: float, torque: float) -> EvalContext:
    return replace(
        base_ctx,
        rpm=float(rpm),
        torque=float(torque),
        fidelity=1,
        constraint_phase="explore",
        calculix_stress_mode="analytical",
        gear_loss_mode="physics",
        machining_mode="analytical",
        thermo_symbolic_mode="off",
    )


def _alignment_ctx(
    base_ctx: EvalContext,
    *,
    rpm: float,
    torque: float,
    fidelity: int,
    constraint_phase: str,
) -> EvalContext:
    return replace(
        base_ctx,
        rpm=float(rpm),
        torque=float(torque),
        fidelity=int(fidelity),
        constraint_phase=str(constraint_phase),
        calculix_stress_mode="analytical",
        gear_loss_mode="physics",
        machining_mode="analytical",
        thermo_symbolic_mode="off",
    )


def _dynamic_slope_limit(x_full: np.ndarray, ctx: EvalContext) -> float:
    candidate = decode_candidate(x_full)
    process_cfg = ManufacturingProcessParams(**(ctx.gear_process_params or {}))
    duration_grid = np.array(
        [candidate.thermo.compression_duration, candidate.thermo.expansion_duration], dtype=float
    )
    rate_env = compute_manufacturable_ratio_rate_limits(
        candidate.gear,
        process=process_cfg,
        durations_deg=duration_grid,
    )
    dynamic_limit = float(
        min(
            rate_env.slope_limit_for_duration(candidate.thermo.compression_duration),
            rate_env.slope_limit_for_duration(candidate.thermo.expansion_duration),
        )
    )
    if not np.isfinite(dynamic_limit) or dynamic_limit < 0.0:
        dynamic_limit = 0.0
    return float(min(dynamic_limit, RATIO_SLOPE_LIMIT_FID1))


def _proxy_failure(
    *,
    x_full: np.ndarray,
    expansion_policy: dict[str, Any],
    error_signature: str,
) -> PrinciplesProxyResult:
    constraint_names = tuple(
        list(THERMO_CONSTRAINTS_FID1)
        + list(GEAR_CONSTRAINTS)
        + ["tol_budget", "tooling_cost"]
        + [
            "rw_lambda_min",
            "rw_scuff_margin",
            "rw_micropitting_sf",
            "rw_material_temp",
            "rw_cost_index",
            "rw_life_damage_10k",
            "rw_material_snap_dist",
        ]
    )
    return PrinciplesProxyResult(
        F=np.full(len(PRINCIPLES_OBJECTIVE_NAMES), 1.0e6, dtype=np.float64),
        G=np.full(len(constraint_names), 1.0e3, dtype=np.float64),
        diag={"error_signature": str(error_signature), "constraints": [], "objectives": {}},
        objective_names=PRINCIPLES_OBJECTIVE_NAMES,
        constraint_names=constraint_names,
        x_full=np.asarray(x_full, dtype=np.float64),
        expansion_policy=expansion_policy,
        error_signature=str(error_signature),
    )


def _alignment_failure(error_signature: str, n_constr: int) -> PrinciplesAlignmentResult:
    return PrinciplesAlignmentResult(
        F=np.full(len(PRINCIPLES_OBJECTIVE_NAMES), 1.0e6, dtype=np.float64),
        G=np.full(max(1, n_constr), 1.0e3, dtype=np.float64),
        diag={"error_signature": str(error_signature), "constraints": [], "objectives": {}},
        objective_names=PRINCIPLES_OBJECTIVE_NAMES,
        constraint_names=tuple(),
        error_signature=str(error_signature),
    )


def evaluate_principles_proxy(
    reduced_vector: np.ndarray,
    *,
    profile_payload: dict[str, Any],
    base_ctx: EvalContext,
    rpm: float,
    torque: float,
) -> PrinciplesProxyResult:
    x_full, expansion_policy = expand_reduced_vector(
        reduced_vector,
        profile_payload=profile_payload,
        rpm=float(rpm),
    )
    ctx = _proxy_ctx(base_ctx, rpm=float(rpm), torque=float(torque))

    try:
        candidate = decode_candidate(x_full)
        slope_limit = _dynamic_slope_limit(x_full, ctx)
        thermo_result = eval_thermo(candidate.thermo, ctx, ratio_slope_limit=slope_limit)

        thermo_diag = dict(thermo_result.diag or {})
        Q_chem = float(thermo_diag.get("Q_chem", 0.0))
        Q_rel = float(thermo_diag.get("Q_rel", 0.0))
        W = float(thermo_diag.get("W", 0.0))
        eta_comb = float(np.clip(Q_rel / Q_chem if Q_chem > 0 else 0.0, 0.0, 1.0))
        eta_exp = float(
            np.clip(W / Q_rel if Q_rel > 0 else float(thermo_result.efficiency), 0.0, 1.0)
        )

        mat_class = _material_from_level(candidate.realworld.material_quality_level)
        from larrak2.cem.material_db import get_material

        ctx_gear = replace(ctx, material_properties=get_material(mat_class))
        gear_result = eval_gear(candidate.gear, thermo_result.requested_ratio_profile, ctx_gear)
        gear_diag = dict(gear_result.diag or {})
        loss_corrected = float(gear_result.loss_total)
        omega = float(ctx.rpm) * 2.0 * np.pi / 60.0
        P_out = float(ctx.torque) * omega
        eta_gear = (
            float(np.clip(P_out / max(P_out + loss_corrected, 1e-12), 0.0, 1.0))
            if P_out > 0
            else 1.0
        )

        machining_eng = get_machining_engine(
            mode="analytical",
            model_path=None,
        )
        rr = np.asarray(thermo_result.requested_ratio_profile, dtype=np.float64)
        dev = rr - 1.0
        max_pos = float(np.max(dev))
        min_neg = float(np.min(dev))
        amp_surr = float(max_pos) if abs(max_pos) > abs(min_neg) else float(min_neg)
        shape_name = str(thermo_diag.get("motion_law", "Sine"))
        dur_comp = float(candidate.thermo.compression_duration)
        dur_exp = float(candidate.thermo.expansion_duration)
        tm_c, bm_c, hd_c, cr_c = machining_eng.predict(dur_comp, amp_surr, shape_name)
        tm_e, bm_e, hd_e, cr_e = machining_eng.predict(dur_exp, amp_surr, shape_name)
        t_min_proxy = min(tm_c, tm_e)
        b_max_survivable = min(bm_c, bm_e)
        min_hole_diam = min(hd_c, hd_e)
        min_curvature = min(cr_c, cr_e)
        tooling_cost_val, is_standard_tool = calculate_tooling_cost(
            min_feature_size_outer_mm=2.0 * b_max_survivable,
            min_feature_size_inner_mm=min_hole_diam,
        )
        tol_req, tol_penalty = calculate_tolerance_budget(
            min_ligament_mm=t_min_proxy,
            min_curvature_mm=min_curvature,
            aspect_ratio=float(
                candidate.gear.face_width_mm / max(candidate.gear.base_radius, 1e-6)
            ),
            torque_nm=float(ctx.torque),
            budget_mm=float(ctx.tolerance_threshold_mm),
            mode=ctx.tolerance_constraint_mode,
        )
        tooling_penalty = max(0.0, float(tooling_cost_val) - 2.0) * 0.1
        machining_G = [float(tol_penalty), float(tooling_penalty)]
        machining_names = ["tol_budget", "tooling_cost"]

        rw_params = RealWorldSurrogateParams(
            surface_finish_level=float(candidate.realworld.surface_finish_level),
            lube_mode_level=float(candidate.realworld.lube_mode_level),
            material_quality_level=float(candidate.realworld.material_quality_level),
            coating_level=float(candidate.realworld.coating_level),
            hunting_level=float(candidate.realworld.hunting_level),
            oil_flow_level=float(candidate.realworld.oil_flow_level),
            oil_supply_temp_level=float(candidate.realworld.oil_supply_temp_level),
            evacuation_level=float(candidate.realworld.evacuation_level),
        )
        operating_temp_C = float(thermo_diag.get("T_mean_wall", 200.0))
        pitch_line_vel = omega * candidate.gear.base_radius / 1000.0
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
        hertz_stress = float(gear_diag.get("hertz_stress_max", 1400.0))
        sliding_vel = float(gear_diag.get("sliding_speed_max", 5.0))
        entrainment_vel = float(gear_diag.get("entrainment_velocity_mean", 15.0))
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
        route_id = str(mat_class.value)
        cleanliness, _status, _messages = get_route_cleanliness_proxy(
            route_id,
            strict_data=bool(ctx.strict_data),
            validation_mode=("strict" if bool(ctx.strict_data) else "warn"),
        )
        sigma_ref = get_sigma_ref_for_route(
            route_id,
            cleanliness_proxy=cleanliness,
            strict_data=bool(ctx.strict_data),
        )
        life_damage_diag = compute_life_damage_scalar_proxy_10k(
            hertz_stress_MPa=float(hertz_stress),
            lambda_min=float(rw_result.lambda_min),
            rpm=float(ctx.rpm),
            hunting_level=float(rw_params.hunting_level),
            service_hours=10_000.0,
            sigma_ref_MPa=float(sigma_ref),
        )
        life_damage_diag.update(
            {
                "life_damage_status": "ok",
                "life_damage_input_mode": "scalar_proxy",
                "route_worst_case": route_id,
                "messages": [],
            }
        )
        life_damage_total = float(life_damage_diag.get("D_total", 0.0))
        rw_G, rw_names = compute_realworld_constraints(
            rw_result,
            operating_temp_C=operating_temp_C,
            life_damage_total=life_damage_total,
            min_snap_distance=0.0,
        )

        ratio_stats = thermo_diag.get("ratio_profile_stats", {})
        ratio_slope = float(ratio_stats.get("max_slope", 0.0))
        ratio_jerk = float(ratio_stats.get("max_jerk", 0.0))
        slope_limit_used = float(thermo_diag.get("ratio_slope_limit_used", slope_limit))
        slope_util = ratio_slope / max(slope_limit_used, 1e-9)
        jerk_util = ratio_jerk / 1e6
        motion_law_penalty = float(0.6 * slope_util + 0.4 * jerk_util)
        life_damage_penalty = float(np.log10(1.0 + max(life_damage_total, 0.0)))
        temp_shortfall = max(0.0, -float(rw_result.material_temp_margin_C))
        snap_norm = 0.0
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

        G, constraint_diag = combine_constraints(
            thermo_G=list(np.asarray(thermo_result.G, dtype=np.float64)),
            gear_G=list(np.asarray(gear_result.G, dtype=np.float64)) + machining_G,
            thermo_names=list(THERMO_CONSTRAINTS_FID1),
            gear_names=list(GEAR_CONSTRAINTS) + machining_names,
            kind_overrides=get_constraint_kinds_for_phase("explore"),
            realworld_G=rw_G,
            realworld_names=rw_names,
        )
        diag = {
            "thermo": thermo_diag,
            "gear": _jsonify(gear_diag),
            "machining": {
                "tooling_cost": float(tooling_cost_val),
                "tol_penalty": float(tol_penalty),
                "tol_required_mm": float(tol_req),
                "is_standard_tool": bool(is_standard_tool),
            },
            "realworld": {
                "lambda_min": float(rw_result.lambda_min),
                "scuff_margin_flash_C": float(rw_result.scuff_margin_flash_C),
                "scuff_margin_integral_C": float(rw_result.scuff_margin_integral_C),
                "scuff_margin_C": float(rw_result.scuff_margin_C),
                "micropitting_safety": float(rw_result.micropitting_safety),
                "material_temp_margin_C": float(rw_result.material_temp_margin_C),
                "life_damage": _jsonify(life_damage_diag),
                "tribology_method_used": str(rw_result.tribology_method_used),
                "tribology_data_status": str(rw_result.tribology_data_status),
                "tribology_provenance": _jsonify(getattr(rw_result, "tribology_provenance", {})),
            },
            "constraints": _jsonify(constraint_diag),
            "objectives": {
                "names": list(PRINCIPLES_OBJECTIVE_NAMES),
                "values": F.tolist(),
            },
        }
        return PrinciplesProxyResult(
            F=F,
            G=np.asarray(G, dtype=np.float64),
            diag=diag,
            objective_names=PRINCIPLES_OBJECTIVE_NAMES,
            constraint_names=tuple(str(rec["name"]) for rec in constraint_diag),
            x_full=np.asarray(x_full, dtype=np.float64),
            expansion_policy=expansion_policy,
        )
    except Exception as exc:
        return _proxy_failure(
            x_full=np.asarray(x_full, dtype=np.float64),
            expansion_policy=expansion_policy,
            error_signature=f"{type(exc).__name__}: {exc}",
        )


def evaluate_principles_alignment(
    x_full: np.ndarray,
    *,
    base_ctx: EvalContext,
    rpm: float,
    torque: float,
    fidelity: int,
    constraint_phase: str,
) -> PrinciplesAlignmentResult:
    ctx = _alignment_ctx(
        base_ctx,
        rpm=float(rpm),
        torque=float(torque),
        fidelity=int(fidelity),
        constraint_phase=str(constraint_phase),
    )
    try:
        res = evaluate_candidate(np.asarray(x_full, dtype=np.float64), ctx)
        diag = dict(res.diag or {})
        constraint_names = tuple(str(rec.get("name", "")) for rec in diag.get("constraints", []))
        objective_names = tuple(str(v) for v in (diag.get("objectives", {}) or {}).get("names", []))
        return PrinciplesAlignmentResult(
            F=np.asarray(res.F, dtype=np.float64),
            G=np.asarray(res.G, dtype=np.float64),
            diag=_jsonify(diag),
            objective_names=objective_names or PRINCIPLES_OBJECTIVE_NAMES,
            constraint_names=constraint_names,
        )
    except Exception as exc:
        return _alignment_failure(f"{type(exc).__name__}: {exc}", n_constr=17)


__all__ = [
    "PrinciplesAlignmentResult",
    "PrinciplesProxyResult",
    "evaluate_principles_alignment",
    "evaluate_principles_proxy",
]
