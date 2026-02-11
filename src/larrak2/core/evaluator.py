"""Candidate evaluation — THE canonical interface.

This is the ONLY interface between physics and optimizers.
No optimizer-specific code should exist in physics modules.

Interface:
    evaluate_candidate(x, ctx) -> EvalResult(F, G, diag)

Flow:
    1. decode_candidate(x) -> Candidate
    2. eval_thermo(params.thermo, ctx) -> ThermoResult
    3. eval_gear(params.gear, i_req_profile, ctx) -> GearResult
    4. Assemble F (objectives) and G (constraints)
    5. Return EvalResult with diagnostics
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from ..core.constants import MODEL_VERSION_GEAR_V1, MODEL_VERSION_THERMO_V1
from ..core.constraints import (
    GEAR_CONSTRAINTS,
    THERMO_CONSTRAINTS_FID0,
    THERMO_CONSTRAINTS_FID1,
    combine_constraints,
)
from ..gear.litvin_core import eval_gear
from ..gear.manufacturability_limits import (
    ManufacturingProcessParams,
    compute_manufacturable_ratio_rate_limits,
)
from ..thermo.motionlaw import eval_thermo
from .encoding import decode_candidate
from .types import EvalContext, EvalResult


def evaluate_candidate(x: np.ndarray, ctx: EvalContext) -> EvalResult:
    """Evaluate candidate solution.

    THE canonical interface between physics and optimizers.

    Args:
        x: Flat decision vector (length N_TOTAL).
        ctx: Evaluation context with rpm, torque, fidelity, seed.

    Returns:
        EvalResult with:
            F: Objectives [efficiency (negated), loss, max_planet_radius]
            G: Constraints (G <= 0 feasible)
            diag: Diagnostics dict
    """
    t0 = time.perf_counter()

    # Decode candidate
    candidate = decode_candidate(x)

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

    # Thermo evaluation
    t_thermo_start = time.perf_counter()
    thermo_result = eval_thermo(candidate.thermo, ctx, ratio_slope_limit=dynamic_slope_limit)
    t_thermo = time.perf_counter() - t_thermo_start

    # Gear evaluation (uses ratio profile from thermo)
    t_gear_start = time.perf_counter()
    gear_result = eval_gear(candidate.gear, thermo_result.requested_ratio_profile, ctx)
    t_gear = time.perf_counter() - t_gear_start

    # Objectives (minimize) are the three efficiencies:
    # - F[0] = 1 - η_comb  (chemical -> released heat)
    # - F[1] = 1 - η_exp   (released heat -> work)
    # - F[2] = 1 - η_gear  (piston/mech power -> output power)

    thermo_diag = thermo_result.diag or {}
    Q_chem = float(thermo_diag.get("Q_chem", 0.0))
    Q_rel = float(thermo_diag.get("Q_rel", 0.0))
    W = float(thermo_diag.get("W", 0.0))

    eta_comb = Q_rel / Q_chem if Q_chem > 0 else 0.0
    eta_exp = W / Q_rel if Q_rel > 0 else float(thermo_result.efficiency)

    eta_comb = float(np.clip(eta_comb, 0.0, 1.0))
    eta_exp = float(np.clip(eta_exp, 0.0, 1.0))

    # Fidelity 2+: Apply surrogate corrections (legacy residual surrogates)
    delta_eff = 0.0
    delta_loss = 0.0
    loss_corrected = gear_result.loss_total
    surrogate_meta: dict[str, object] = {"surrogate_used": False}

    # Legacy residual surrogates (sklearn) are disabled when OpenFOAM NN is active,
    # since their semantics don't match the new efficiency decomposition.
    enable_residual = ctx.fidelity >= 2 and not bool(thermo_diag.get("openfoam_nn_used", False))

    if enable_residual:
        from larrak2.surrogate.inference import get_surrogate_engine

        engine = get_surrogate_engine()
        delta_eff, delta_loss, meta = engine.predict_corrections(x)

        # Interpret delta_eff as an additive correction to η_exp (bounded)
        eta_exp = float(np.clip(eta_exp + float(delta_eff), 0.0, 1.0))

        # Gear loss correction directly impacts η_gear
        loss_corrected = float(loss_corrected + float(delta_loss))

        surrogate_meta = {
            "surrogate_used": True,
            "delta_eff": float(delta_eff),
            "delta_loss": float(delta_loss),
            "version_surrogate": "SurrogateEngine_v1",
            "active_models": meta.get("surrogates_active", []),
            "uncertainty": meta.get("uncertainty", {}),
        }

    omega = float(ctx.rpm) * 2.0 * np.pi / 60.0
    P_out = float(ctx.torque) * omega
    denom = max(P_out + loss_corrected, 1e-12)
    eta_gear = float(np.clip(P_out / denom, 0.0, 1.0)) if P_out > 0 else 1.0

    F = np.array([1.0 - eta_comb, 1.0 - eta_exp, 1.0 - eta_gear], dtype=np.float64)

    # Constraints (G <= 0 feasible) with centralized scaling/naming
    thermo_names = THERMO_CONSTRAINTS_FID1 if ctx.fidelity >= 1 else THERMO_CONSTRAINTS_FID0
    gear_names = GEAR_CONSTRAINTS

    thermo_G_values: list[float] = list(np.asarray(thermo_result.G, dtype=float))
    power_balance_raw = 0.0

    if ctx.fidelity >= 1:
        # Operating-point demand constraint:
        # indicated power must meet demanded output power plus gear loss.
        # For fidelity=2 (OpenFOAM NN), enforce unconditionally.
        # For fidelity=1, keep permissive behavior unless NN is explicitly active.
        if ctx.fidelity >= 2 or bool(thermo_diag.get("openfoam_nn_used", False)):
            cycles_per_sec = float(ctx.rpm) / 60.0
            W_for_balance = float(thermo_diag.get("W", thermo_diag.get("w_indicated", 0.0)))
            P_indicated = max(W_for_balance, 0.0) * max(cycles_per_sec, 0.0)
            P_required = max(P_out, 0.0) + max(float(loss_corrected), 0.0)
            power_balance_raw = float(P_required - P_indicated)  # <= 0 feasible
        else:
            power_balance_raw = 0.0
        thermo_G_values.append(power_balance_raw)

    G, constraint_diag = combine_constraints(
        thermo_G_values,
        list(np.asarray(gear_result.G, dtype=float)),
        thermo_names,
        gear_names,
    )

    # Feasibility based only on hard constraints
    hard_mask = np.array([c["kind"] == "hard" for c in constraint_diag], dtype=bool)
    G_hard = G[hard_mask]

    # Diagnostics
    t_total = time.perf_counter() - t0
    diag = {
        "thermo": thermo_result.diag,
        "gear": gear_result.diag,
        "timings": {
            "total_ms": t_total * 1000,
            "thermo_ms": t_thermo * 1000,
            "gear_ms": t_gear * 1000,
        },
        "metrics": {
            "eta_comb": eta_comb,
            "eta_exp": eta_exp,
            "eta_gear": eta_gear,
            "eta_therm": float(np.clip(eta_comb * eta_exp, 0.0, 1.0)),
            "eta_total": float(np.clip(eta_comb * eta_exp * eta_gear, 0.0, 1.0)),
            "efficiency_raw": thermo_result.efficiency,
            "ratio_error_mean": gear_result.ratio_error_mean,
            "ratio_error_max": gear_result.ratio_error_max,
            "max_planet_radius": gear_result.max_planet_radius,
            "loss_total": float(loss_corrected),
            "loss_raw": gear_result.loss_total,
            "power_out": P_out,
            "power_balance_raw": power_balance_raw,
        },
        "versions": {
            "thermo_v1": MODEL_VERSION_THERMO_V1,
            "gear_v1": MODEL_VERSION_GEAR_V1,
            **surrogate_meta,
        },
        "constraints": constraint_diag,
        "feasible_hard": bool(np.all(G_hard <= 0)) if len(G_hard) else True,
        "manufacturability_limits": {
            "durations_deg": rate_env.duration_deg,
            "max_delta_ratio": rate_env.max_delta_ratio,
            "max_ratio_slope": rate_env.max_ratio_slope,
            "applied_ratio_slope_limit": dynamic_slope_limit,
            "process": {
                "kerf_mm": process_cfg.kerf_mm,
                "overcut_mm": process_cfg.overcut_mm,
                "min_ligament_mm": process_cfg.min_ligament_mm,
                "min_feature_radius_mm": process_cfg.min_feature_radius_mm,
                "max_pressure_angle_deg": process_cfg.max_pressure_angle_deg,
            },
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
