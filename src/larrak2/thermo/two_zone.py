"""Equation-first two-zone crank-angle thermo solver with strict validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.artifact_paths import DEFAULT_OPENFOAM_NN_ARTIFACT, assert_not_legacy_models_path
from ..core.constants import RATIO_SLOPE_LIMIT_FID0, RATIO_SLOPE_LIMIT_FID1
from ..core.encoding import ThermoParams
from ..core.types import BreathingConfig, EvalContext
from .combustion import burn_increment, double_wiebe_burn_fraction
from .constants import load_thermo_constants
from .scavenging import evaluate_rotary_scavenging
from .validation import (
    build_validation_report,
    compute_nn_disagreement,
    evaluate_trend_checks,
    in_validated_envelope,
    load_validation_manifest,
    validate_benchmark_agreement,
)
from .vaporization import step_vapor_fraction


@dataclass
class TwoZoneThermoResult:
    """Result from two-zone thermo solve."""

    efficiency: float
    requested_ratio_profile: np.ndarray
    G: np.ndarray
    diag: dict[str, Any] = field(default_factory=dict)


def _ratio_profile_stats(profile: np.ndarray) -> dict[str, float | bool]:
    arr = np.asarray(profile, dtype=np.float64)
    finite = bool(np.all(np.isfinite(arr)))
    slope = np.gradient(arr) if arr.size > 1 else np.array([0.0], dtype=np.float64)
    jerk = np.gradient(slope) if slope.size > 1 else np.array([0.0], dtype=np.float64)
    return {
        "finite": finite,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "max_slope": float(np.max(np.abs(slope))) if slope.size else 0.0,
        "max_jerk": float(np.max(np.abs(jerk))) if jerk.size else 0.0,
    }


def _newton_pressure(
    rhs: float,
    volume: float,
    p0: float,
    *,
    residual_tol: float,
    max_iter: int,
) -> tuple[float, int, bool]:
    """Solve p*V-rhs=0 by Newton iteration (explicitly enforcing solver contract)."""
    p_floor = 1e-6
    v = max(float(volume), 1e-12)
    p = max(float(p0), p_floor)
    for it in range(max(1, int(max_iter))):
        f = p * v - float(rhs)
        if abs(f) <= float(residual_tol):
            return max(p, p_floor), it + 1, True
        df = v
        p = p - f / max(df, 1e-12)
        p = max(p, p_floor)
    return max(p, p_floor), int(max_iter), False


def _build_theta_grid(theta_start: float) -> np.ndarray:
    base = np.arange(0.0, 360.0, 1.0, dtype=np.float64)
    window_lo = max(0.0, float(theta_start) - 30.0)
    window_hi = min(360.0, float(theta_start) + 40.0)
    fine = np.arange(window_lo, window_hi, 0.25, dtype=np.float64)
    return np.array(sorted(set(np.round(np.concatenate([base, fine]), 8))), dtype=np.float64)


def _phase_driven_volume(
    theta_deg: np.ndarray,
    *,
    compression_duration: float,
    expansion_duration: float,
    v_clearance: float,
    v_displaced: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta_deg, dtype=np.float64)
    V = np.zeros_like(theta)
    dV = np.zeros_like(theta)

    comp = max(float(compression_duration), 1e-6)
    expn = max(float(expansion_duration), 1e-6)

    for i, th in enumerate(theta):
        t = float(th) % 360.0
        if t <= expn:
            s = t / expn
            f = 0.5 * (1.0 - np.cos(np.pi * s))
            V[i] = v_clearance + f * v_displaced
            dV[i] = v_displaced * 0.5 * np.pi * np.sin(np.pi * s) / expn
        elif t <= (360.0 - comp):
            V[i] = v_clearance + v_displaced
            dV[i] = 0.0
        else:
            s = (t - (360.0 - comp)) / comp
            f = 0.5 * (1.0 - np.cos(np.pi * s))
            V[i] = v_clearance + (1.0 - f) * v_displaced
            dV[i] = -v_displaced * 0.5 * np.pi * np.sin(np.pi * s) / comp

    return V, dV


def _generate_ratio_profile(dV_dtheta: np.ndarray, base_ratio: float = 2.0) -> np.ndarray:
    dv = np.asarray(dV_dtheta, dtype=np.float64)
    amp = 0.3 * float(base_ratio)
    denom = max(float(np.max(np.abs(dv))), 1e-12)
    profile = float(base_ratio) + amp * (dv / denom)
    return np.maximum(profile, 0.3)


def _resample_periodic_profile(
    theta_deg: np.ndarray,
    values: np.ndarray,
    *,
    n_points: int = 360,
) -> np.ndarray:
    """Resample periodic profile onto fixed 0..360 grid for downstream modules."""
    theta = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if theta.size != vals.size:
        raise ValueError("theta/values length mismatch for periodic resampling")
    if theta.size == 0:
        return np.zeros(int(n_points), dtype=np.float64)

    theta_mod = np.mod(theta, 360.0)
    order = np.argsort(theta_mod)
    theta_s = theta_mod[order]
    vals_s = vals[order]

    theta_ext = np.concatenate([theta_s, [theta_s[0] + 360.0]])
    vals_ext = np.concatenate([vals_s, [vals_s[0]]])

    query = np.linspace(0.0, 360.0, int(n_points), endpoint=False, dtype=np.float64)
    return np.interp(query, theta_ext, vals_ext)


def _resolve_openfoam_surrogate_path(ctx: EvalContext) -> Path:
    model_path = str(ctx.openfoam_model_path).strip() if ctx.openfoam_model_path else ""
    if not model_path:
        env_path = str(os.environ.get("LARRAK2_OPENFOAM_NN_PATH", "")).strip()
        model_path = env_path if env_path else str(DEFAULT_OPENFOAM_NN_ARTIFACT)
    model_path = str(assert_not_legacy_models_path(model_path, purpose="OpenFOAM NN artifact"))
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"OpenFOAM NN surrogate not found at '{p}' for fidelity=2 thermo benchmark path"
        )
    return p


def _predict_openfoam_breathing(
    *,
    params: ThermoParams,
    ctx: EvalContext,
    breathing: BreathingConfig,
) -> dict[str, float]:
    model_path = _resolve_openfoam_surrogate_path(ctx)
    from ..surrogate.openfoam_nn import get_openfoam_surrogate

    surrogate = get_openfoam_surrogate(
        model_path,
        validation_mode=str(getattr(ctx, "surrogate_validation_mode", "strict")),
    )
    pred = surrogate.predict_one(
        {
            "rpm": float(ctx.rpm),
            "torque": float(ctx.torque),
            "lambda_af": float(params.lambda_af),
            "bore_mm": float(breathing.bore_mm),
            "stroke_mm": float(breathing.stroke_mm),
            "intake_port_area_m2": float(breathing.intake_port_area_m2),
            "exhaust_port_area_m2": float(breathing.exhaust_port_area_m2),
            "p_manifold_Pa": float(breathing.p_manifold_Pa),
            "p_back_Pa": float(breathing.p_back_Pa),
            "overlap_deg": float(breathing.overlap_deg),
            "intake_open_deg": float(breathing.intake_open_deg),
            "intake_close_deg": float(breathing.intake_close_deg),
            "exhaust_open_deg": float(breathing.exhaust_open_deg),
            "exhaust_close_deg": float(breathing.exhaust_close_deg),
        }
    )
    return {k: float(v) for k, v in pred.items()}


def _hybrid_correct(
    *,
    x_eq: float,
    x_nn: float,
    k: float,
    beta: float,
) -> float:
    denom = max(abs(float(x_eq)), 1e-12)
    delta = (float(x_nn) - float(x_eq)) / denom
    corr = np.clip(float(k) * delta, -float(beta), float(beta))
    return float(x_eq * (1.0 + corr))


def _compute_power_balance_constraint(work_j: float, rpm: float, torque_nm: float) -> float:
    omega = float(rpm) * 2.0 * np.pi / 60.0
    p_out_req = float(torque_nm) * omega
    p_indicated = max(float(work_j), 0.0) * float(rpm) / 60.0
    return float(p_out_req - p_indicated)


def evaluate_two_zone_thermo(
    params: ThermoParams,
    ctx: EvalContext,
    *,
    ratio_slope_limit: float | None = None,
) -> TwoZoneThermoResult:
    """Evaluate thermo cycle using two-zone equation-first model."""
    constants = load_thermo_constants(
        constants_path=ctx.thermo_constants_path,
        manifest_path=None,
    )

    breathing = ctx.breathing or BreathingConfig()
    theta = _build_theta_grid(float(params.heat_release_center))

    bore_m = float(breathing.bore_mm) / 1000.0
    stroke_m = float(breathing.stroke_mm) / 1000.0
    area = np.pi * (0.5 * bore_m) ** 2
    v_disp = area * stroke_m
    compression_ratio = float(getattr(breathing, "compression_ratio", 10.0))
    if compression_ratio <= 1.0:
        raise ValueError(
            f"BreathingConfig.compression_ratio must be > 1.0, got {compression_ratio}"
        )
    v_clear = v_disp / max(compression_ratio - 1.0, 1e-6)
    V, dV_dtheta = _phase_driven_volume(
        theta,
        compression_duration=float(params.compression_duration),
        expansion_duration=float(params.expansion_duration),
        v_clearance=v_clear,
        v_displaced=v_disp,
    )

    p_guess = np.maximum(
        float(breathing.p_manifold_Pa) * (V.max() / np.maximum(V, 1e-12)) ** constants.gamma_u, 1.0
    )
    t_guess = np.full_like(theta, float(constants.t_intake_k), dtype=np.float64)

    m0_guess = (
        float(breathing.p_manifold_Pa)
        * float(np.max(V))
        / (max(constants.r_u * constants.t_intake_k, 1e-12))
    )

    scav_eq = evaluate_rotary_scavenging(
        theta_deg=theta,
        p_cyl=p_guess,
        t_cyl=t_guess,
        rpm=float(ctx.rpm),
        p_manifold_pa=float(breathing.p_manifold_Pa),
        p_back_pa=float(breathing.p_back_Pa),
        intake_open_deg=float(breathing.intake_open_deg),
        intake_close_deg=float(breathing.intake_close_deg),
        exhaust_open_deg=float(breathing.exhaust_open_deg),
        exhaust_close_deg=float(breathing.exhaust_close_deg),
        intake_port_area_m2=float(breathing.intake_port_area_m2),
        exhaust_port_area_m2=float(breathing.exhaust_port_area_m2),
        cd_intake=float(constants.cd_intake),
        cd_exhaust=float(constants.cd_exhaust),
        gamma=float(constants.gamma_u),
        r_specific=float(constants.r_u),
        m_initial=float(m0_guess),
    )

    m_air_eq = float(scav_eq.m_air_trapped)
    residual_eq = float(scav_eq.residual_fraction)
    scav_eff_eq = float(scav_eq.scavenging_efficiency)

    manifest = load_validation_manifest(ctx.thermo_anchor_manifest_path)
    if (
        int(ctx.fidelity) >= 2
        and str(getattr(ctx, "surrogate_validation_mode", "strict")) == "strict"
    ):
        anchors = manifest.get("anchors", [])
        if not anchors:
            raise RuntimeError(
                "Fidelity-2 thermo benchmark requires non-empty anchor manifest in strict mode."
            )
    in_env = in_validated_envelope(rpm=ctx.rpm, torque=ctx.torque, manifest=manifest)

    nn_disagreement = {
        "delta_m_air": 0.0,
        "delta_residual": 0.0,
        "delta_scavenging_eff": 0.0,
    }
    benchmark_ok = True
    benchmark_status = "not_applicable"
    benchmark_messages: list[str] = []
    openfoam_nn_used = False

    hybrid_correction_active = False
    if int(ctx.fidelity) >= 2:
        pred = _predict_openfoam_breathing(params=params, ctx=ctx, breathing=breathing)
        openfoam_nn_used = True
        nn_disagreement = compute_nn_disagreement(
            m_air_eq=m_air_eq,
            m_air_nn=float(pred.get("m_air_trapped", m_air_eq)),
            residual_eq=residual_eq,
            residual_nn=float(pred.get("residual_fraction", residual_eq)),
            scavenging_eq=scav_eff_eq,
            scavenging_nn=float(pred.get("scavenging_efficiency", scav_eff_eq)),
        )

        benchmark_ok, benchmark_status, benchmark_messages = validate_benchmark_agreement(
            disagreement=nn_disagreement,
            thresholds=manifest.get("thresholds", {}),
            in_envelope=in_env,
        )

        if in_env:
            hybrid_correction_active = True
            m_air_used = _hybrid_correct(
                x_eq=m_air_eq,
                x_nn=float(pred.get("m_air_trapped", m_air_eq)),
                k=float(constants.nn_correction_k),
                beta=float(constants.nn_correction_beta),
            )
            residual_used = _hybrid_correct(
                x_eq=max(residual_eq, 1e-9),
                x_nn=max(float(pred.get("residual_fraction", residual_eq)), 1e-9),
                k=float(constants.nn_correction_k),
                beta=float(constants.nn_correction_beta),
            )
            scav_eff_used = _hybrid_correct(
                x_eq=max(scav_eff_eq, 1e-9),
                x_nn=max(float(pred.get("scavenging_efficiency", scav_eff_eq)), 1e-9),
                k=float(constants.nn_correction_k),
                beta=float(constants.nn_correction_beta),
            )
        else:
            benchmark_messages.append("hybrid correction disabled outside validated envelope")
            m_air_used = m_air_eq
            residual_used = residual_eq
            scav_eff_used = scav_eff_eq
    else:
        m_air_used = m_air_eq
        residual_used = residual_eq
        scav_eff_used = scav_eff_eq

    # Trend checks from local perturbation experiments.
    scav_intake_up = evaluate_rotary_scavenging(
        theta_deg=theta,
        p_cyl=p_guess,
        t_cyl=t_guess,
        rpm=float(ctx.rpm),
        p_manifold_pa=float(breathing.p_manifold_Pa),
        p_back_pa=float(breathing.p_back_Pa),
        intake_open_deg=float(breathing.intake_open_deg),
        intake_close_deg=float(breathing.intake_close_deg),
        exhaust_open_deg=float(breathing.exhaust_open_deg),
        exhaust_close_deg=float(breathing.exhaust_close_deg),
        intake_port_area_m2=float(breathing.intake_port_area_m2) * 1.05,
        exhaust_port_area_m2=float(breathing.exhaust_port_area_m2),
        cd_intake=float(constants.cd_intake),
        cd_exhaust=float(constants.cd_exhaust),
        gamma=float(constants.gamma_u),
        r_specific=float(constants.r_u),
        m_initial=float(m0_guess),
    )
    scav_backpressure_up = evaluate_rotary_scavenging(
        theta_deg=theta,
        p_cyl=p_guess,
        t_cyl=t_guess,
        rpm=float(ctx.rpm),
        p_manifold_pa=float(breathing.p_manifold_Pa),
        p_back_pa=float(breathing.p_back_Pa) * 1.05,
        intake_open_deg=float(breathing.intake_open_deg),
        intake_close_deg=float(breathing.intake_close_deg),
        exhaust_open_deg=float(breathing.exhaust_open_deg),
        exhaust_close_deg=float(breathing.exhaust_close_deg),
        intake_port_area_m2=float(breathing.intake_port_area_m2),
        exhaust_port_area_m2=float(breathing.exhaust_port_area_m2),
        cd_intake=float(constants.cd_intake),
        cd_exhaust=float(constants.cd_exhaust),
        gamma=float(constants.gamma_u),
        r_specific=float(constants.r_u),
        m_initial=float(m0_guess),
    )

    lam = max(float(params.lambda_af), 1e-6)
    m_fuel = float(m_air_used / (lam * constants.afr_stoich))
    q_chem = m_fuel * float(constants.fuel_lhv)
    trapped_o2 = (
        float(m_air_used)
        * float(constants.o2_mass_fraction_air)
        * max(1.0 - float(residual_used), 0.0)
    )
    burn_cap = float(
        np.clip(
            trapped_o2 / max(m_fuel * float(constants.o2_required_per_fuel), 1e-12),
            0.0,
            1.0,
        )
    )

    lam_richer = max(0.7, lam * 0.95)
    m_fuel_richer = float(m_air_used / (lam_richer * constants.afr_stoich))
    burn_cap_richer = float(
        np.clip(
            trapped_o2 / max(m_fuel_richer * float(constants.o2_required_per_fuel), 1e-12),
            0.0,
            1.0,
        )
    )

    trend_checks = evaluate_trend_checks(
        trapped_mass_base=m_air_eq,
        trapped_mass_intake_up=float(scav_intake_up.m_air_trapped),
        scavenging_base=scav_eff_eq,
        scavenging_backpressure_up=float(scav_backpressure_up.scavenging_efficiency),
        burn_cap_base=burn_cap,
        burn_cap_richer=burn_cap_richer,
    )

    # Two-zone states.
    n = theta.size
    m_u = np.zeros(n, dtype=np.float64)
    m_b = np.zeros(n, dtype=np.float64)
    t_u = np.zeros(n, dtype=np.float64)
    t_b = np.zeros(n, dtype=np.float64)
    p = np.zeros(n, dtype=np.float64)
    vapor = np.zeros(n, dtype=np.float64)
    q_rel_step = np.zeros(n, dtype=np.float64)
    q_wall_u = np.zeros(n, dtype=np.float64)
    q_wall_b = np.zeros(n, dtype=np.float64)
    newton_iters = np.zeros(n, dtype=np.int64)

    m_u[0] = max(float(m_air_used + m_fuel), 1e-9)
    m_b[0] = 1e-9
    t_u[0] = (
        float(constants.t_intake_k) * (1.0 - residual_used)
        + float(constants.t_residual_k) * residual_used
    )
    t_b[0] = t_u[0] + 100.0
    p[0], newton_iters[0], converged = _newton_pressure(
        m_u[0] * constants.r_u * t_u[0] + m_b[0] * constants.r_b * t_b[0],
        V[0],
        p0=float(breathing.p_manifold_Pa),
        residual_tol=1e-7,
        max_iter=20,
    )
    if not converged:
        raise RuntimeError("Thermo pressure closure failed at initial state")

    vapor[0] = 0.2

    burn_profile = double_wiebe_burn_fraction(
        theta,
        theta_start=float(params.heat_release_center),
        duration=max(1e-6, float(params.heat_release_width)),
        split=float(constants.wiebe_split),
        a1=float(constants.wiebe_a1),
        m1=float(constants.wiebe_m1),
        a2=float(constants.wiebe_a2),
        m2=float(constants.wiebe_m2),
    )

    dtheta = np.diff(theta)
    dtheta = np.concatenate([dtheta, [max(1e-6, 360.0 - (theta[-1] - theta[0]))]])
    dt = dtheta / max(6.0 * float(ctx.rpm), 1e-9)

    # Pre-compute burn increments from oxygen limit; vapor will gate dynamically in-loop.
    dx_base = burn_increment(burn_profile, oxygen_completion_cap=burn_cap, vapor_fraction=1.0)

    for i in range(1, n):
        vapor[i] = step_vapor_fraction(
            float(vapor[i - 1]),
            dt_s=float(dt[i - 1]),
            temp_k=float(t_u[i - 1]),
            pressure_pa=float(p[i - 1]),
            tau_ref_s=float(constants.vapor_tau_ref_s),
            temp_exponent=float(constants.vapor_temp_exponent),
            pressure_exponent=float(constants.vapor_pressure_exponent),
        )

        dm_burn = max(0.0, m_fuel * dx_base[i] * vapor[i])
        dm_burn = min(dm_burn, max(m_u[i - 1] - 1e-9, 0.0))

        xb_mid = float(np.clip(burn_profile[i], 0.0, 1.0))
        dV = float(V[i] - V[i - 1])
        p_prev = float(p[i - 1])

        u_u_prev = m_u[i - 1] * constants.cv_u * t_u[i - 1]
        u_b_prev = m_b[i - 1] * constants.cv_b * t_b[i - 1]

        q_comb = float(dm_burn * constants.fuel_lhv)
        q_rel_step[i] = q_comb

        # Simple wall transfer based on bore area proxy.
        a_wall = max(1e-8, np.pi * bore_m * stroke_m)
        qwu_raw = float(
            constants.h_wall_u * a_wall * max(t_u[i - 1] - constants.wall_temp_k, 0.0) * dt[i - 1]
        )
        qwb_raw = float(
            constants.h_wall_b * a_wall * max(t_b[i - 1] - constants.wall_temp_k, 0.0) * dt[i - 1]
        )
        # Prevent unphysical single-step energy extraction at low speed/load.
        qwu = min(qwu_raw, 0.9 * max(u_u_prev, 0.0))
        qwb = min(qwb_raw, 0.9 * max(u_b_prev + q_comb, 0.0))
        q_wall_u[i] = qwu
        q_wall_b[i] = qwb

        u_u = u_u_prev - p_prev * dV * max(1.0 - xb_mid, 0.0) - qwu
        u_b = u_b_prev - p_prev * dV * max(xb_mid, 0.0) + q_comb - qwb

        m_u[i] = max(m_u[i - 1] - dm_burn, 1e-9)
        m_b[i] = max(m_b[i - 1] + dm_burn, 1e-9)

        t_u[i] = max(u_u / max(m_u[i] * constants.cv_u, 1e-9), 120.0)
        t_b[i] = max(u_b / max(m_b[i] * constants.cv_b, 1e-9), 120.0)

        rhs = m_u[i] * constants.r_u * t_u[i] + m_b[i] * constants.r_b * t_b[i]
        p[i], newton_iters[i], converged = _newton_pressure(
            rhs,
            float(V[i]),
            p0=float(p_prev),
            residual_tol=1e-7,
            max_iter=20,
        )
        if not converged:
            raise RuntimeError(f"Thermo pressure closure failed at theta index {i}")

    # Integrals and efficiencies.
    work = -float(np.sum(0.5 * (p[:-1] + p[1:]) * np.diff(V)))
    q_rel = float(np.sum(q_rel_step))
    q_wall_total = float(np.sum(q_wall_u + q_wall_b))
    efficiency = float(np.clip(work / max(q_rel, 1e-9), 0.0, constants.efficiency_upper_bound))

    # Mass and energy residual checks.
    # This two-zone solve starts at trapped state after gas-exchange closure,
    # so mass should remain conserved inside the closed combustion loop.
    m_expected_end = m_u[0] + m_b[0]
    m_end = m_u[-1] + m_b[-1]
    mass_residual = abs(m_end - m_expected_end) / max(abs(m_expected_end), 1e-9)

    u0 = m_u[0] * constants.cv_u * t_u[0] + m_b[0] * constants.cv_b * t_b[0]
    u_end = m_u[-1] * constants.cv_u * t_u[-1] + m_b[-1] * constants.cv_b * t_b[-1]
    energy_balance = (u_end - u0) - (q_rel - work - q_wall_total)
    energy_residual = abs(float(energy_balance)) / max(
        abs(q_rel) + abs(work) + abs(q_wall_total), 1.0
    )

    non_negative_ok = bool(
        np.all(m_u > 0.0)
        and np.all(m_b > 0.0)
        and np.all(t_u > 0.0)
        and np.all(t_b > 0.0)
        and np.all(p > 0.0)
    )
    monotonic_burn_ok = bool(np.all(np.diff(np.maximum.accumulate(burn_profile)) >= -1e-12))

    validation_report = build_validation_report(
        mass_residual=float(mass_residual),
        energy_residual=float(energy_residual),
        non_negative_states_ok=non_negative_ok,
        monotonic_burn_ok=monotonic_burn_ok,
        choked_branch_continuity_error=float(scav_eq.branch_continuity_error),
        benchmark_status=benchmark_status,
        benchmark_ok=benchmark_ok,
        in_validated_envelope_flag=in_env,
        trend_checks=trend_checks,
        nn_disagreement=nn_disagreement,
        messages=benchmark_messages,
    )

    if not validation_report.passed(mass_tol=1e-4, energy_tol=3e-2, branch_tol=2e-2):
        raise RuntimeError(
            "Thermo validation failed: "
            f"mass_residual={validation_report.mass_residual:.3e}, "
            f"energy_residual={validation_report.energy_residual:.3e}, "
            f"benchmark_status={validation_report.benchmark_status}, "
            f"trends={validation_report.trend_checks}"
        )

    requested_ratio_profile = _resample_periodic_profile(
        theta,
        _generate_ratio_profile(
            dV_dtheta,
            base_ratio=2.0 + 0.5 * (compression_ratio - 10.0) / 10.0,
        ),
        n_points=360,
    )
    ratio_stats = _ratio_profile_stats(requested_ratio_profile)
    static_limit = RATIO_SLOPE_LIMIT_FID1 if int(ctx.fidelity) >= 1 else RATIO_SLOPE_LIMIT_FID0
    slope_limit = (
        min(static_limit, float(ratio_slope_limit))
        if ratio_slope_limit is not None
        else static_limit
    )

    p_max = float(np.max(p))
    p_limit_pa = float(constants.p_limit_bar) * 1e5

    G = np.array(
        [
            0.0 - efficiency,
            efficiency - float(constants.efficiency_upper_bound),
            (p_max - p_limit_pa) / max(p_limit_pa, 1e-12),
            float(ratio_stats.get("max_slope", 0.0)) - float(slope_limit),
            _compute_power_balance_constraint(work, ctx.rpm, ctx.torque),
        ],
        dtype=np.float64,
    )

    return TwoZoneThermoResult(
        efficiency=efficiency,
        requested_ratio_profile=requested_ratio_profile,
        G=G,
        diag={
            "theta_deg": theta,
            "p": p,
            "V": V,
            "T_u": t_u,
            "T_b": t_b,
            "m_u": m_u,
            "m_b": m_b,
            "vapor_fraction": vapor,
            "burn_fraction": burn_profile,
            "q_rel_step": q_rel_step,
            "q_wall_u": q_wall_u,
            "q_wall_b": q_wall_b,
            "W": float(work),
            "Q_chem": float(q_chem),
            "Q_rel": float(q_rel),
            "m_air_trapped": float(m_air_used),
            "residual_fraction": float(residual_used),
            "scavenging_efficiency": float(scav_eff_used),
            "trapped_o2_mass": float(trapped_o2),
            "burn_frac_o2": float(burn_cap),
            "openfoam_nn_used": bool(openfoam_nn_used),
            "ratio_profile_stats": ratio_stats,
            "ratio_slope_limit_used": float(slope_limit),
            "ratio_slope_limit_source": "manufacturability"
            if ratio_slope_limit is not None
            else "static",
            "T_mean_wall": float(constants.wall_temp_k),
            "T_wall_C": float(constants.wall_temp_k - 273.15),
            "thermo_solver_status": "ok",
            "thermo_model_version": "two_zone_eq_v1",
            "thermo_constants_version": constants.version,
            "thermo_mass_residual": float(validation_report.mass_residual),
            "thermo_energy_residual": float(validation_report.energy_residual),
            "thermo_benchmark_status": validation_report.benchmark_status,
            "thermo_nn_disagreement": validation_report.nn_disagreement,
            "thermo_validation_messages": validation_report.messages,
            "thermo_trend_checks": validation_report.trend_checks,
            "in_validated_envelope": bool(validation_report.in_validated_envelope),
            "thermo_hybrid_correction_active": bool(hybrid_correction_active),
            "branch_continuity_error": float(scav_eq.branch_continuity_error),
            "newton_iter_max": int(np.max(newton_iters)),
            # Compatibility field retained for legacy tests/consumers.
            "v1_port": bool(int(ctx.fidelity) >= 1),
            "phase_driven": True,
        },
    )
