"""Litvin-style gear evaluation with ratio tracking and losses.

This module implements toy gear physics for initial testing.
The API is stable; only the internals will get more realistic later.

Outputs:
    - ratio_error_mean: mean absolute ratio tracking error
    - ratio_error_max: maximum ratio tracking error
    - max_planet_radius: maximum planet pitch radius (mm)
    - loss_total: total mesh friction loss (W)
    - G_gear: constraint array (all G <= 0 feasible)
    - diag: diagnostics dict with profiles
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.artifact_paths import (
    DEFAULT_CALCULIX_NN_ARTIFACT,
    DEFAULT_GEAR_LOSS_NN_DIR,
    assert_not_legacy_models_path,
)
from ..core.constants import GEAR_INTERFERENCE_CLEARANCE_MM, GEAR_MIN_THICKNESS_MM
from ..core.encoding import GearParams
from ..core.types import EvalContext
from ..ports.larrak_v1.gear_forward import litvin_synthesize
from ..surrogate.gear_loss_net import get_gear_surrogate
from .energy_ledger import EnergyLedger
from .loss_model import total_loss
from .pitchcurve import fourier_pitch_curve

# Number of discretization points
N_POINTS = 360

# Physical constants for toy model
MU_MESH = 0.05  # Friction coefficient
SPEED_SCALE = 1.0  # m/s per rad/s (toy)

# Constraint limits
RATIO_ERROR_TOL = 5.0  # 500% max ratio error (relaxed for toy physics)
MIN_RADIUS = 5.0  # mm
MAX_RADIUS = 100.0  # mm
MAX_CURVATURE = 0.5  # 1/mm
STRESS_HOTSPOT_LIMIT_MPA = 1600.0


@dataclass
class GearResult:
    """Result from gear evaluation."""

    ratio_error_mean: float
    ratio_error_max: float
    max_planet_radius: float
    loss_total: float
    loss_total: float
    G: np.ndarray
    diag: dict[str, Any] = field(default_factory=dict)
    ledger: EnergyLedger | None = None


def _compute_ratio_error(
    r_actual: np.ndarray,
    i_req_profile: np.ndarray,
    r_ring: float,
) -> tuple[float, float]:
    """Compute ratio tracking error.

    Args:
        r_actual: Actual planet radius profile.
        i_req_profile: Required ratio profile.
        r_ring: Ring gear radius.

    Returns:
        (mean_error, max_error) tuple.
    """
    # Ratio = r_ring / r_planet (toy model)
    ratio_actual = r_ring / np.maximum(r_actual, 1e-6)
    error = np.abs(ratio_actual - i_req_profile) / np.maximum(i_req_profile, 1e-6)

    return float(np.mean(error)), float(np.max(error))


def _compute_mesh_loss(
    theta: np.ndarray,
    r_planet: np.ndarray,
    omega_rpm: float,
    torque: float,
    mu: float = MU_MESH,
) -> tuple[float, np.ndarray]:
    """Compute mesh friction loss using toy model.

    P_loss ~ mean(μ * Fn * v_sliding)

    Args:
        theta: Angle array.
        r_planet: Planet radius profile (mm).
        omega_rpm: Rotational speed (rpm).
        torque: Torque (Nm).
        mu: Friction coefficient.

    Returns:
        (total_loss_watts, loss_profile) tuple.
    """
    omega_rad = omega_rpm * 2 * np.pi / 60  # rad/s

    # Toy model: loss varies with radius variation (sliding velocity depends on radius)
    # P_loss ~ mu * Fn * v_slide where v_slide ~ omega * dr/dtheta (radius change rate)
    r_m = r_planet / 1000  # mm to m
    dr = np.gradient(r_m)  # radius change rate

    # Normal force from torque, sliding from radius change
    fn = torque / np.maximum(np.mean(r_m), 1e-6)  # N (average)
    v_sliding = omega_rad * np.abs(dr) * 100 + omega_rad * r_m * 0.01  # m/s (toy)

    loss_profile = mu * fn * np.abs(v_sliding)  # W
    total_loss = float(np.mean(loss_profile)) + omega_rad * torque * 0.001  # base loss

    # Enforce non-negativity
    loss_profile = np.maximum(loss_profile, 0.0)
    total_loss = max(total_loss, 0.0)

    return total_loss, loss_profile


def _compute_curvature(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute curvature of pitch curve (simplified).

    Args:
        r: Radius profile.
        theta: Angle array.

    Returns:
        Curvature array (1/mm).
    """
    dr = np.gradient(r)
    ddr = np.gradient(dr)

    # Simplified curvature formula for polar curve
    # κ = |r² + 2(dr)² - r*d²r| / (r² + (dr)²)^(3/2)
    r_safe = np.maximum(r, 1e-6)
    num = np.abs(r_safe**2 + 2 * dr**2 - r_safe * ddr)
    denom = (r_safe**2 + dr**2) ** 1.5
    denom = np.maximum(denom, 1e-10)

    return num / denom


def _mean_pressure_angle_deg(gear_diag: dict[str, Any]) -> float:
    phi = gear_diag.get("pressure_angle_rad")
    if phi is None:
        return 20.0
    arr = np.asarray(phi, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 20.0
    return float(np.clip(np.degrees(np.mean(finite)), 10.0, 40.0))


def _evaluate_radius_strategy_stress(
    *,
    params: GearParams,
    i_req_profile: np.ndarray,
    ctx: EvalContext,
    gear_diag: dict[str, Any],
) -> dict[str, Any]:
    """Assess planet/ring/shared radius adaptation strategies via stress risk.

    The selected strategy remains a single choice, but diagnostics are expanded so
    the optimizer and downselect stages can use explicit advisory reasons.
    """

    def _analytical_stress_proxy(
        *,
        hertz_base_mpa: float,
        planet_mean_mm: float,
        planet_mm: float,
        ring_delta: float,
        planet_delta: float,
        ratio_residual: float,
    ) -> float:
        return float(
            hertz_base_mpa
            * (planet_mean_mm / max(planet_mm, 1e-9))
            * (1.0 + 0.45 * ring_delta + 0.25 * planet_delta + 0.9 * ratio_residual)
        )

    r_planet = np.asarray(gear_diag.get("r_planet", [params.base_radius]), dtype=np.float64)
    planet_mean = float(np.mean(np.maximum(r_planet, 1e-6)))
    ring_nominal = float(gear_diag.get("r_ring", 80.0))
    ratio_target = float(np.mean(np.maximum(np.asarray(i_req_profile, dtype=np.float64), 1e-6)))

    baseline_ratio = ring_nominal / max(planet_mean, 1e-6)
    ratio_factor = ratio_target / max(baseline_ratio, 1e-6)
    shared_factor = float(np.sqrt(max(ratio_factor, 1e-9)))

    strategies = [
        {"name": "planet_only", "ring_mm": ring_nominal, "planet_mm": ring_nominal / ratio_target},
        {"name": "ring_only", "ring_mm": ratio_target * planet_mean, "planet_mm": planet_mean},
        {
            "name": "shared_ring_planet",
            "ring_mm": ring_nominal * shared_factor,
            "planet_mm": planet_mean / max(shared_factor, 1e-9),
        },
    ]

    hertz_base = float(gear_diag.get("hertz_stress_max", 1200.0))
    p_angle_deg = _mean_pressure_angle_deg(gear_diag)
    calc_mode = str(getattr(ctx, "calculix_stress_mode", "nn"))
    ctx_model_path = getattr(ctx, "calculix_model_path", None)
    model_path_str = str(ctx_model_path).strip() if isinstance(ctx_model_path, str) else ""
    if not model_path_str:
        env_path = str(os.environ.get("LARRAK2_CALCULIX_NN_PATH", "")).strip()
        model_path_str = env_path if env_path else str(DEFAULT_CALCULIX_NN_ARTIFACT)
    model_path_str = str(
        assert_not_legacy_models_path(model_path_str, purpose="CalculiX NN artifact")
    )
    model_path = Path(model_path_str)
    calc_surrogate = None

    if calc_mode == "nn":
        if not model_path.exists():
            raise FileNotFoundError(
                "CalculiX NN surrogate is required but missing: "
                f"'{model_path}'. Run `larrak-run train-surrogates` first or run with "
                "--calculix-stress-mode analytical for an explicit physics-only bypass."
            )
        try:
            from ..surrogate.calculix_nn import get_calculix_surrogate

            calc_surrogate = get_calculix_surrogate(
                model_path,
                validation_mode=str(getattr(ctx, "surrogate_validation_mode", "strict")),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CalculiX NN surrogate from '{model_path}'."
            ) from e
    elif calc_mode != "analytical":
        raise ValueError(f"Unsupported calculix_stress_mode '{calc_mode}'")

    rows: list[dict[str, Any]] = []
    for row in strategies:
        ring_mm = float(max(row["ring_mm"], 1e-6))
        planet_mm = float(max(row["planet_mm"], 1e-6))
        produced_ratio = ring_mm / planet_mm
        ratio_residual = abs(produced_ratio - ratio_target) / max(ratio_target, 1e-9)
        ring_delta = abs(ring_mm - ring_nominal) / max(ring_nominal, 1e-9)
        planet_delta = abs(planet_mm - planet_mean) / max(planet_mean, 1e-9)
        radius_shift = ring_delta + planet_delta
        balance_delta = abs(ring_delta - planet_delta)

        if calc_surrogate is not None:
            module_mm = max(0.5, planet_mm / 20.0)
            features = {
                "rpm": float(ctx.rpm),
                "torque": float(ctx.torque),
                "base_radius_mm": planet_mm,
                "face_width_mm": float(params.face_width_mm),
                "module_mm": float(module_mm),
                "pressure_angle_deg": float(p_angle_deg),
                "helix_angle_deg": 0.0,
                "profile_shift": float(np.clip(ring_delta - planet_delta, -1.0, 1.0)),
            }
            try:
                stress_pred = float(calc_surrogate.predict_one(features)["max_stress"])
            except Exception as e:
                raise RuntimeError(
                    "CalculiX NN inference failed while evaluating radius strategies. "
                    f"Strategy='{row['name']}', features={features}"
                ) from e
            stress_source = "calculix_nn"
        else:
            stress_pred = _analytical_stress_proxy(
                hertz_base_mpa=hertz_base,
                planet_mean_mm=planet_mean,
                planet_mm=planet_mm,
                ring_delta=ring_delta,
                planet_delta=planet_delta,
                ratio_residual=ratio_residual,
            )
            stress_source = "analytical_proxy"

        stress_util = float(stress_pred / STRESS_HOTSPOT_LIMIT_MPA)
        score_components = {
            "stress_util": stress_util,
            "ratio_residual": float(ratio_residual),
            "radius_shift": float(radius_shift),
            "balance_delta": float(balance_delta),
        }
        score = float(
            6.0 * score_components["stress_util"]
            + 4.0 * score_components["ratio_residual"]
            + 1.2 * score_components["radius_shift"]
            + 0.8 * score_components["balance_delta"]
        )

        reasons: list[str] = []
        if stress_util > 1.2:
            reasons.append("stress exceeds hotspot limit by more than 20%")
        elif stress_util > 1.0:
            reasons.append("stress exceeds hotspot limit")
        elif stress_util > 0.9:
            reasons.append("stress is near hotspot limit")
        if ratio_residual > 0.05:
            reasons.append("ratio mismatch remains high")
        if radius_shift > 0.35:
            reasons.append("large radius migration required")
        if balance_delta > 0.2:
            reasons.append("unbalanced ring/planet adjustment increases concentration risk")
        if not reasons:
            reasons.append("balanced stress and ratio tracking margin")

        severity = "nominal"
        if stress_util > 1.2 or ratio_residual > 0.1:
            severity = "critical"
        elif stress_util > 1.0 or ratio_residual > 0.05:
            severity = "high"
        elif stress_util > 0.9 or radius_shift > 0.3:
            severity = "watch"

        rows.append(
            {
                "strategy": str(row["name"]),
                "ring_radius_mm": ring_mm,
                "planet_radius_mm": planet_mm,
                "ratio_target": ratio_target,
                "ratio_produced": produced_ratio,
                "ratio_residual": float(ratio_residual),
                "stress_pred_mpa": float(stress_pred),
                "stress_source": stress_source,
                "score_components": score_components,
                "score": score,
                "severity": severity,
                "advisory_reasons": reasons,
                "avoid_at_all_costs": bool(severity == "critical"),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: float(r["score"]))
    selected = rows_sorted[0]

    stress_vals = np.array([float(r["stress_pred_mpa"]) for r in rows], dtype=np.float64)
    stress_median = float(np.median(stress_vals))
    band_abs = 0.2 * max(abs(stress_median), 1e-9)
    inlier_mask = np.abs(stress_vals - stress_median) <= band_abs
    if not np.any(inlier_mask):
        inlier_mask = np.ones_like(stress_vals, dtype=bool)
    inlier_vals = stress_vals[inlier_mask]

    selected_idx = next(i for i, r in enumerate(rows) if r["strategy"] == selected["strategy"])
    selected_is_outlier = not bool(inlier_mask[selected_idx])
    gate_stress = (
        float(np.median(inlier_vals))
        if selected_is_outlier
        else float(selected["stress_pred_mpa"])
    )
    omitted = [rows[i]["strategy"] for i in range(len(rows)) if not bool(inlier_mask[i])]

    avoid_list = [str(r["strategy"]) for r in rows_sorted if bool(r["avoid_at_all_costs"])]
    reject_list = [str(r["strategy"]) for r in rows_sorted[1:]]

    return {
        "calculix_stress_mode": calc_mode,
        "model_used": bool(calc_surrogate is not None),
        "model_path": str(model_path),
        "strategies": rows_sorted,
        "selected_strategy": str(selected["strategy"]),
        "selected_stress_mpa": float(selected["stress_pred_mpa"]),
        "selected_gate_stress_mpa": float(gate_stress),
        "selected_is_outlier_for_gate": bool(selected_is_outlier),
        "stress_median_mpa": float(stress_median),
        "stress_inlier_band_pct": 20.0,
        "stress_source": str(selected["stress_source"]),
        "selected_ring_radius_mm": float(selected["ring_radius_mm"]),
        "selected_planet_radius_mm": float(selected["planet_radius_mm"]),
        "selected_ratio_residual": float(selected["ratio_residual"]),
        "selected_advisory_reasons": list(selected["advisory_reasons"]),
        "avoid_at_all_costs": avoid_list,
        "rejected_strategies": reject_list,
        "omitted_outlier_strategies_for_gate": omitted,
        "gate_rule": {
            "description": "hard gate uses selected strategy stress unless that stress is >20% away from median across strategies",
            "gate_stress_mpa": float(gate_stress),
        },
    }


def eval_gear(
    params: GearParams,
    i_req_profile: np.ndarray,
    ctx: EvalContext,
) -> GearResult:
    """Evaluate gear system.

    Fidelity routing:
    - fidelity=0: toy friction model
    - fidelity>=1: v1 Litvin synthesis with enhanced mesh loss

    Args:
        params: Gear control parameters.
        i_req_profile: Required ratio profile from thermo (length N_POINTS).
        ctx: Evaluation context (rpm, torque, fidelity).

    Returns:
        GearResult with errors, losses, constraints, and diagnostics.
    """
    # Fidelity routing
    if ctx.fidelity >= 1:
        from ..ports.larrak_v1.gear_forward import v1_eval_gear_forward

        v1_result = v1_eval_gear_forward(params, i_req_profile, ctx)

        # Start with V1 values
        loss_total = v1_result.loss_total
        ledger = v1_result.ledger
        diag_update = {"gear_loss_mode": str(getattr(ctx, "gear_loss_mode", "physics"))}

        # Fidelity 2+: Optional NN gear-loss path (no implicit fallback).
        if ctx.fidelity >= 2:
            gear_loss_mode = str(getattr(ctx, "gear_loss_mode", "physics"))
            if gear_loss_mode == "nn":
                ctx_model_dir = getattr(ctx, "gear_loss_model_dir", None)
                model_dir_str = str(ctx_model_dir).strip() if isinstance(ctx_model_dir, str) else ""
                if not model_dir_str:
                    env_dir = str(os.environ.get("LARRAK2_GEAR_LOSS_NN_DIR", "")).strip()
                    model_dir_str = env_dir if env_dir else str(DEFAULT_GEAR_LOSS_NN_DIR)
                model_dir_str = str(
                    assert_not_legacy_models_path(
                        model_dir_str,
                        purpose="Gear-loss NN model directory",
                    )
                )
                model_dir = Path(model_dir_str)
                if not model_dir.exists():
                    raise FileNotFoundError(
                        "Gear-loss NN mode selected but model directory is missing: "
                        f"'{model_dir}'. Provide --gear-loss-model-dir or switch "
                        "--gear-loss-mode physics."
                    )
                try:
                    surrogate = get_gear_surrogate(model_dir)
                    preds = surrogate.predict(
                        rpm=float(ctx.rpm),
                        torque=float(ctx.torque),
                        base_radius=float(params.base_radius),
                        coeffs=list(params.pitch_coeffs),
                        face_width=float(params.face_width_mm),
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Gear-loss NN inference failed for model directory '{model_dir}'."
                    ) from e

                loss_work_J = float(preds["loss_total"])
                mesh_work_J = float(preds["loss_mesh"])
                bearing_work_J = float(preds["loss_bearing"])
                churning_work_J = float(preds["loss_churning"])

                cycles_per_sec = float(ctx.rpm) / 60.0
                loss_total = loss_work_J * cycles_per_sec

                diag_update.update(
                    {
                        "gear_surrogate_used": True,
                        "gear_loss_model_dir": str(model_dir),
                        "preds": preds,
                    }
                )

                if ledger:
                    ledger = EnergyLedger(
                        W_out_shaft=ledger.W_out_shaft,
                        W_loss_mesh=mesh_work_J,
                        W_loss_bearing=bearing_work_J,
                        W_loss_churning=churning_work_J,
                    )
            elif gear_loss_mode == "physics":
                diag_update["gear_surrogate_used"] = False
            else:
                raise ValueError(f"Unsupported gear_loss_mode '{gear_loss_mode}'")

        stress_strategy = _evaluate_radius_strategy_stress(
            params=params,
            i_req_profile=i_req_profile,
            ctx=ctx,
            gear_diag=v1_result.diag,
        )
        g_stress_hotspot = (
            float(stress_strategy["selected_gate_stress_mpa"]) - STRESS_HOTSPOT_LIMIT_MPA
        )
        G_aug = np.concatenate([np.asarray(v1_result.G, dtype=np.float64), [g_stress_hotspot]])

        return GearResult(
            ratio_error_mean=v1_result.ratio_error_mean,
            ratio_error_max=v1_result.ratio_error_max,
            max_planet_radius=v1_result.max_planet_radius,
            loss_total=loss_total,
            G=G_aug,
            diag={
                **v1_result.diag,
                **diag_update,
                "radius_strategy": stress_strategy,
                "stress_hotspot_limit_mpa": STRESS_HOTSPOT_LIMIT_MPA,
                "stress_hotspot_gate_mpa": float(stress_strategy["selected_gate_stress_mpa"]),
            },
            ledger=ledger,
        )

    theta = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)

    # Generate planet pitch curve from Fourier coefficients
    r_planet = fourier_pitch_curve(theta, params.base_radius, params.pitch_coeffs)

    # Ring radius (fixed toy value)
    r_ring = 80.0  # mm

    # Litvin synthesis to obtain conjugate ring profile
    synth = litvin_synthesize(theta, r_planet, target_ratio=r_ring / params.base_radius)
    R_psi = synth["R_psi"]
    psi = synth["psi"]

    # Ratio tracking error
    ratio_error_mean, ratio_error_max = _compute_ratio_error(r_planet, i_req_profile, r_ring)

    # Loss model (mesh + optional windage)
    loss_total, loss_profile, loss_diag = total_loss(
        theta,
        r_planet,
        synth["rho_c"],
        omega_rpm=ctx.rpm,
        torque=ctx.torque,
        enable_windage=ctx.fidelity >= 1,
    )

    # Geometry metrics
    max_planet_radius = float(np.max(r_planet))
    min_planet_radius = float(np.min(r_planet))

    thickness_profile = R_psi - r_planet
    min_thickness = float(np.min(thickness_profile))
    interference_flag = bool(np.any(thickness_profile < GEAR_INTERFERENCE_CLEARANCE_MM))
    thickness_ok = min_thickness >= GEAR_MIN_THICKNESS_MM
    contact_ratio = float(np.mean(R_psi) / np.maximum(np.mean(r_planet), 1e-6))
    self_intersection = bool(np.any(np.diff(psi) <= 0))

    # Curvature
    curvature = _compute_curvature(r_planet, theta)
    max_curvature = float(np.max(curvature))
    curvature_smoothness = float(np.sqrt(np.mean(np.gradient(curvature) ** 2)))

    omega_rad = ctx.rpm * 2.0 * np.pi / 60.0

    # ---------------------------------------------------------
    # Rigorous Kinematics & Stress (Aligned with fidelity=1)
    # ---------------------------------------------------------
    # 1. Normal Force
    phi_nominal = np.radians(20.0)  # Nominal for fidelity=0
    r_planet_m = np.maximum(r_planet / 1000.0, 1e-6)
    fn_profile = ctx.torque / (r_planet_m * np.maximum(np.cos(phi_nominal), 1e-6))

    # 2. Velocities (Sliding and Entrainment)
    rho_c = synth["rho_c"]
    sliding_speed_profile = omega_rad * np.abs(r_planet / 1000.0 - np.abs(rho_c) / 1000.0)
    sliding_speed_mean = float(np.mean(sliding_speed_profile))
    sliding_speed_max = float(np.max(sliding_speed_profile))

    entrainment_velocity_profile = omega_rad * (r_planet / 1000.0 + r_ring / 1000.0) / 2.0
    entrainment_velocity_mean = float(np.mean(entrainment_velocity_profile))

    # 3. Hertzian Contact Stress (using true E')
    if hasattr(ctx, "material_properties") and ctx.material_properties is not None:
        E_GPa = ctx.material_properties.youngs_modulus_GPa
        nu = ctx.material_properties.poissons_ratio
    else:
        E_GPa = 205.0  # fallback to steel
        nu = 0.29

    E_Pa = E_GPa * 1e9
    E_prime = E_Pa / (1.0 - nu**2)  # self-mated properties

    face_width_m = params.face_width_mm / 1000.0
    rho_reduced = 1.0 / (1.0 / np.maximum(np.abs(rho_c), 1e-6) + 1.0 / r_ring)
    rho_reduced_m = rho_reduced / 1000.0

    hertz_stress_Pa = np.sqrt(
        np.maximum(fn_profile, 0.0)
        * E_prime
        / (np.pi * face_width_m * np.maximum(rho_reduced_m, 1e-9))
    )
    hertz_stress_profile = hertz_stress_Pa / 1e6  # MPa
    hertz_stress_max = float(np.max(hertz_stress_profile))

    # Constraints (G <= 0 feasible)
    constraints = []

    # C1: Ratio error <= tolerance
    g_ratio = ratio_error_max - RATIO_ERROR_TOL
    constraints.append(g_ratio)

    # C2: Min radius >= MIN_RADIUS
    g_min_r = MIN_RADIUS - min_planet_radius
    constraints.append(g_min_r)

    # C3: Max radius <= MAX_RADIUS
    g_max_r = max_planet_radius - MAX_RADIUS
    constraints.append(g_max_r)

    # C4: Max curvature <= MAX_CURVATURE
    g_curv = max_curvature - MAX_CURVATURE
    constraints.append(g_curv)

    # C5: Interference check (soft constraint -> small penalty if present)
    g_interference = 0.1 if interference_flag else -0.1
    constraints.append(g_interference)

    # C6: Min thickness >= threshold
    g_thickness = GEAR_MIN_THICKNESS_MM - min_thickness
    constraints.append(g_thickness)

    # C7: Contact ratio >= 1.0 (soft)
    g_contact = 1.0 - contact_ratio
    constraints.append(g_contact)

    # C8: Self-intersection risk (soft)
    g_self_int = 0.1 if self_intersection else -0.1
    constraints.append(g_self_int)

    stress_strategy = _evaluate_radius_strategy_stress(
        params=params,
        i_req_profile=i_req_profile,
        ctx=ctx,
        gear_diag={
            "r_planet": r_planet,
            "r_ring": r_ring,
            "hertz_stress_max": hertz_stress_max,
        },
    )
    g_stress_hotspot = (
        float(stress_strategy["selected_gate_stress_mpa"]) - STRESS_HOTSPOT_LIMIT_MPA
    )
    constraints.append(g_stress_hotspot)

    G = np.array(constraints, dtype=np.float64)

    # Diagnostics
    diag = {
        "theta": theta,
        "r_planet": r_planet,
        "R_psi": R_psi,
        "thickness_profile": thickness_profile,
        "curvature": curvature,
        "loss_profile": loss_profile,
        "loss_components": loss_diag,
        "r_ring": r_ring,
        "min_planet_radius": min_planet_radius,
        "max_curvature": max_curvature,
        "curvature_smoothness_rms": curvature_smoothness,
        "sliding_speed_profile": sliding_speed_profile,
        "sliding_speed_mean": sliding_speed_mean,
        "sliding_speed_max": sliding_speed_max,
        "entrainment_velocity_profile": entrainment_velocity_profile,
        "entrainment_velocity_mean": entrainment_velocity_mean,
        "hertz_stress_profile": hertz_stress_profile,
        "hertz_stress_max": hertz_stress_max,
        "fn_profile": fn_profile,
        "face_width_mm": params.face_width_mm,
        "min_thickness": min_thickness,
        "thickness_ok": thickness_ok,
        "interference_flag": interference_flag,
        "contact_ratio": contact_ratio,
        "self_intersection": self_intersection,
        "radius_strategy": stress_strategy,
        "stress_hotspot_limit_mpa": STRESS_HOTSPOT_LIMIT_MPA,
        "stress_hotspot_gate_mpa": float(stress_strategy["selected_gate_stress_mpa"]),
    }

    # Populate Energy Ledger (Toy Mode)
    # W_in = Torque * Theta? No, that's shaft input.
    # W_in_piston comes from Thermo, not strictly known here unless passed.
    # But we can calculate W_out_shaft and W_loss.
    # For Toy Mode, we assume Input = Output + Loss for now, or just track what we can.
    # The actual "Piston Work" input is external.
    # Let's populate what we know.

    # Calculate Mesh Work Dissipated (Integral of Power Loss)
    # P_loss = loss_profile (W)
    # Energy = Integral P_loss dt = Integral P_loss * (dtheta / omega)
    # dt = dtheta / omega_rad
    omega_rad = ctx.rpm * 2 * np.pi / 60
    dtheta = 2 * np.pi / N_POINTS
    dt_step = dtheta / omega_rad if omega_rad > 1e-9 else 0.0

    W_mesh = float(np.sum(loss_profile) * dt_step)

    # Shaft Work = Torque * 2pi (per rev)
    # This is W_out if torque is output torque.
    W_out = float(ctx.torque * 2 * np.pi)

    # Ledger
    ledger = EnergyLedger(
        W_out_shaft=W_out,
        W_loss_mesh=W_mesh,
        # In toy mode, efficiency is not perfectly 1.0 unless mu=0
        # If mu=0, W_mesh=0.
    )

    return GearResult(
        ratio_error_mean=ratio_error_mean,
        ratio_error_max=ratio_error_max,
        max_planet_radius=max_planet_radius,
        loss_total=loss_total,
        G=G,
        diag=diag,
        ledger=ledger,
    )
