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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
        diag_update = {}

        # Fidelity 2+: Surrogate Loss Override
        if ctx.fidelity >= 2:
            try:
                # Load surrogate (lazy singleton)
                # For CI/Testing, this function can be patched to return a mock
                surrogate = get_gear_surrogate("models/gear_surrogate_v1")

                # Predict (Assumes surrogate outputs Energy in Joules per Cycle)
                preds = surrogate.predict(
                    rpm=float(ctx.rpm),
                    torque=float(ctx.torque),
                    base_radius=float(params.base_radius),
                    coeffs=list(params.pitch_coeffs),
                )

                loss_work_J = preds["loss_total"]
                mesh_work_J = preds["loss_mesh"]
                bearing_work_J = preds["loss_bearing"]
                churning_work_J = preds["loss_churning"]

                # Update Power (Watts) = Energy (J) * Cycles/sec
                cycles_per_sec = float(ctx.rpm) / 60.0
                loss_total = loss_work_J * cycles_per_sec

                diag_update = {"gear_surrogate_used": True, "preds": preds}

                # Update Ledger
                if ledger:
                    ledger = EnergyLedger(
                        W_out_shaft=ledger.W_out_shaft,
                        W_loss_mesh=mesh_work_J,
                        W_loss_bearing=bearing_work_J,
                        W_loss_churning=churning_work_J,
                        # Other fields remain default/0 or lost if not copied,
                        # but EnergyLedger is simple.
                    )

            except (ImportError, FileNotFoundError, ValueError):
                # Fallback to V1 if surrogate fails
                pass

        return GearResult(
            ratio_error_mean=v1_result.ratio_error_mean,
            ratio_error_max=v1_result.ratio_error_max,
            max_planet_radius=v1_result.max_planet_radius,
            loss_total=loss_total,
            G=v1_result.G,
            diag={**v1_result.diag, **diag_update},
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

    # Sliding speed proxy
    omega_rad = ctx.rpm * 2 * np.pi / 60
    sliding_speed = omega_rad * np.abs(np.gradient(r_planet / 1000))
    sliding_speed_mean = float(np.mean(sliding_speed))
    sliding_speed_max = float(np.max(sliding_speed))

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
        "sliding_speed_mean": sliding_speed_mean,
        "sliding_speed_max": sliding_speed_max,
        "min_thickness": min_thickness,
        "thickness_ok": thickness_ok,
        "interference_flag": interference_flag,
        "contact_ratio": contact_ratio,
        "self_intersection": self_intersection,
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
