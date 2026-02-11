"""Thermo forward-evaluation ported from Larrak v1 - Phase 2 Reparameterized.

This module contains pure forward-evaluation logic with:
- Phase-duration-driven V(θ) mapping
- First-law thermodynamic stepping
- No artificial pressure clamps
- Wiebe heat release

NO optimizer code, NO CasADi, NO v1 imports.

Attribution: Original code from Larrak v1 by Max Holden.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...core.constants import N_THETA, P_ATM, RATIO_SLOPE_LIMIT_FID1
from ...core.encoding import ThermoParams
from ...core.types import BreathingConfig, EvalContext
from ...thermo.motionlaw import _ratio_profile_stats

# Constants
DEFAULT_WIEBE_M = 2.0  # Wiebe shape factor
DEFAULT_WIEBE_A = 6.9  # Wiebe efficiency factor (99% burn)
GAMMA = 1.3  # Ratio of specific heats
T_INITIAL = 350.0  # K (intake temperature)
CV = 718.0  # J/(kg·K) specific heat at constant volume
R_SPECIFIC = 287.0  # J/(kg·K) specific gas constant for air

# Fuel / mixture constants (used for λ-based fuel mass derivation)
AFR_STOICH_GASOLINE = 14.7  # [-] stoichiometric air-fuel ratio
LHV_GASOLINE = 44.0e6  # J/kg (lower heating value)
# Oxygen required per kg fuel for complete combustion (gasoline approximation).
# For iso-octane: C8H18 + 12.5 O2 -> ... gives ~3.51 kg O2 / kg fuel.
O2_REQUIRED_PER_KG_FUEL = 3.51  # kg_O2 / kg_fuel
O2_MASS_FRACTION_AIR = 0.233  # mass fraction of O2 in dry air

# Simple residual-energy mixing (kept explicit; tune/replace with better physics as data improves)
T_FRESH = 300.0  # K
T_RESIDUAL = 900.0  # K
RESIDUAL_HEAT_GAIN = 0.02  # fraction of released heat retained in initial state

# Pressure limits (for constraint, not clamp)
P_LIMIT_BAR = 300.0  # Target pressure limit [bar]


@dataclass
class V1ThermoResult:
    """Result from v1 thermo forward evaluation."""

    efficiency: float
    requested_ratio_profile: np.ndarray  # Required ratio profile for gear
    G: np.ndarray  # G <= 0 feasible
    diag: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Wiebe functions (copied from campro/physics/chem.py)
# =============================================================================


def wiebe_function(
    theta: float,
    theta_start: float,
    theta_duration: float,
    m: float,
    a: float,
) -> float:
    """Wiebe function for cumulative heat release fraction."""
    if theta < theta_start:
        return 0.0
    if theta > theta_start + theta_duration:
        return 1.0

    theta_norm = (theta - theta_start) / theta_duration
    x_b = 1.0 - math.exp(-a * (theta_norm ** (m + 1.0)))

    return x_b


def wiebe_array(
    theta_deg: np.ndarray,
    theta_start: float,
    theta_duration: float,
    m: float,
    a: float,
) -> np.ndarray:
    """Vectorized Wiebe function."""
    x_b = np.zeros_like(theta_deg, dtype=float)

    in_burn = (theta_deg >= theta_start) & (theta_deg <= theta_start + theta_duration)
    theta_norm = np.clip((theta_deg - theta_start) / theta_duration, 0.0, 1.0)
    x_b[in_burn] = 1.0 - np.exp(-a * (theta_norm[in_burn] ** (m + 1.0)))

    past_burn = theta_deg > theta_start + theta_duration
    x_b[past_burn] = 1.0

    return x_b


# =============================================================================
# Phase-driven volume model
# =============================================================================


def compute_phase_driven_volume(
    theta_deg: np.ndarray,
    compression_duration: float,
    expansion_duration: float,
    v_clearance: float,
    v_displaced: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute volume trace where durations affect the actual kinematics.

    The cycle is divided into 4 phases:
    - Phase 1: Expansion (TDC → BDC): controlled by expansion_duration
    - Phase 2: BDC dwell
    - Phase 3: Compression (BDC → TDC): controlled by compression_duration
    - Phase 4: TDC dwell
    """
    n = len(theta_deg)
    V = np.zeros(n)
    dV_dtheta = np.zeros(n)

    for i, theta in enumerate(theta_deg):
        # Wrap theta to [0, 360)
        theta = theta % 360.0

        if theta <= expansion_duration:
            # Expansion phase: TDC → BDC
            s = theta / expansion_duration if expansion_duration > 0 else 0.0
            f = 0.5 * (1.0 - np.cos(np.pi * s))
            V[i] = v_clearance + f * v_displaced
            # df/ds = π/2 * sin(π*s), ds/dθ = 1/expansion_duration
            if expansion_duration > 0:
                dV_dtheta[i] = v_displaced * 0.5 * np.pi * np.sin(np.pi * s) / expansion_duration
            else:
                dV_dtheta[i] = 0.0

        elif theta <= (360.0 - compression_duration):
            # Dwell at or near BDC (volume stays near maximum)
            V[i] = v_clearance + v_displaced  # Full BDC volume
            dV_dtheta[i] = 0.0

        else:
            # Compression phase: BDC → TDC
            s = (
                (theta - (360.0 - compression_duration)) / compression_duration
                if compression_duration > 0
                else 1.0
            )
            f = 0.5 * (1.0 - np.cos(np.pi * s))  # 0→1 as we compress
            V[i] = v_clearance + (1.0 - f) * v_displaced  # 1-f because we're going TDC-ward
            # df/ds = π/2 * sin(π*s), ds/dθ = 1/compression_duration
            if compression_duration > 0:
                dV_dtheta[i] = -v_displaced * 0.5 * np.pi * np.sin(np.pi * s) / compression_duration
            else:
                dV_dtheta[i] = 0.0

    return V, dV_dtheta


# =============================================================================
# First-law thermodynamic integration
# =============================================================================


def compute_firstlaw_cycle(
    theta_deg: np.ndarray,
    V: np.ndarray,
    dV_dtheta: np.ndarray,
    theta_comb_start: float,
    theta_comb_duration: float,
    q_total: float,
    *,
    T0: float | None = None,
    p0: float | None = None,
    m_gas_override: float | None = None,
    m_wiebe: float = DEFAULT_WIEBE_M,
    a_wiebe: float = DEFAULT_WIEBE_A,
) -> dict[str, Any]:
    """Compute p-V-T cycle using first-law integration."""
    n = len(theta_deg)

    # Initialize
    p = np.zeros(n)
    T = np.zeros(n)
    x_b = np.zeros(n)
    dQ = np.zeros(n)
    W_cumulative = np.zeros(n)

    # Initial state (beginning of cycle at TDC)
    T[0] = float(T_INITIAL if T0 is None else T0)
    p[0] = float(P_ATM if p0 is None else p0)

    if m_gas_override is not None:
        m_gas = float(m_gas_override)
        # If p0 not provided, compute from ideal gas
        if p0 is None:
            p[0] = m_gas * R_SPECIFIC * T[0] / max(float(V[0]), 1e-12)
    else:
        # Mass from ideal gas
        m_gas = p[0] * V[0] / (R_SPECIFIC * T[0])

    for i in range(1, n):
        # Burn fraction
        x_b[i] = wiebe_function(
            theta_deg[i], theta_comb_start, theta_comb_duration, m_wiebe, a_wiebe
        )

        # Heat release this step
        dx_b = x_b[i] - x_b[i - 1]
        dQ[i] = dx_b * q_total

        # Volume change this step
        dV = V[i] - V[i - 1]

        # Work done by gas this step
        dW = p[i - 1] * dV
        W_cumulative[i] = W_cumulative[i - 1] + dW

        # First law: dU = dQ - dW
        # dU = m * cv * dT
        dT = (dQ[i] - dW) / (m_gas * CV)
        T[i] = T[i - 1] + dT

        # Clamp temperature to prevent negative values
        T[i] = max(T[i], 100.0)

        # Pressure from ideal gas law
        p[i] = m_gas * R_SPECIFIC * T[i] / V[i]

        # No artificial pressure clamp! Let physics determine pressure.
        # Only clamp to prevent numerical issues (not 500 bar ceiling)
        p[i] = max(p[i], 100.0)  # Prevent numerical underflow only

    return {
        "p": p,
        "T": T,
        "x_b": x_b,
        "dQ": dQ,
        "W_cumulative": W_cumulative,
        "m_gas": m_gas,
        "T0": T[0],
        "p0": p[0],
    }


def compute_cycle_work(p: np.ndarray, V: np.ndarray) -> float:
    """Compute cycle work from p-V diagram (shoelace formula)."""
    # Close the cycle by appending first point
    V_closed = np.append(V, V[0])
    p_closed = np.append(p, p[0])

    # Shoelace formula for closed polygon area
    dV = np.diff(V_closed)
    p_avg = 0.5 * (p_closed[:-1] + p_closed[1:])
    W = -float(np.sum(p_avg * dV))  # Negative for clockwise = positive work

    return W


def compute_efficiency(W: float, Q_in: float) -> float:
    """Compute thermal efficiency."""
    if Q_in <= 0:
        return 0.0
    return min(max(W / Q_in, 0.0), 1.0)


# =============================================================================
# Ratio profile generation
# =============================================================================


def generate_requested_ratio_profile(
    theta_deg: np.ndarray,
    dV_dtheta: np.ndarray,
    base_ratio: float,
) -> np.ndarray:
    """Generate required gear ratio profile from volume derivative."""
    # Normalize dV/dtheta to create ratio variation
    dV_max = np.max(np.abs(dV_dtheta))
    if dV_max > 0:
        dV_norm = dV_dtheta / dV_max
    else:
        dV_norm = np.zeros_like(dV_dtheta)

    # Ratio varies with piston velocity requirement
    amplitude = 0.3 * base_ratio
    i_req = base_ratio + amplitude * dV_norm
    return np.maximum(i_req, 0.5)  # prevent negative/zero ratios


# =============================================================================
# Main forward-evaluation function
# =============================================================================


def v1_eval_thermo_forward(
    params: ThermoParams,
    ctx: EvalContext,
    ratio_slope_limit: float | None = None,
) -> V1ThermoResult:
    """Evaluate thermodynamics using phase-driven v1 model."""
    # Theta grid using constant N_THETA
    theta_deg = np.linspace(0, 360, N_THETA, endpoint=False)

    # Geometry defaults
    stroke_mm = 90.0  # mm
    bore_mm = 80.0  # mm
    compression_ratio = 10.0  # default

    # Compute volumes
    area_m2 = np.pi * (bore_mm / 2000) ** 2
    stroke_m = stroke_mm / 1000
    v_clearance = area_m2 * stroke_m / (compression_ratio - 1)
    v_displaced = area_m2 * stroke_m

    # Phase-driven volume trace
    V, dV_dtheta = compute_phase_driven_volume(
        theta_deg,
        params.compression_duration,
        params.expansion_duration,
        v_clearance,
        v_displaced,
    )

    # Combustion timing
    theta_comb_start = params.heat_release_center  # degrees after TDC
    theta_comb_duration = params.heat_release_width

    # -------------------------------------------------------------------------
    # Operating-point-dependent initial conditions (fuel mass + temperature)
    # -------------------------------------------------------------------------
    # Baseline trapped air mass estimate (if no OpenFOAM NN is enabled)
    V_bdc = float(v_clearance + v_displaced)  # max volume
    m_air_trapped = P_ATM * V_bdc / (R_SPECIFIC * T_INITIAL)
    scav_eff = 1.0
    residual_fraction = 0.0
    trapped_o2_mass = m_air_trapped * O2_MASS_FRACTION_AIR
    openfoam_nn_used = False

    # Fidelity >= 2: use OpenFOAM NN surrogate for breathing/richness state
    if ctx.fidelity >= 2:
        model_path = os.environ.get(
            "LARRAK2_OPENFOAM_NN_PATH", "models/openfoam_nn/openfoam_breathing.pt"
        )
        mp = Path(model_path)
        if not mp.exists():
            raise FileNotFoundError(
                f"OpenFOAM NN surrogate not found at '{model_path}'. "
                "Train it first (see scripts/train_openfoam_surrogate.py) "
                "or set LARRAK2_OPENFOAM_NN_PATH to the artifact path."
            )

        from ...surrogate.openfoam_nn import get_openfoam_surrogate

        surrogate = get_openfoam_surrogate(mp)
        bcfg = ctx.breathing or BreathingConfig()

        pred = surrogate.predict_one(
            {
                "rpm": float(ctx.rpm),
                "torque": float(ctx.torque),
                "lambda_af": float(params.lambda_af),
                "bore_mm": float(bcfg.bore_mm),
                "stroke_mm": float(bcfg.stroke_mm),
                "intake_port_area_m2": float(bcfg.intake_port_area_m2),
                "exhaust_port_area_m2": float(bcfg.exhaust_port_area_m2),
                "p_manifold_Pa": float(bcfg.p_manifold_Pa),
                "p_back_Pa": float(bcfg.p_back_Pa),
                "overlap_deg": float(bcfg.overlap_deg),
                "intake_open_deg": float(bcfg.intake_open_deg),
                "intake_close_deg": float(bcfg.intake_close_deg),
                "exhaust_open_deg": float(bcfg.exhaust_open_deg),
                "exhaust_close_deg": float(bcfg.exhaust_close_deg),
            }
        )
        m_air_trapped = float(pred["m_air_trapped"])
        scav_eff = float(pred["scavenging_efficiency"])
        residual_fraction = float(pred["residual_fraction"])
        trapped_o2_mass = float(pred["trapped_o2_mass"])
        openfoam_nn_used = True

    # Fuel mass from lambda (λ is a decision variable; do NOT derive λ from fuel mass)
    lam = float(max(params.lambda_af, 1e-6))
    m_fuel = float(m_air_trapped / (lam * AFR_STOICH_GASOLINE))

    Q_chem = m_fuel * LHV_GASOLINE

    # Oxygen-limited burn completion (physically grounded):
    # burn fraction is limited by trapped oxygen availability.
    # This naturally penalizes rich mixtures (lower lambda -> more fuel for a given trapped air/O2).
    denom_o2 = max(m_fuel * O2_REQUIRED_PER_KG_FUEL, 1e-12)
    burn_frac_o2 = float(np.clip(trapped_o2_mass / denom_o2, 0.0, 1.0))
    combustion_completion_factor = 1.0
    Q_rel = Q_chem * burn_frac_o2 * combustion_completion_factor

    # Initial temperature derived from fuel input + surrogate residual state (not a direct rpm curve)
    m_total = max(m_air_trapped + m_fuel, 1e-9)
    base_T = T_FRESH * (1.0 - residual_fraction) + T_RESIDUAL * residual_fraction
    delta_T = RESIDUAL_HEAT_GAIN * Q_rel / (m_total * CV)
    T0 = float(max(base_T + delta_T, 100.0))

    # Initial pressure from ideal gas at TDC
    p0 = float(max(m_total * R_SPECIFIC * T0 / max(float(V[0]), 1e-12), 100.0))

    # Heat released to the working fluid in this cycle
    q_total = float(max(Q_rel, 0.0))

    # forward steps
    cycle = compute_firstlaw_cycle(
        theta_deg,
        V,
        dV_dtheta,
        theta_comb_start,
        theta_comb_duration,
        q_total,
        T0=T0,
        p0=p0,
        m_gas_override=m_total,
    )

    # Work and efficiency
    W = compute_cycle_work(cycle["p"], V)
    efficiency = compute_efficiency(W, q_total)

    # Ratio profile
    base_ratio = 2.0 + 0.5 * (compression_ratio - 10) / 10
    requested_ratio_profile = generate_requested_ratio_profile(theta_deg, dV_dtheta, base_ratio)

    # Pressure metrics
    p_max = float(np.max(cycle["p"]))
    p_min = float(np.min(cycle["p"]))
    p_mean = float(np.mean(cycle["p"]))

    # Explicit Constraint Assembly
    # Mapping conventions: G[k] <= 0 is feasible
    constraints = {}

    # C1: Efficiency >= 0
    constraints["g_eff_min"] = 0.0 - efficiency

    # C2: Efficiency <= 0.6
    constraints["g_eff_max"] = efficiency - 0.6

    # C3: Max pressure <= P_LIMIT (normalized)
    p_limit_pa = P_LIMIT_BAR * 1e5
    constraints["g_p_max"] = (p_max - p_limit_pa) / p_limit_pa

    # C4: Ratio profile slope <= limit
    ratio_stats = _ratio_profile_stats(requested_ratio_profile)
    static_limit = RATIO_SLOPE_LIMIT_FID1
    slope_limit = min(static_limit, float(ratio_slope_limit)) if ratio_slope_limit is not None else static_limit
    constraints["g_ratio_slope"] = ratio_stats["max_slope"] - slope_limit

    # Pack into array in fixed order
    G_list = [
        constraints["g_eff_min"],
        constraints["g_eff_max"],
        constraints["g_p_max"],
        constraints["g_ratio_slope"],
    ]
    G = np.array(G_list, dtype=np.float64)

    # Diagnostics with named constraints for easier debugging
    diag = {
        "theta_deg": theta_deg,
        "p": cycle["p"],
        "V": V,
        "T": cycle["T"],
        "x_b": cycle["x_b"],
        "dQ": cycle["dQ"],
        "dV_dtheta": dV_dtheta,
        "W": W,
        "W_cumulative": cycle["W_cumulative"],
        "T0": float(cycle.get("T0", T0)),
        "p0": float(cycle.get("p0", p0)),
        "p_max": p_max,
        "p_min": p_min,
        "p_mean": p_mean,
        "p_limit_bar": P_LIMIT_BAR,
        "q_total": q_total,
        "m_gas": cycle["m_gas"],
        "m_air_trapped": float(m_air_trapped),
        "m_fuel": float(m_fuel),
        "lambda_af": float(params.lambda_af),
        "Q_chem": float(Q_chem),
        "Q_rel": float(Q_rel),
        "trapped_o2_mass": float(trapped_o2_mass),
        "burn_frac_o2": float(burn_frac_o2),
        "scavenging_efficiency": float(scav_eff),
        "residual_fraction": float(residual_fraction),
        "openfoam_nn_used": bool(openfoam_nn_used),
        "compression_duration": params.compression_duration,
        "expansion_duration": params.expansion_duration,
        "heat_release_center": params.heat_release_center,
        "heat_release_width": params.heat_release_width,
        "requested_ratio_profile": requested_ratio_profile,
        "ratio_profile_stats": ratio_stats,
        "v1_port": True,
        "phase_driven": True,
        "constraints": constraints,  # Explicit map
        "ratio_slope_limit_used": float(slope_limit),
        "ratio_slope_limit_source": "manufacturability" if ratio_slope_limit is not None else "static",
    }

    return V1ThermoResult(
        efficiency=efficiency,
        requested_ratio_profile=requested_ratio_profile,
        G=G,
        diag=diag,
    )
