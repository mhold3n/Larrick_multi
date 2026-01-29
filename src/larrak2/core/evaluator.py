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

import numpy as np

from ..core.constants import MODEL_VERSION_GEAR_V1, MODEL_VERSION_THERMO_V1
from ..gear.litvin_core import eval_gear
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

    # Thermo evaluation
    t_thermo_start = time.perf_counter()
    thermo_result = eval_thermo(candidate.thermo, ctx)
    t_thermo = time.perf_counter() - t_thermo_start

    # Gear evaluation (uses ratio profile from thermo)
    t_gear_start = time.perf_counter()
    gear_result = eval_gear(candidate.gear, thermo_result.requested_ratio_profile, ctx)
    t_gear = time.perf_counter() - t_gear_start

    # Objectives (minimize):
    # - F[0]: negative efficiency (we want to maximize efficiency → minimize -efficiency)
    # - F[1]: gear loss (minimize)
    # - F[2]: max planet radius (minimize packaging size) [NEW]
    
    # Fidelity 2: Apply surrogate corrections
    eff_corrected = thermo_result.efficiency
    loss_corrected = gear_result.loss_total
    
    surrogate_meta = {}
    
    if ctx.fidelity >= 2:

        # Load surrogate engine (lazy singleton)
        from larrak2.surrogate.inference import get_surrogate_engine
        
        engine = get_surrogate_engine()
        
        # Predict corrections (returns 0 if models missing)
        delta_eff, delta_loss, meta = engine.predict_corrections(x)
        
        eff_corrected += delta_eff
        loss_corrected += delta_loss
        
        surrogate_meta = {
            "surrogate_used": True,
            "delta_eff": delta_eff,
            "delta_loss": delta_loss,
            "version_surrogate": "SurrogateEngine_v1",
            "active_models": meta.get("surrogates_active", []),
            "uncertainty": meta.get("uncertainty", {}),
        }

    F = np.array(
        [
            -eff_corrected,  # negate for minimization
            loss_corrected,
            gear_result.max_planet_radius,
        ],
        dtype=np.float64,
    )

    # Constraints (G <= 0 feasible)
    G = np.concatenate([thermo_result.G, gear_result.G])

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
            "efficiency": eff_corrected,
            "efficiency_raw": thermo_result.efficiency,
            "ratio_error_mean": gear_result.ratio_error_mean,
            "ratio_error_max": gear_result.ratio_error_max,
            "max_planet_radius": gear_result.max_planet_radius,
            "loss_total": loss_corrected,
            "loss_raw": gear_result.loss_total,
        },
        "versions": {
            "thermo_v1": MODEL_VERSION_THERMO_V1,
            "gear_v1": MODEL_VERSION_GEAR_V1,
            **surrogate_meta,
        },
    }

    return EvalResult(F=F, G=G, diag=diag)
