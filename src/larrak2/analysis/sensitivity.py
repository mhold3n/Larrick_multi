"""Sensitivity and Constraint Analysis Logic."""

from __future__ import annotations

import numpy as np

from larrak2.core.encoding import bounds, decode_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def sensitivity_scan(x_base: np.ndarray, ctx: EvalContext) -> dict:
    """Perturb each DOF ±1% and measure output sensitivity."""

    result_base = evaluate_candidate(x_base, ctx)
    _cand_base = decode_candidate(x_base)

    base_metrics = {
        "efficiency": -result_base.F[0],  # F[0] is -efficiency
        "loss": result_base.F[1],
        "p_max": result_base.diag.get("thermo", {}).get("p_max", 0),
        "W": result_base.diag.get("thermo", {}).get("W", 0),
    }

    xl, xu = bounds()

    # Thermo parameters are indices 0-3
    thermo_names = [
        "compression_duration",
        "expansion_duration",
        "heat_release_center",
        "heat_release_width",
    ]
    # Gear parameters are indices 4-11
    gear_names = [
        "base_radius",
        "pitch_coeff_0",
        "pitch_coeff_1",
        "pitch_coeff_2",
        "pitch_coeff_3",
        "pitch_coeff_4",
        "pitch_coeff_5",
        "pitch_coeff_6",
    ]

    all_names = thermo_names + gear_names

    sensitivities = []

    for i, name in enumerate(all_names):
        x_plus = x_base.copy()
        x_minus = x_base.copy()

        # Perturb by 1% of value (or 1% of range if at zero)
        delta = max(abs(x_base[i]) * 0.01, (xu[i] - xl[i]) * 0.01)

        x_plus[i] = min(x_base[i] + delta, xu[i])
        x_minus[i] = max(x_base[i] - delta, xl[i])

        result_plus = evaluate_candidate(x_plus, ctx)
        result_minus = evaluate_candidate(x_minus, ctx)

        # Compute deltas
        metrics_plus = {
            "efficiency": -result_plus.F[0],
            "loss": result_plus.F[1],
            "p_max": result_plus.diag.get("thermo", {}).get("p_max", 0),
            "W": result_plus.diag.get("thermo", {}).get("W", 0),
        }
        metrics_minus = {
            "efficiency": -result_minus.F[0],
            "loss": result_minus.F[1],
            "p_max": result_minus.diag.get("thermo", {}).get("p_max", 0),
            "W": result_minus.diag.get("thermo", {}).get("W", 0),
        }

        sens = {
            "param": name,
            "idx": i,
            "type": "thermo" if i < 4 else "gear",
            "base_value": float(x_base[i]),
            "delta": float(delta),
        }

        for metric in ["efficiency", "loss", "p_max", "W"]:
            d_plus = metrics_plus[metric] - base_metrics[metric]
            d_minus = metrics_minus[metric] - base_metrics[metric]
            sens[f"d_{metric}_plus"] = float(d_plus)
            sens[f"d_{metric}_minus"] = float(d_minus)
            # Normalized sensitivity: (d_metric / metric) / (d_param / param)
            if base_metrics[metric] != 0 and x_base[i] != 0:
                sens[f"sens_{metric}"] = float(
                    (d_plus - d_minus) / (2 * base_metrics[metric]) / (delta / x_base[i])
                )
            else:
                sens[f"sens_{metric}"] = 0.0

        sensitivities.append(sens)

    return {
        "base_metrics": base_metrics,
        "sensitivities": sensitivities,
    }


def constraint_activation_report(X: np.ndarray, ctx: EvalContext) -> dict:
    """Analyze how close candidates are to constraint limits."""

    # Constraint limits (from thermo_forward.py and litvin_core.py)
    limits = {
        "g0_eff_min": {"limit": 0.0, "type": ">=", "desc": "efficiency >= 0"},
        "g1_eff_max": {"limit": 0.6, "type": "<=", "desc": "efficiency <= 0.6"},
        "g2_p_max": {"limit": 500.0, "type": "<=", "desc": "p_max <= 500 bar"},
        "g3_ratio_error": {"limit": 0.5, "type": "<=", "desc": "ratio_error <= 0.5"},
        "g4_min_radius": {"limit": 5.0, "type": ">=", "desc": "min_planet_r >= 5 mm"},
        "g5_max_radius": {"limit": 100.0, "type": "<=", "desc": "max_planet_r <= 100 mm"},
        "g6_curvature": {"limit": 0.5, "type": ">=", "desc": "min_osc_radius >= 0.5 mm"},
    }

    n_candidates = len(X)

    # Collect constraint values for all candidates
    G_all = []
    metrics_all = []

    for x in X:
        result = evaluate_candidate(x, ctx)
        G_all.append(result.G)

        # Extract actual metrics for ratio calculation
        thermo_diag = result.diag.get("thermo", {})
        gear_diag = result.diag.get("gear", {})

        metrics_all.append(
            {
                "efficiency": -result.F[0],
                "p_max_bar": thermo_diag.get("p_max", 0) / 1e5,
                "ratio_error_max": gear_diag.get("ratio_error_max", 0)
                if "ratio_error_max" in gear_diag
                else 0,
                "min_planet_r": gear_diag.get("min_planet_radius", 0),
                "max_planet_r": gear_diag.get("max_planet_radius", 0),
            }
        )

    G_all = np.array(G_all)

    # Activation analysis
    constraint_stats = []
    for j in range(G_all.shape[1]):
        g_vals = G_all[:, j]

        stat = {
            "constraint": f"g{j}",
            "description": list(limits.values())[j]["desc"]
            if j < len(limits)
            else f"constraint {j}",
            "mean": float(np.mean(g_vals)),
            "std": float(np.std(g_vals)),
            "min": float(np.min(g_vals)),
            "max": float(np.max(g_vals)),
            "n_violated": int(np.sum(g_vals > 0)),
            "n_active": int(np.sum(g_vals > -0.01)),  # Within 1% of being violated
            "pct_active": float(np.sum(g_vals > -0.01) / n_candidates * 100),
        }
        constraint_stats.append(stat)

    # Metric distributions
    metric_stats = {
        "efficiency": {
            "mean": float(np.mean([m["efficiency"] for m in metrics_all])),
            "std": float(np.std([m["efficiency"] for m in metrics_all])),
            "min": float(np.min([m["efficiency"] for m in metrics_all])),
            "max": float(np.max([m["efficiency"] for m in metrics_all])),
        },
        "p_max_bar": {
            "mean": float(np.mean([m["p_max_bar"] for m in metrics_all])),
            "std": float(np.std([m["p_max_bar"] for m in metrics_all])),
            "min": float(np.min([m["p_max_bar"] for m in metrics_all])),
            "max": float(np.max([m["p_max_bar"] for m in metrics_all])),
            "limit": 500.0,
            "headroom_pct": float(
                (500 - np.max([m["p_max_bar"] for m in metrics_all])) / 500 * 100
            ),
        },
    }

    return {
        "n_candidates": n_candidates,
        "constraint_stats": constraint_stats,
        "metric_distributions": metric_stats,
    }
