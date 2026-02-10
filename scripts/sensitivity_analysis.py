#!/usr/bin/env python3
"""Sensitivity scan and constraint activation analysis.

Checks:
A) Sensitivity: Perturb each thermo DOF ±1%, measure Δeff, Δp_max, Δwork, Δloss
B) Constraint activation: How close are candidates to constraint limits?

Usage:
    python scripts/sensitivity_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from larrak2.core.encoding import bounds, decode_candidate, mid_bounds_candidate
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


def main():
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=123)

    print("=" * 70)
    print("SENSITIVITY AND CONSTRAINT ACTIVATION ANALYSIS")
    print("=" * 70)

    # A) Sensitivity scan on mid-bounds candidate
    print("\n[A] Sensitivity Scan (±1% perturbation on mid-bounds candidate)")
    print("-" * 70)

    x_mid = mid_bounds_candidate()
    sens_results = sensitivity_scan(x_mid, ctx)

    print("\nBase metrics:")
    for k, v in sens_results["base_metrics"].items():
        if k == "efficiency":
            print(f"  {k}: {v:.4%}")
        elif k in ["p_max", "W"]:
            print(f"  {k}: {v:.2e}")
        else:
            print(f"  {k}: {v:.4f}")

    print("\nSensitivity table (Δ for ±1% parameter change):")
    print(f"{'Parameter':<25} {'Type':<8} {'Δeff%':<12} {'Δloss':<12} {'Δp_max':<12}")
    print("-" * 70)

    for s in sens_results["sensitivities"]:
        d_eff = (s["d_efficiency_plus"] - s["d_efficiency_minus"]) / 2 * 100  # as percentage points
        d_loss = (s["d_loss_plus"] - s["d_loss_minus"]) / 2
        d_pmax = (s["d_p_max_plus"] - s["d_p_max_minus"]) / 2 / 1e5  # as bar
        print(f"{s['param']:<25} {s['type']:<8} {d_eff:+.6f}% {d_loss:+.4e} {d_pmax:+.2f} bar")

    # B) Constraint activation on Pareto front
    print("\n" + "=" * 70)
    print("[B] Constraint Activation Report")
    print("-" * 70)

    # Load Pareto front from diagnostic run
    pareto_file = Path("diagnostic_results_v2/pareto_X.npy")
    if pareto_file.exists():
        X_pareto = np.load(pareto_file)
        print(f"Analyzing {len(X_pareto)} Pareto solutions from diagnostic_results_v2/")
    else:
        # Fall back to mid-bounds and some random samples
        print("Pareto file not found, using synthetic samples")
        xl, xu = bounds()
        rng = np.random.default_rng(123)
        X_pareto = np.vstack(
            [mid_bounds_candidate()] + [xl + rng.random(len(xl)) * (xu - xl) for _ in range(15)]
        )

    activation = constraint_activation_report(X_pareto, ctx)

    print(f"\nConstraint activation (n={activation['n_candidates']} candidates):")
    print(f"{'Constraint':<12} {'Description':<30} {'Mean':<10} {'Active%':<10} {'Violated'}")
    print("-" * 80)

    for stat in activation["constraint_stats"]:
        print(
            f"{stat['constraint']:<12} {stat['description']:<30} {stat['mean']:+.4f} "
            f"{stat['pct_active']:>6.1f}% {stat['n_violated']}"
        )

    print("\nMetric distributions:")
    for metric, stats in activation["metric_distributions"].items():
        print(f"  {metric}:")
        print(f"    range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    mean:  {stats['mean']:.4f} ± {stats['std']:.4f}")
        if "limit" in stats:
            print(f"    limit: {stats['limit']}, headroom: {stats['headroom_pct']:.1f}%")

    # Save results
    output = {
        "sensitivity": sens_results,
        "activation": activation,
    }

    with open("sensitivity_analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Check for low thermo sensitivity
    thermo_sens = [s for s in sens_results["sensitivities"] if s["type"] == "thermo"]
    max_thermo_eff_sens = max(
        abs((s["d_efficiency_plus"] - s["d_efficiency_minus"]) / 2) for s in thermo_sens
    )

    if max_thermo_eff_sens < 0.001:  # less than 0.1% change
        print("\n⚠️  THERMO SENSITIVITY IS LOW:")
        print("   The thermo DOFs have negligible effect on efficiency.")
        print("   This explains the tight efficiency range in Pareto front.")
        print("   → Need to reparameterize thermo or add more DOFs.")
    else:
        print(
            f"\n✓  Thermo sensitivity: max Δeff = {max_thermo_eff_sens:.4%} per 1% parameter change"
        )

    # Check for constraint activation
    active_constraints = [s for s in activation["constraint_stats"] if s["pct_active"] > 0]
    if not active_constraints:
        print("\n⚠️  NO CONSTRAINTS ARE ACTIVE:")
        print("   All constraints have significant headroom.")
        print("   Search is essentially unconstrained inside bounds.")
        print("   → Consider tightening constraint limits.")
    else:
        print(f"\n✓  Active constraints: {[s['constraint'] for s in active_constraints]}")

    print("\nResults saved to sensitivity_analysis.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
