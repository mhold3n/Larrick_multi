"""Analysis Workflows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.adapters.pymoo_problem import ParetoProblem
from larrak2.analysis.sensitivity import constraint_activation_report, sensitivity_scan
from larrak2.core.encoding import bounds, mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def sensitivity_workflow(args: Any) -> int:
    """Run sensitivity analysis and constraint check."""
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

    # Load Pareto front from diagnostic run if available
    # Defaulting to diagnostic_results (default outdir for diagnostic run)
    diagnostic_dir = Path("diagnostic_results")
    pareto_file = diagnostic_dir / "pareto_X.npy"

    if pareto_file.exists():
        X_pareto = np.load(pareto_file)
        print(f"Analyzing {len(X_pareto)} Pareto solutions from {diagnostic_dir}/")
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
    if thermo_sens:
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

    return 0


def diagnostic_workflow(args: Any) -> int:
    """Run diagnostic Pareto optimization."""
    # Lazy imports
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = EvalContext(
        rpm=args.rpm,
        torque=args.torque,
        fidelity=args.fidelity,
        seed=args.seed,
    )

    problem = ParetoProblem(ctx=ctx)

    # Custom callback to collect per-generation stats
    class DiagnosticCallback(Callback):
        def __init__(self):
            super().__init__()
            self.gen_stats = []

        def notify(self, algorithm):
            gen = algorithm.n_gen
            pop = algorithm.pop

            # Get F and G for current population
            F = pop.get("F")
            G = pop.get("G")

            # Count feasible (all G <= 0)
            feasible_mask = np.all(G <= 0, axis=1)
            n_feasible = np.sum(feasible_mask)
            feasible_frac = n_feasible / len(pop)

            # Constraint violation stats
            max_violation = np.max(G, axis=1)  # Per-individual max violation
            mean_max_violation = np.mean(max_violation)
            median_max_violation = np.median(max_violation)

            # Objective stats
            f0_mean = np.mean(F[:, 0])  # -efficiency
            f1_mean = np.mean(F[:, 1])  # loss

            # Pareto size (non-dominated)
            opt = algorithm.opt
            n_pareto = len(opt) if opt is not None else 0

            self.gen_stats.append(
                {
                    "generation": gen,
                    "n_feasible": int(n_feasible),
                    "feasible_fraction": float(feasible_frac),
                    "mean_max_violation": float(mean_max_violation),
                    "median_max_violation": float(median_max_violation),
                    "f0_mean": float(f0_mean),
                    "f1_mean": float(f1_mean),
                    "n_pareto": n_pareto,
                }
            )

    callback = DiagnosticCallback()
    algorithm = NSGA2(pop_size=args.pop)
    termination = get_termination("n_gen", args.gen)

    print("Starting diagnostic Pareto run:")
    print(f"  pop={args.pop}, gen={args.gen}, fidelity={args.fidelity}")
    print(f"  rpm={args.rpm}, torque={args.torque}, seed={args.seed}")
    print()

    t_start = time.perf_counter()

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=args.seed,
        callback=callback,
        verbose=True,
    )

    t_elapsed = time.perf_counter() - t_start

    # Extract final Pareto front
    X = result.X if result.X is not None else np.array([]).reshape(0, problem.n_var)
    F = result.F if result.F is not None else np.array([]).reshape(0, 2)
    n_pareto = len(X)

    # Get G for Pareto solutions
    G_pareto = (
        np.zeros((n_pareto, problem.N_CONSTR))
        if n_pareto > 0
        else np.array([]).reshape(0, problem.N_CONSTR)
    )
    eval_times = []
    for i, x in enumerate(X):
        t0 = time.perf_counter()
        res = evaluate_candidate(x, ctx)
        eval_times.append(time.perf_counter() - t0)
        G_pareto[i] = res.G

    # Count feasible in Pareto
    if n_pareto > 0:
        pareto_feasible = int(np.sum(np.all(G_pareto <= 0, axis=1)))
    else:
        pareto_feasible = 0

    # Compute summary stats
    avg_eval_time_ms = np.mean(eval_times) * 1000 if eval_times else 0.0
    std_eval_time_ms = np.std(eval_times) * 1000 if eval_times else 0.0

    summary = {
        "config": {
            "pop": args.pop,
            "gen": args.gen,
            "rpm": args.rpm,
            "torque": args.torque,
            "fidelity": args.fidelity,
            "seed": args.seed,
        },
        "results": {
            "total_time_s": float(t_elapsed),
            "n_evals": problem.n_evals,
            "n_pareto": n_pareto,
            "pareto_feasible": pareto_feasible,
            "pareto_feasible_fraction": pareto_feasible / n_pareto if n_pareto > 0 else 0.0,
            "avg_eval_time_ms": float(avg_eval_time_ms),
            "std_eval_time_ms": float(std_eval_time_ms),
        },
        "objectives": {
            "best_efficiency": float(-F[:, 0].min()) if n_pareto > 0 else 0.0,
            "worst_efficiency": float(-F[:, 0].max()) if n_pareto > 0 else 0.0,
            "best_loss": float(F[:, 1].min()) if n_pareto > 0 else 0.0,
            "worst_loss": float(F[:, 1].max()) if n_pareto > 0 else 0.0,
        },
        "final_generation": callback.gen_stats[-1] if callback.gen_stats else {},
    }

    # Write outputs
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generation stats CSV
    with open(output_dir / "generation_stats.csv", "w") as f:
        headers = [
            "generation",
            "n_feasible",
            "feasible_fraction",
            "mean_max_violation",
            "median_max_violation",
            "f0_mean",
            "f1_mean",
            "n_pareto",
        ]
        f.write(",".join(headers) + "\n")
        for row in callback.gen_stats:
            vals = [str(row[h]) for h in headers]
            f.write(",".join(vals) + "\n")

    # Pareto front CSV
    if n_pareto > 0:
        with open(output_dir / "pareto_front.csv", "w") as f:
            f.write("idx,f0_neg_eff,f1_loss,feasible\n")
            for i in range(n_pareto):
                feasible = "1" if np.all(G_pareto[i] <= 0) else "0"
                f.write(f"{i},{F[i, 0]},{F[i, 1]},{feasible}\n")

    # Save arrays for reuse by sensitivity analysis
    np.save(output_dir / "pareto_X.npy", X)

    # Constraint distribution CSV
    if n_pareto > 0:
        with open(output_dir / "constraint_dist.csv", "w") as f:
            headers = (
                ["idx"] + [f"g{j}" for j in range(problem.N_CONSTR)] + ["max_violation", "feasible"]
            )
            f.write(",".join(headers) + "\n")
            for i in range(n_pareto):
                max_viol = max(G_pareto[i])
                feasible = "1" if np.all(G_pareto[i] <= 0) else "0"
                vals = (
                    [str(i)]
                    + [f"{G_pareto[i, j]:.6f}" for j in range(problem.N_CONSTR)]
                    + [f"{max_viol:.6f}", feasible]
                )
                f.write(",".join(vals) + "\n")

    # Print summary
    print()
    print("=" * 60)
    print("DIAGNOSTIC RUN SUMMARY")
    print("=" * 60)
    print(f"Total time: {t_elapsed:.1f}s")
    print(f"Total evals: {problem.n_evals}")
    print(f"Avg eval time: {avg_eval_time_ms:.2f} ± {std_eval_time_ms:.2f} ms")
    print()
    print(f"Final Pareto size: {n_pareto}")
    print(
        f"Feasible in Pareto: {pareto_feasible}/{n_pareto} ({100 * pareto_feasible / n_pareto:.1f}%)"
        if n_pareto > 0
        else "N/A"
    )
    print()
    if n_pareto > 0:
        print(f"Best efficiency: {-F[:, 0].min():.4f}")
        print(f"Best loss: {F[:, 1].min():.2e} W")
    print()
    print("Generation progression (first/mid/last):")
    if callback.gen_stats:
        for idx in [0, len(callback.gen_stats) // 2, -1]:
            s = callback.gen_stats[idx]
            print(
                f"  Gen {s['generation']:3d}: feasible={s['feasible_fraction']:.1%}, "
                f"max_viol={s['mean_max_violation']:.2f}, pareto={s['n_pareto']}"
            )
    print()
    print(f"Results written to: {output_dir}/")
    print("=" * 60)

    return 0
