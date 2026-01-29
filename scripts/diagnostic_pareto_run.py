#!/usr/bin/env python3
"""Diagnostic Pareto run at fidelity=1.

Collects statistics to assess readiness for:
- Constraint tightening
- Fidelity promotion
- Adding objectives

Usage:
    python scripts/diagnostic_pareto_run.py --pop 64 --gen 50 --fidelity 1

Outputs:
    diagnostic_results/
    ├── run_summary.json      # Overall stats
    ├── generation_stats.csv  # Per-generation metrics
    ├── pareto_front.csv      # Final Pareto solutions
    └── constraint_dist.csv   # Constraint violation distribution
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnostic Pareto run")
    parser.add_argument("--pop", type=int, default=64, help="Population size")
    parser.add_argument("--gen", type=int, default=50, help="Number of generations")
    parser.add_argument("--rpm", type=float, default=3000.0, help="Engine speed")
    parser.add_argument("--torque", type=float, default=200.0, help="Torque demand")
    parser.add_argument("--fidelity", type=int, default=1, help="Model fidelity")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--outdir", type=str, default="diagnostic_results", help="Output dir")
    args = parser.parse_args(argv)

    # Lazy imports
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    from larrak2.adapters.pymoo_problem import ParetoProblem
    from larrak2.core.evaluator import evaluate_candidate
    from larrak2.core.types import EvalContext

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
            self.eval_times = []

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

            self.gen_stats.append({
                "generation": gen,
                "n_feasible": int(n_feasible),
                "feasible_fraction": float(feasible_frac),
                "mean_max_violation": float(mean_max_violation),
                "median_max_violation": float(median_max_violation),
                "f0_mean": float(f0_mean),
                "f1_mean": float(f1_mean),
                "n_pareto": n_pareto,
            })

    callback = DiagnosticCallback()
    algorithm = NSGA2(pop_size=args.pop)
    termination = get_termination("n_gen", args.gen)

    print(f"Starting diagnostic Pareto run:")
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
    G_pareto = np.zeros((n_pareto, problem.N_CONSTR)) if n_pareto > 0 else np.array([]).reshape(0, problem.N_CONSTR)
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
        headers = ["generation", "n_feasible", "feasible_fraction", "mean_max_violation",
                   "median_max_violation", "f0_mean", "f1_mean", "n_pareto"]
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

    # Constraint distribution CSV
    if n_pareto > 0:
        with open(output_dir / "constraint_dist.csv", "w") as f:
            headers = ["idx"] + [f"g{j}" for j in range(problem.N_CONSTR)] + ["max_violation", "feasible"]
            f.write(",".join(headers) + "\n")
            for i in range(n_pareto):
                max_viol = max(G_pareto[i])
                feasible = "1" if np.all(G_pareto[i] <= 0) else "0"
                vals = [str(i)] + [f"{G_pareto[i, j]:.6f}" for j in range(problem.N_CONSTR)] + [f"{max_viol:.6f}", feasible]
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
    print(f"Feasible in Pareto: {pareto_feasible}/{n_pareto} ({100*pareto_feasible/n_pareto:.1f}%)" if n_pareto > 0 else "N/A")
    print()
    if n_pareto > 0:
        print(f"Best efficiency: {-F[:, 0].min():.4f}")
        print(f"Best loss: {F[:, 1].min():.2e} W")
    print()
    print("Generation progression (first/mid/last):")
    if callback.gen_stats:
        for idx in [0, len(callback.gen_stats)//2, -1]:
            s = callback.gen_stats[idx]
            print(f"  Gen {s['generation']:3d}: feasible={s['feasible_fraction']:.1%}, "
                  f"max_viol={s['mean_max_violation']:.2f}, pareto={s['n_pareto']}")
    print()
    print(f"Results written to: {output_dir}/")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
