"""Pareto optimization CLI runner.

Usage:
    python -m larrak2.cli.run_pareto --pop 64 --gen 50 --rpm 3000 --torque 200
    python -m larrak2.cli.run_pareto --fidelity 1 --seed 123 --outdir ./results

Outputs:
    pareto_X.npy  - Decision vectors of Pareto front
    pareto_F.npy  - Objective values of Pareto front
    pareto_G.npy  - Constraint values of Pareto front
    summary.json  - Run metadata and statistics
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    """Run Pareto optimization.

    Args:
        argv: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(description="Run multi-objective Pareto optimization")
    parser.add_argument("--pop", type=int, default=64, help="Population size")
    parser.add_argument("--gen", type=int, default=100, help="Number of generations")
    parser.add_argument("--rpm", type=float, default=3000.0, help="Engine speed (rpm)")
    parser.add_argument("--torque", type=float, default=200.0, help="Torque demand (Nm)")
    parser.add_argument(
        "--fidelity", type=int, default=0, choices=[0, 1, 2], help="Model fidelity (0=toy, 1=v1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", "--outdir", type=str, default=".", dest="output", help="Output directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args(argv)

    # Import here to avoid loading pymoo at module level
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    from ..adapters.pymoo_problem import ParetoProblem
    from ..core.types import EvalContext

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = EvalContext(
        rpm=args.rpm,
        torque=args.torque,
        fidelity=args.fidelity,
        seed=args.seed,
    )

    problem = ParetoProblem(ctx=ctx)

    algorithm = NSGA2(pop_size=args.pop)
    termination = get_termination("n_gen", args.gen)

    if args.verbose:
        print(f"Starting NSGA-II: pop={args.pop}, gen={args.gen}")
        print(f"Context: rpm={args.rpm}, torque={args.torque}, fidelity={args.fidelity}")

    # Run optimization
    t_start = time.perf_counter()

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=args.seed,
        verbose=args.verbose,
    )

    t_elapsed = time.perf_counter() - t_start

    # Extract Pareto front (may be None if no feasible solutions)
    X = result.X
    F = result.F

    # Handle case where no solutions found
    if X is None or len(X) == 0:
        n_pareto = 0
        G = np.array([]).reshape(0, problem.N_CONSTR)
        # Save empty arrays
        np.save(output_dir / "pareto_X.npy", np.array([]).reshape(0, problem.n_var))
        np.save(output_dir / "pareto_F.npy", np.array([]).reshape(0, problem.n_obj))
        np.save(output_dir / "pareto_G.npy", G)
    else:
        # Get constraint values for Pareto solutions
        n_pareto = X.shape[0]
        G = np.zeros((n_pareto, problem.N_CONSTR), dtype=np.float64)
        for i, x in enumerate(X):
            from ..core.evaluator import evaluate_candidate

            res = evaluate_candidate(x, ctx)
            G[i] = res.G

        # Save results
        np.save(output_dir / "pareto_X.npy", X)
        np.save(output_dir / "pareto_F.npy", F)
        np.save(output_dir / "pareto_G.npy", G)

    # Summary
    if n_pareto > 0 and F is not None:
        f_min = F.min(axis=0).tolist()
        f_max = F.max(axis=0).tolist()
        best_efficiency = float(-F[:, 0].min())
        best_loss = float(F[:, 1].min())
    else:
        f_min = [0.0, 0.0]
        f_max = [0.0, 0.0]
        best_efficiency = 0.0
        best_loss = 0.0

    summary = {
        "n_pareto": n_pareto,
        "n_evals": problem.n_evals,
        "elapsed_s": t_elapsed,
        "pop_size": args.pop,
        "n_gen": args.gen,
        "rpm": args.rpm,
        "torque": args.torque,
        "fidelity": args.fidelity,
        "seed": args.seed,
        "F_min": f_min,
        "F_max": f_max,
        "n_feasible": int(np.sum(np.all(G <= 0, axis=1))) if len(G) > 0 else 0,
        "feasible_fraction": float(np.sum(np.all(G <= 0, axis=1)) / n_pareto)
        if n_pareto > 0
        else 0.0,
        "best_efficiency": best_efficiency,
        "best_loss": best_loss,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.verbose:
        print(f"\nCompleted in {t_elapsed:.1f}s")
        print(f"Pareto front: {n_pareto} solutions")
        print(f"Feasible: {summary['n_feasible']}")
        print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
