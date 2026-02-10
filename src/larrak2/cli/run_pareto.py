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
import time
from pathlib import Path

import numpy as np

from ..core.archive_io import save_archive
from ..core.constants import MODEL_VERSION_GEAR_V1, MODEL_VERSION_THERMO_V1
from ..core.constraints import get_constraint_names, get_constraint_scales
from ..core.encoding import ENCODING_VERSION
from ..core.types import BreathingConfig


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

    # Breathing/BC/timing inputs for OpenFOAM NN (fidelity>=2)
    parser.add_argument("--bore-mm", type=float, default=80.0)
    parser.add_argument("--stroke-mm", type=float, default=90.0)
    parser.add_argument("--intake-port-area-m2", type=float, default=4.0e-4)
    parser.add_argument("--exhaust-port-area-m2", type=float, default=4.0e-4)
    parser.add_argument("--p-manifold-pa", type=float, default=101325.0)
    parser.add_argument("--p-back-pa", type=float, default=101325.0)
    parser.add_argument("--overlap-deg", type=float, default=0.0)
    parser.add_argument("--intake-open-deg", type=float, default=0.0)
    parser.add_argument("--intake-close-deg", type=float, default=0.0)
    parser.add_argument("--exhaust-open-deg", type=float, default=0.0)
    parser.add_argument("--exhaust-close-deg", type=float, default=0.0)

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

    breathing = BreathingConfig(
        bore_mm=args.bore_mm,
        stroke_mm=args.stroke_mm,
        intake_port_area_m2=args.intake_port_area_m2,
        exhaust_port_area_m2=args.exhaust_port_area_m2,
        p_manifold_Pa=args.p_manifold_pa,
        p_back_Pa=args.p_back_pa,
        overlap_deg=args.overlap_deg,
        intake_open_deg=args.intake_open_deg,
        intake_close_deg=args.intake_close_deg,
        exhaust_open_deg=args.exhaust_open_deg,
        exhaust_close_deg=args.exhaust_close_deg,
    )

    ctx = EvalContext(
        rpm=args.rpm,
        torque=args.torque,
        fidelity=args.fidelity,
        seed=args.seed,
        breathing=breathing,
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
    G_result = getattr(result, "G", None)

    # Handle case where no solutions found
    if X is None or len(X) == 0:
        n_pareto = 0
        G = np.array([]).reshape(0, problem.N_CONSTR)
        # Save empty arrays
        np.save(output_dir / "pareto_X.npy", np.array([]).reshape(0, problem.n_var))
        np.save(output_dir / "pareto_F.npy", np.array([]).reshape(0, problem.n_obj))
        np.save(output_dir / "pareto_G.npy", G)
    else:
        # Get constraint values for Pareto solutions (reuse result.G if present)
        n_pareto = X.shape[0]
        if G_result is not None and getattr(G_result, "shape", (0,))[0] == n_pareto:
            G = G_result
        else:
            from ..core.evaluator import evaluate_candidate

            G = np.zeros((n_pareto, problem.N_CONSTR), dtype=np.float64)
            for i, x in enumerate(X):
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
        best_eta_comb = float(1.0 - F[:, 0].min())
        best_eta_exp = float(1.0 - F[:, 1].min())
        best_eta_gear = float(1.0 - F[:, 2].min())
        eta_total = (1.0 - F[:, 0]) * (1.0 - F[:, 1]) * (1.0 - F[:, 2])
        best_eta_total = float(np.max(eta_total))
    else:
        f_min = [0.0, 0.0, 0.0]
        f_max = [0.0, 0.0, 0.0]
        best_eta_comb = 0.0
        best_eta_exp = 0.0
        best_eta_gear = 0.0
        best_eta_total = 0.0

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
        "encoding_version": ENCODING_VERSION,
        "model_versions": {
            "thermo_v1": MODEL_VERSION_THERMO_V1,
            "gear_v1": MODEL_VERSION_GEAR_V1,
        },
        "n_obj": problem.n_obj,
        "n_constr": problem.N_CONSTR,
        "constraint_names": get_constraint_names(args.fidelity),
        "constraint_scales": get_constraint_scales(),
        "F_min": f_min,
        "F_max": f_max,
        "n_feasible": int(np.sum(np.all(G <= 0, axis=1))) if len(G) > 0 else 0,
        "feasible_fraction": float(np.sum(np.all(G <= 0, axis=1)) / n_pareto)
        if n_pareto > 0
        else 0.0,
        "best_eta_comb": best_eta_comb,
        "best_eta_exp": best_eta_exp,
        "best_eta_gear": best_eta_gear,
        "best_eta_total": best_eta_total,
    }

    save_archive(
        output_dir,
        X if X is not None else np.array([]),
        F if F is not None else np.array([]),
        G,
        summary,
    )

    if args.verbose:
        print(f"\nCompleted in {t_elapsed:.1f}s")
        print(f"Pareto front: {n_pareto} solutions")
        print(f"Feasible: {summary['n_feasible']}")
        print(
            "Best (component) efficiencies: "
            f"η_comb={summary['best_eta_comb']:.3f}, "
            f"η_exp={summary['best_eta_exp']:.3f}, "
            f"η_gear={summary['best_eta_gear']:.3f}, "
            f"η_total={summary['best_eta_total']:.3f}"
        )
        print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
