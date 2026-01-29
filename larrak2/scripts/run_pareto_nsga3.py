#!/usr/bin/env python3
"""Run higher-dimensional Pareto optimization using NSGA-III.

This script runs the optimization with 3 objectives:
1. Efficiency (maximize -> minimize negative)
2. Gear Loss (minimize)
3. Max Planet Radius (minimize)

Uses NSGA-III with Das-Dennis reference directions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from larrak2.adapters.pymoo_problem import ParetoProblem
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def main():
    parser = argparse.ArgumentParser(description="Run 3-obj NSGA-III Pareto Optimization")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--gen", type=int, default=50, help="Number of generations")
    parser.add_argument("--rpm", type=float, default=3000.0, help="Engine RPM")
    parser.add_argument("--torque", type=float, default=200.0, help="Torque (Nm)")
    parser.add_argument("--fidelity", type=int, default=1, help="Fidelity level")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--outdir", type=str, default="results_nsga3", help="Output directory")
    parser.add_argument("--partitions", type=int, default=12, help="Ref dir partitions")
    
    args = parser.parse_args()
    
    # Setup context
    ctx = EvalContext(
        rpm=args.rpm,
        torque=args.torque,
        fidelity=args.fidelity,
        seed=args.seed,
    )
    
    print(f"Starting NSGA-III run: pop={args.pop}, gen={args.gen}, fidelity={args.fidelity}")
    
    # Setup problem
    problem = ParetoProblem(ctx=ctx)
    if problem.N_OBJ != 3:
        raise ValueError(f"ParetoProblem must have 3 objectives, got {problem.N_OBJ}")
        
    # Setup reference directions
    # Das-Dennis directions for 3 objectives
    ref_dirs = get_reference_directions(
        "das-dennis", 
        3, 
        n_partitions=args.partitions
    )
    print(f"Reference directions: {len(ref_dirs)} points")
    
    if args.pop < len(ref_dirs):
        print(f"WARNING: Population size ({args.pop}) < Reference directions ({len(ref_dirs)})")
        print("Increasing population size to match reference directions.")
        args.pop = len(ref_dirs)
        
    # Algorithm
    algorithm = NSGA3(
        pop_size=args.pop,
        ref_dirs=ref_dirs,
        prob_neighbor_mating=0.7,
    )
    
    # Termination
    termination = get_termination("n_gen", args.gen)
    
    # Run
    t0 = time.time()
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=args.seed,
        verbose=True,
    )
    t_total = time.time() - t0
    
    # Process results
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X = result.X
    F = result.F
    
    if X is None or len(X) == 0:
        print("No feasible solutions found!")
        n_pareto = 0
        G = np.array([])
    else:
        n_pareto = len(X)
        # Compute constraints G specifically for Pareto set
        G = np.zeros((n_pareto, problem.N_CONSTR), dtype=np.float64)
        diag_list = []
        for i, x in enumerate(X):
            res = evaluate_candidate(x, ctx)
            G[i] = res.G
            # Store some diagnostics for the Pareto front
            diag_list.append({
                "efficiency": res.diag["metrics"]["efficiency"],
                "loss_total": res.diag["metrics"]["loss_total"],
                "max_planet_radius": res.diag["metrics"]["max_planet_radius"],
                "p_max_bar": res.diag["thermo"]["p_max"] / 1e5,
            })
            
        np.save(output_dir / "pareto_X.npy", X)
        np.save(output_dir / "pareto_F.npy", F)
        np.save(output_dir / "pareto_G.npy", G)
        
        # Save readable summary
        with open(output_dir / "pareto_metrics.json", "w") as f:
            json.dump(diag_list, f, indent=2)

    # Global summary
    summary = {
        "config": vars(args),
        "metrics": {
            "total_time_s": t_total,
            "n_evals": problem.n_evals,
            "n_pareto": n_pareto,
        },
        "objectives_min": F.min(axis=0).tolist() if n_pareto > 0 else [],
        "objectives_max": F.max(axis=0).tolist() if n_pareto > 0 else [],
    }
    
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Run complete in {t_total:.2f}s. Results in {output_dir}/")
    if n_pareto > 0:
        print(f"Pareto size: {n_pareto}")
        print(f"Eff range: {-F[:,0].min():.1%} to {-F[:,0].max():.1%}")
        print(f"Loss range: {F[:,1].min():.1f} to {F[:,1].max():.1f} W")
        print(f"Radius range: {F[:,2].min():.1f} to {F[:,2].max():.1f} mm")


if __name__ == "__main__":
    main()
