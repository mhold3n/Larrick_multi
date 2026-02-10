#!/usr/bin/env python3
"""Run strict multi-fidelity Staged Pareto optimization.

Wrapper around larrak2.promote.staged.StagedWorkflow.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from larrak2.promote.staged import StagedWorkflow


def main():
    parser = argparse.ArgumentParser(description="Run Staged Multi-Fidelity Pareto Optimization")
    parser.add_argument("--pop", type=int, default=100, help="Stage 1 Population size")
    parser.add_argument("--gen", type=int, default=50, help="Stage 1 Generations")
    parser.add_argument(
        "--promote", type=int, default=20, help="Number of candidates to promote to Fid 2"
    )
    parser.add_argument("--rpm", type=float, default=3000.0, help="Engine RPM")
    parser.add_argument("--torque", type=float, default=200.0, help="Torque (Nm)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--outdir", type=str, default="results_staged", help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.outdir)
    # StagedWorkflow creates the dir

    workflow = StagedWorkflow(outdir=output_dir, rpm=args.rpm, torque=args.torque, seed=args.seed)

    t0 = time.time()

    # 1. Stage 1
    archive_s1 = workflow.run_stage1(args.pop, args.gen)

    # 2. Promotion
    # Note: promote arg is "k_promote"
    archive_s2 = workflow.run_promotion(archive_s1, args.promote)

    # 3. Stage 3 (Refinement)
    # Usually same pop/gen as stage 1 or smaller?
    # Let's default to same for now or allow args.
    archive_s3 = workflow.run_stage3(archive_s2, args.pop, args.gen)

    t_total = time.time() - t0

    # Summary
    s1_vals = archive_s1.to_arrays()[1]
    s2_vals = archive_s2.to_arrays()[1]
    s3_vals = archive_s3.to_arrays()[1]

    summary = {
        "config": vars(args),
        "metrics": {
            "total_time_s": t_total,
            "stage1_n": len(s1_vals),
            "stage2_n": len(s2_vals),
            "stage3_n": len(s3_vals),
            "stage1_eff_max": float(-np.min(s1_vals[:, 0])) if len(s1_vals) > 0 else 0,
            "stage3_eff_max": float(-np.min(s3_vals[:, 0])) if len(s3_vals) > 0 else 0,
        },
    }

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWorkflow complete in {t_total:.2f}s")
    print(f"Stage 1 Max Eff: {summary['metrics']['stage1_eff_max']:.1%}")
    print(f"Stage 3 Max Eff: {summary['metrics']['stage3_eff_max']:.1%}")


if __name__ == "__main__":
    main()
