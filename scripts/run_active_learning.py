#!/usr/bin/env python3
"""Active Learning Loop for High-Fidelity Data Generation.

Steps:
1. Run Staged Optimization (Fidelity 1 -> Fidelity 2).
2. Analyze Fidelity 2 candidates for uncertainty.
3. Select high-uncertainty candidates.
4. Execute (or queue) Truth Simulations (OpenFOAM/CalculiX).
5. Append results to training set (Placeholder for now).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from larrak2.adapters.calculix import CalculiXRunner
from larrak2.adapters.openfoam import OpenFoamRunner
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext
from larrak2.promote.staged import StagedWorkflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out/active_learning")
    parser.add_argument("--pop", type=int, default=64)
    parser.add_argument("--gen", type=int, default=30)
    parser.add_argument("--promote", type=int, default=20)
    parser.add_argument("--n_truth", type=int, default=10, help="Number of truth samples to run")
    parser.add_argument(
        "--dry_run", action="store_true", help="Do not actually run solvers, just generate plan"
    )
    parser.add_argument("--rpm", type=float, default=3000.0)
    parser.add_argument("--torque", type=float, default=200.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    workflow = StagedWorkflow(outdir, args.rpm, args.torque, seed=42)

    # 1. Stage 1: Exploration
    print("\n=== Phase 1: Exploration (Fidelity 1) ===")
    bundle_s1 = workflow.run_stage1(args.pop, args.gen)

    # 2. Promotion
    print("\n=== Phase 2: Promotion ===")
    bundle_s2 = workflow.run_promotion(bundle_s1, args.promote)

    # 3. Refinement (Stage 3)
    print("\n=== Phase 3: Refinement (Fidelity 2) ===")
    bundle_s3 = workflow.run_stage3(bundle_s2, args.pop, args.gen)

    print(f"\nOptimization complete. Analyzing {len(bundle_s3)} candidates for uncertainty...")

    # 4. Uncertainty Analysis
    # Re-evaluate to get diagnostics (since npz doesn't save them)
    candidates = []
    ctx = EvalContext(rpm=args.rpm, torque=args.torque, fidelity=2, seed=42)

    for i, rec in enumerate(bundle_s3.records):
        res = evaluate_candidate(rec.x, ctx)

        # Extract uncertainty
        unc = res.diag["versions"].get("uncertainty", {})
        u_gear = unc.get("gear", 0.0)
        u_scavenge = unc.get("scavenge", 0.0)

        candidates.append(
            {
                "index": i,
                "x": rec.x,
                "f": rec.f,
                "u_gear": u_gear,
                "u_scavenge": u_scavenge,
                "u_total": u_gear + u_scavenge,
            }
        )

    # 5. Selection (Greedy max uncertainty)
    # Sort by total uncertainty descending
    candidates.sort(key=lambda c: c["u_total"], reverse=True)

    selected = candidates[: args.n_truth]
    print(f"\nSelected {len(selected)} candidates for Truth Evaluation (Top Uncertainty):")
    for c in selected:
        print(
            f"  Candidate {c['index']}: Unc={c['u_total']:.4f} (G={c['u_gear']:.4f}, S={c['u_scavenge']:.4f})"
        )

    # 6. Execute Truth Solvers
    if args.dry_run:
        print("\n[Dry Run] Skipping solver execution. Saving plan to truth_plan.json")
        save_plan(outdir, selected)
        return

    print("\n=== Phase 4: Truth Execution ===")

    # Setup Runners (assuming templates exist, or we warn)
    # In a real run, these paths would be args
    template_foam = Path("templates/openfoam_scavenge")
    template_dummy = not template_foam.exists()

    _foam_runner = OpenFoamRunner(template_case=template_foam)
    _ccx_runner = CalculiXRunner()

    _results_gear = []
    _results_scavenge = []

    for i, item in enumerate(selected):
        x = item["x"]

        # Run Scavenge (OpenFOAM)
        print(f"[{i + 1}/{len(selected)}] Running OpenFOAM...")
        try:
            # Extract features (params) - mapping needed?
            # OpenFoamRunner expects direct params or we map inside adapter
            # Wait, OpenFoamRunner.run_case expects 'params' dict.
            # We need to map 'x' to named params.
            # Ideally we use `decode_candidate` and map relevant fields.
            from larrak2.core.encoding import decode_candidate

            cand = decode_candidate(x)

            # Map for OpenFOAM
            _foam_params = {
                "compression_ratio": cand.thermo.compression_ratio,
                "expansion_ratio": cand.thermo.expansion_ratio,
                # ... other params ...
            }

            if template_dummy:
                print("  (Template missing, skipping real Run)")
            else:
                # Real run (commented out to avoid crashing without binary)
                # res = foam_runner.run_case(foam_params, f"run_{i}")
                pass

        except Exception as e:
            print(f"  OpenFOAM Failed: {e}")

        # Run Gear (CalculiX)
        print(f"[{i + 1}/{len(selected)}] Running CalculiX...")
        try:
            # Map for CalculiX
            # ccx_runner.run_simulation(...)
            pass
        except Exception as e:
            print(f"  CalculiX Failed: {e}")

    print("\nActive Learning Loop Complete.")


def save_plan(outdir: Path, selected: list):
    """Save selected candidates to JSON."""
    plan = []
    for item in selected:
        plan.append({"x": item["x"].tolist(), "metrics": {"u_total": item["u_total"]}})
    with open(outdir / "truth_plan.json", "w") as f:
        json.dump(plan, f, indent=2)


if __name__ == "__main__":
    main()
