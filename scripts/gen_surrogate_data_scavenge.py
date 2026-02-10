#!/usr/bin/env python3
"""Generate training data for Scavenge Surrogate Model (V1).

Saves:
- X: Features (N x 4)
- y: Residuals (N x 1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from larrak2.core.encoding import random_candidate
from larrak2.surrogate.features import extract_scavenge_features_v1, get_scavenge_schema_v1


def synthetic_scavenge_truth(x: np.ndarray) -> float:
    """Simulate High-Fidelity Scavenging Efficiency."""
    feats = extract_scavenge_features_v1(x)
    # feats: [comp, exp, center, width]

    comp = feats[0]
    width = feats[3]

    # Fake logic: High compression + narrow width -> efficiency penalty
    residual = -0.02 * (comp / 50.0) * (50.0 / (width + 1e-6))
    return residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="data/surrogate_v1/scavenge")
    parser.add_argument("--use-sim", action="store_true", help="Run actual OpenFOAM simulation")
    parser.add_argument(
        "--template", type=str, default="templates/scavenge_case", help="OpenFOAM template dir"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runner = None
    if args.use_sim:
        from larrak2.adapters.openfoam import OpenFoamRunner

        runner = OpenFoamRunner(template_dir=args.template)

    print(f"Generating {args.n_samples} samples for Scavenge Surrogate...")

    X_list = []
    y_list = []

    for i in range(args.n_samples):
        x = random_candidate()
        feats = extract_scavenge_features_v1(x)

        if args.use_sim and runner:
            # Prepare params
            # Map feats to named params expected by template
            params = {
                "compression_ratio": feats[0],
                "expansion_ratio": feats[1],
                "hr_center": feats[2],
                "hr_width": feats[3],
            }

            run_dir = outdir / f"run_{i:04d}"
            res = runner.execute(run_dir, params)

            # Check success (assuming we want efficiency as target)
            val = res.get("scavenging_efficiency", None)

            if val is None:
                print(f"Sample {i} failed or no result. Skipping.")
                continue

            # Calculate residual: Truth - Baseline(x)
            # For now, we are treating 'y' as the residual directly?
            # Wait, the training script expects residuals.
            # If we run simulation, we get absolute Truth.
            # We need to compute Baseline to get residual.
            # Assuming we can run baseline here or just save Truth and post-process.
            # Let's save Truth as y for now, but label it?
            # Or stick to the convention y = residual.
            # We need to call baseline eval_thermo(x) (fid=1)

            # Import baseline evaluator
            from larrak2.core.evaluator import EvalContext, evaluate_candidate

            # Construct context
            ctx = EvalContext(rpm=3000, torque=200, fidelity=1, seed=1)
            # Note: rpm/torque might affect baseline?

            # Baseline
            res_base = evaluate_candidate(x, ctx)
            eff_base = res_base.F[0] * -1  # It returns -efficiency

            resid = val - eff_base
        else:
            resid = synthetic_scavenge_truth(x)

        X_list.append(feats)
        y_list.append(resid)

        if (i + 1) % 10 == 0:
            print(f"Sample {i + 1}/{args.n_samples}")

    X = np.array(X_list)
    y = np.array(y_list)

    np.savez_compressed(outdir / "train.npz", X=X, y=y)
    print(f"Saved to {outdir / 'train.npz'}")

    schema = get_scavenge_schema_v1()
    print(f"Schema Hash: {schema._hash}")


if __name__ == "__main__":
    main()
