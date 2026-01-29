#!/usr/bin/env python3
"""Generate training data for Gear Surrogate Model (V1).

Saves:
- X: Features (N x 8)
- y: Residuals (N x 1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from larrak2.core.encoding import bounds, random_candidate
from larrak2.surrogate.features import extract_gear_features_v1, get_gear_schema_v1


def synthetic_gear_truth(x: np.ndarray) -> float:
    """Simulate High-Fidelity Gear Loss.
    
    Placeholder: returns a synthetic correction based on features.
    """
    # Features: [radius, c1..c7]
    feats = extract_gear_features_v1(x)
    r = feats[0]
    c1 = feats[1]
    
    # Fake logic: Large radius + high c1 -> extra nonlinear loss
    residual = 0.05 * (r / 50.0) * (c1 ** 2)
    return residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="data/surrogate_v1/gear")
    parser.add_argument("--use-sim", action="store_true", help="Run actual CalculiX simulation")
    parser.add_argument("--template", type=str, default="templates/gear_case.inp", help="CalculiX template inp")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    runner = None
    if args.use_sim:
        from larrak2.adapters.calculix import CalculiXRunner
        runner = CalculiXRunner(template_path=args.template)
    
    print(f"Generating {args.n_samples} samples for Gear Surrogate...")
    
    X_list = []
    y_list = []
    
    for i in range(args.n_samples):
        # Sample full candidate
        x = random_candidate()
        
        # Extract specific features for this model
        feats = extract_gear_features_v1(x)
        
        if args.use_sim and runner:
            # Map features to CCX params
            # feats: [radius, c1..c7]
            params = {
                "base_radius": feats[0],
            }
            # Add coeffs c1..c7
            for j in range(1, 8):
                params[f"c{j}"] = feats[j]
                
            res = runner.execute(outdir, f"job_{i:04d}", params)
            val = res.get("max_stress", None)
            
            if val is None:
                print(f"Sample {i} failed. Skipping.")
                continue
                
            resid = val
        else:
            resid = synthetic_gear_truth(x)
        
        X_list.append(feats)
        y_list.append(resid)
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    np.savez_compressed(outdir / "train.npz", X=X, y=y)
    print(f"Saved to {outdir / 'train.npz'}")
    
    # Check schema consistency
    schema = get_gear_schema_v1()
    print(f"Schema Hash: {schema._hash}")


if __name__ == "__main__":
    main()
