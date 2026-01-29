#!/usr/bin/env python3
"""Generate training data for residual surrogates.

Pipeline:
1. Sample design space (LHS).
2. Evaluate at Fidelity 1 (Baseline).
3. Evaluate at "Truth" (Synthetic for now).
4. Compute residuals (Truth - Baseline).
5. Save dataset.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from pymoo.operators.sampling.lhs import LHS

from larrak2.core.encoding import bounds, N_TOTAL
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def synthetic_truth_function(x: np.ndarray, res_v1: any) -> tuple[float, float]:
    """Simulate high-fidelity physics by adding structure to v1 results.
    
    This acts as a placeholder for OpenFOAM/CalculiX.
    
    Returns:
        (eff_truth, loss_truth)
    """
    # Extract some meaningful parameters for synthetic noise
    # x[0] -> compression ratio
    # x[1] -> alpha (burn duration)
    
    eff_v1 = res_v1.diag["metrics"]["efficiency_raw"]
    loss_v1 = res_v1.diag["metrics"]["loss_raw"]
    
    # 1. Efficiency Correction
    # Truth has a "hump" depending on compression ratio not captured by V1
    # residual = 0.02 * sin(10 * x[0])
    delta_eff = 0.02 * np.sin(0.1 * x[0])  # Scaled for raw values (deg)
    
    # 2. Loss Correction
    # Truth has extra loss components non-linear in geometry
    # residual = 5.0 * (x[2]**2)
    delta_loss = 0.01 * (x[2] ** 2) # Scaled
    
    eff_truth = eff_v1 + delta_eff
    loss_truth = loss_v1 + delta_loss
    
    return eff_truth, loss_truth


def main():
    parser = argparse.ArgumentParser(description="Generate Surrogate Training Data")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="data/surrogate_v1", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    rng = np.random.default_rng(args.seed)
    outpath = Path(args.outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.n_samples} samples...")
    
    # 1. Sampling (LHS)
    # Generate scaled samples using bounds
    xl, xu = bounds()
    
    # Simple LHS implementation
    X_lhs = np.zeros((args.n_samples, N_TOTAL))
    for col in range(N_TOTAL):
        intervals = np.linspace(0, 1, args.n_samples + 1)
        points = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(points)
        X_lhs[:, col] = points
        
    # Scale to bounds
    X = xl + X_lhs * (xu - xl)
        
    # 2. Evaluation Loop
    ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1)
    
    data_records = []
    
    print("Evaluating Physics...")
    t0 = time.time()
    
    for i, x in enumerate(X):
        if i % 10 == 0:
            print(f"  {i}/{args.n_samples}")
            
        # Baseline
        res_v1 = evaluate_candidate(x, ctx)
        
        # Truth (Synthetic)
        eff_true, loss_true = synthetic_truth_function(x, res_v1)
        
        # Residuals
        eff_v1 = res_v1.diag["metrics"]["efficiency_raw"]
        loss_v1 = res_v1.diag["metrics"]["loss_raw"]
        
        delta_eff = eff_true - eff_v1
        delta_loss = loss_true - loss_v1
        
        # Store features and targets
        # Features: X (design vars) + maybe operating conditions?
        # For now just X.
        
        # Check feasibility of baseline
        is_feasible = np.all(res_v1.G <= 1e-6)
        
        data_records.append({
            "X": x.tolist(),
            "y_base": [eff_v1, loss_v1],
            "y_true": [eff_true, loss_true],
            "y_resid": [delta_eff, delta_loss],
            "feasible_v1": bool(is_feasible)
        })
        
    print(f"Done in {time.time() - t0:.2f}s")
    
    # 3. Save
    # Save as compressed numpy for efficiency
    keys = list(data_records[0].keys())
    arrays = {}
    for k in keys:
        arrays[k] = np.array([d[k] for d in data_records])
        
    np.savez_compressed(
        outpath / "training_data.npz",
        **arrays
    )
    
    print(f"Saved to {outpath}/training_data.npz")
    print("Stats:")
    print(f"  Delta Efficiency Mean: {np.mean(arrays['y_resid'][:,0]):.4f}")
    print(f"  Delta Loss Mean:       {np.mean(arrays['y_resid'][:,1]):.4f}")


if __name__ == "__main__":
    main()
