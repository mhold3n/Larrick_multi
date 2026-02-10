"""Generate Element-wise training data for Gear Logic Surrogate.

Focuses on creating a dataset mapping Design Parameters -> Loss Components.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from larrak2.core.encoding import GearParams
from larrak2.core.types import EvalContext
from larrak2.gear.litvin_core import eval_gear
from larrak2.gear.pitchcurve import fourier_pitch_curve


def generate_gear_doe(
    n_samples: int,
    output_path: str,
    fixed_rpm: float | None = None,
    fixed_torque: float | None = None,
):
    """Generate DOE samples for Gear Physics."""
    print(f"Generating {n_samples} samples...")
    if fixed_rpm:
        print(f"  Fixed RPM: {fixed_rpm}")
    if fixed_torque:
        print(f"  Fixed Torque: {fixed_torque}")

    data = []

    # Fixed Context for now (or vary it?)
    # Loss NN usually needs Speed/Torque as inputs.
    # So we should vary RPM and Torque too.

    rng = np.random.default_rng(42)

    # Paired Data Loop: Single (Circular) vs Dual (Non-Circular)
    # Target: We want to see the cost of meeting a variable ratio requirement.

    rng = np.random.default_rng(42)

    for i in tqdm(range(n_samples)):
        # 1. Operating Point
        rpm = fixed_rpm if fixed_rpm is not None else rng.uniform(500, 6000)
        torque = fixed_torque if fixed_torque is not None else rng.uniform(10, 200)

        # 2. Loss Coefficients (Randomized around defaults)
        # Mu: 0.03 - 0.08
        mu = rng.uniform(0.03, 0.08)
        # Bearing: C0..C3. Randomize base friction magnitude.
        c_bear = (
            rng.uniform(0.005, 0.02),  # C0
            1e-4,
            1e-6,
            0.002,
        )
        c_bear_t = (float(c_bear[0]), float(c_bear[1]), float(c_bear[2]), float(c_bear[3]))
        loss_coeffs = {"mesh": (mu,), "bearing": c_bear_t}

        # 3. Geometry / Requirement
        base_radius = rng.uniform(30.0, 50.0)

        # Generate a "Requirement" (Ratio Profile)
        # i_req = 1.0 + Amplitude * sin(...)
        amp = rng.uniform(0.0, 0.2)  # Up to 20% variation
        # Define the Non-Circular coefficients that MATCH this requirements
        # r_planet ~ base_radius * (1 + sum(Cn cos...))
        # Use C1 (eccentricity-like) for simplicity in pilot
        c1 = amp * base_radius
        coeffs_dual = (0.0, c1, 0.0, 0.0, 0.0)

        # Create "Dual" Params (The Optimized Solution)
        params_dual = GearParams(base_radius=base_radius, pitch_coeffs=coeffs_dual)

        # Create "Single" Params (The Baseline Circular)
        params_single = GearParams(base_radius=base_radius, pitch_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0))

        ctx = EvalContext(rpm=rpm, torque=torque, fidelity=1, seed=i, loss_coeffs=loss_coeffs)

        # Generate i_req from params_dual so Dual has ~0 error
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        r_ring = 80.0
        r_p_dual = fourier_pitch_curve(theta, base_radius, coeffs_dual)
        i_req = r_ring / np.maximum(r_p_dual, 1e-6)

        try:
            # Eval Dual
            res_dual = eval_gear(params_dual, i_req, ctx)
            l_dual = res_dual.ledger

            # Eval Single
            res_single = eval_gear(params_single, i_req, ctx)
            l_single = res_single.ledger

            if l_dual is None or l_single is None:
                continue

            # Check closure before accepting (silent for speed)
            if not l_dual.is_closed or not l_single.is_closed:
                continue

            row = {
                "rpm": rpm,
                "torque": torque,
                "base_radius": base_radius,
                "req_amp": amp,
                "mu": mu,
                "c_bear_0": c_bear_t[0],
                # Single Results
                "single_W_mesh": l_single.W_loss_mesh,
                "single_W_bearing": l_single.W_loss_bearing,
                "single_ratio_error": res_single.ratio_error_max,
                # Dual Results
                "dual_W_mesh": l_dual.W_loss_mesh,
                "dual_W_bearing": l_dual.W_loss_bearing,
                "dual_ratio_error": res_dual.ratio_error_max,
                # Comparison
                "delta_W_mesh": l_single.W_loss_mesh - l_dual.W_loss_mesh,
                "closure_error_dual": l_dual.compute_closure_error(),
            }
            data.append(row)

        except Exception:
            continue

    df = pd.DataFrame(data)
    print(f"Generated {len(df)} valid samples.")

    # Save
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=str, default="data/gear_doe_v1.parquet")
    parser.add_argument("--rpm", type=float, default=None, help="Fix RPM (e.g. 3000)")
    parser.add_argument("--torque", type=float, default=None, help="Fix Torque (e.g. 100)")
    args = parser.parse_args()

    generate_gear_doe(args.n, args.out, fixed_rpm=args.rpm, fixed_torque=args.torque)
