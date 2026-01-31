"""Pareto refinement CLI.

Loads Pareto archive from run_pareto and refines top-K candidates
using CasADi/scipy gradient-based optimization.

Usage:
    python -m larrak2.cli.refine_pareto --input . --top-k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..core.constants import MODEL_VERSION_GEAR_V1, MODEL_VERSION_THERMO_V1
from ..core.constraints import get_constraint_names, get_constraint_scales
from ..core.encoding import ENCODING_VERSION


def main(argv: list[str] | None = None) -> int:
    """Refine Pareto front candidates.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Refine Pareto front candidates")
    parser.add_argument("--input", type=str, default=".", help="Input directory with pareto_*.npy")
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory (default: input)"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to refine")
    parser.add_argument(
        "--mode",
        type=str,
        default="weighted_sum",
        choices=["weighted_sum", "eps_constraint"],
        help="Refinement mode",
    )
    parser.add_argument("--rpm", type=float, default=3000.0, help="Engine speed (rpm)")
    parser.add_argument("--torque", type=float, default=200.0, help="Torque demand (Nm)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args(argv)

    from ..adapters.casadi_refine import RefinementMode, refine_candidate
    from ..core.types import EvalContext

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir

    # Load Pareto archive
    X = np.load(input_dir / "pareto_X.npy")
    F = np.load(input_dir / "pareto_F.npy")

    if args.verbose:
        print(f"Loaded {X.shape[0]} Pareto solutions")

    # Select top-K by first objective
    indices = np.argsort(F[:, 0])[: args.top_k]

    ctx = EvalContext(rpm=args.rpm, torque=args.torque, fidelity=0, seed=42)
    mode = RefinementMode(args.mode)

    refined_X = []
    refined_F = []
    refined_G = []
    results_diag = []

    for i, idx in enumerate(indices):
        x0 = X[idx]

        if args.verbose:
            print(f"Refining candidate {i + 1}/{len(indices)} (original F={F[idx]})")

        result = refine_candidate(x0, ctx, mode=mode)

        refined_X.append(result.x_refined)
        refined_F.append(result.F_refined)
        refined_G.append(result.G_refined)
        results_diag.append(
            {
                "original_idx": int(idx),
                "original_F": F[idx].tolist(),
                "refined_F": result.F_refined.tolist(),
                "success": result.success,
                "message": result.message,
            }
        )

        if args.verbose:
            print(f"  -> refined F={result.F_refined}, success={result.success}")

    # Save results
    np.save(output_dir / "refined_X.npy", np.array(refined_X))
    np.save(output_dir / "refined_F.npy", np.array(refined_F))
    np.save(output_dir / "refined_G.npy", np.array(refined_G))

    with open(output_dir / "refinement_summary.json", "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "rpm": args.rpm,
                "torque": args.torque,
                "fidelity": ctx.fidelity,
                "seed": ctx.seed,
                "encoding_version": ENCODING_VERSION,
                "model_versions": {
                    "thermo_v1": MODEL_VERSION_THERMO_V1,
                    "gear_v1": MODEL_VERSION_GEAR_V1,
                },
                "constraint_names": get_constraint_names(ctx.fidelity),
                "constraint_scales": get_constraint_scales(),
                "n_refined": len(refined_X),
                "results": results_diag,
            },
            f,
            indent=2,
        )

    if args.verbose:
        print(f"\nRefined {len(refined_X)} candidates")
        print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
