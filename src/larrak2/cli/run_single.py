"""Single candidate evaluation CLI.

Usage:
    python -m larrak2.cli.run_single --rpm 3000 --torque 200

Outputs JSON with F, G, and diagnostics to stdout.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv: list[str] | None = None) -> int:
    """Run single candidate evaluation.

    Args:
        argv: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(description="Evaluate a single candidate solution")
    parser.add_argument("--rpm", type=float, default=3000.0, help="Engine speed (rpm)")
    parser.add_argument("--torque", type=float, default=200.0, help="Torque demand (Nm)")
    parser.add_argument("--fidelity", type=int, default=0, choices=[0, 1, 2], help="Model fidelity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--x", type=str, default=None, help="Candidate vector as JSON array")
    parser.add_argument("--random", action="store_true", help="Use random candidate")

    args = parser.parse_args(argv)

    from ..core.encoding import mid_bounds_candidate, random_candidate
    from ..core.evaluator import evaluate_candidate
    from ..core.types import EvalContext

    # Get candidate
    if args.x is not None:
        x = np.array(json.loads(args.x), dtype=np.float64)
    elif args.random:
        rng = np.random.default_rng(args.seed)
        x = random_candidate(rng)
    else:
        x = mid_bounds_candidate()

    ctx = EvalContext(
        rpm=args.rpm,
        torque=args.torque,
        fidelity=args.fidelity,
        seed=args.seed,
    )

    # Evaluate
    result = evaluate_candidate(x, ctx)

    # Format output
    output = {
        "x": x.tolist(),
        "F": result.F.tolist(),
        "G": result.G.tolist(),
        "is_feasible": result.is_feasible,
        "max_violation": result.max_violation,
        "metrics": result.diag.get("metrics", {}),
        "timings": result.diag.get("timings", {}),
    }

    print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
