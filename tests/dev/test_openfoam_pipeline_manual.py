#!/usr/bin/env python3
"""Validation gates for the real OpenFOAM DOE + surrogate pipeline.

This script is intentionally explicit and conservative. It runs:
1) A small DOE sanity batch (default 10) to confirm the OpenFOAM case produces
   the required log markers and that outputs are finite.
2) A surrogate training sanity run (default uses the DOE JSONL output).
3) A single `fidelity=2` evaluation smoke (requires the trained artifact path).

Notes:
- This is not run in CI; it is meant for local bring-up with Docker OpenFOAM.
- Expect to tune the OpenFOAM case/template before these gates pass reliably.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Run OpenFOAM pipeline validation gates")
    p.add_argument("--n-sanity", type=int, default=10)
    p.add_argument("--doe-jsonl", type=str, default="data/openfoam_doe/results.jsonl")
    p.add_argument("--doe-outdir", type=str, default="data/openfoam_doe")
    p.add_argument("--runs-root", type=str, default="runs/openfoam_doe")
    p.add_argument(
        "--template", type=str, default="openfoam_templates/opposed_piston_rotary_valve_case"
    )
    p.add_argument("--solver", type=str, default="rhoPimpleFoam")
    p.add_argument("--train-epochs", type=int, default=200)
    p.add_argument("--artifact-outdir", type=str, default="models/openfoam_nn")
    p.add_argument("--artifact-name", type=str, default="openfoam_breathing.pt")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]

    doe_jsonl = root / args.doe_jsonl
    doe_checkpoint = root / args.doe_outdir / "checkpoint.json"

    # Always start the sanity DOE from a clean slate.
    doe_jsonl.unlink(missing_ok=True)
    doe_checkpoint.unlink(missing_ok=True)

    # 1) DOE sanity
    subprocess.run(
        [
            "python",
            "scripts/run_openfoam_doe.py",
            "--template",
            args.template,
            "--solver",
            args.solver,
            "--n",
            str(args.n_sanity),
            "--jsonl",
            args.doe_jsonl,
            "--checkpoint",
            str(Path(args.doe_outdir) / "checkpoint.json"),
            "--outdir",
            args.doe_outdir,
            "--runs-root",
            args.runs_root,
            "--checkpoint-every",
            "1",
            "--snappy",
            "--endTime",
            "0.0001",
            "--deltaT",
            "0.0001",
        ],
        cwd=root,
        check=True,
    )

    # 2) Surrogate sanity training
    subprocess.run(
        [
            "python",
            "scripts/train_openfoam_surrogate.py",
            "--data",
            args.doe_jsonl,
            "--outdir",
            args.artifact_outdir,
            "--name",
            args.artifact_name,
            "--epochs",
            str(args.train_epochs),
        ],
        cwd=root,
        check=True,
    )

    artifact_path = str(Path(args.artifact_outdir) / args.artifact_name)
    os.environ["LARRAK2_OPENFOAM_NN_PATH"] = artifact_path

    # 3) fidelity=2 evaluation smoke
    subprocess.run(
        [
            "python",
            "-c",
            "import numpy as np; "
            "from larrak2.core.encoding import mid_bounds_candidate; "
            "from larrak2.core.types import EvalContext; "
            "from larrak2.core.evaluator import evaluate_candidate; "
            "x=mid_bounds_candidate(); "
            "ctx=EvalContext(rpm=3000.0, torque=200.0, fidelity=2, seed=1); "
            "r=evaluate_candidate(x, ctx); "
            "print('F', r.F); print('G_max', float(np.max(r.G)));",
        ],
        cwd=root,
        check=True,
    )

    print("\nValidation gates completed.")
    print(f"Trained artifact: {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
