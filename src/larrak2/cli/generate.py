"""Larrak2 Unified Data Generation CLI.

Usage:
    python scripts/generate.py gear     --n 100 --out data/gear_doe.parquet
    python scripts/generate.py residual --n 200 --out data/residual/training_data.npz
    python scripts/generate.py scavenge --n 100 --out data/scavenge
    python scripts/generate.py openfoam --template-dir ... --n 50 --out data/openfoam.json
"""

from __future__ import annotations

import argparse
import sys

from larrak2.geometry.generate_stl import generate_stl_workflow
from larrak2.pipelines.data.workflows import (
    generate_gear_workflow,
    generate_openfoam_workflow,
    generate_residual_workflow,
    generate_scavenge_workflow,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Larrak2 Unified Data Generation CLI")
    subparsers = parser.add_subparsers(dest="data_type", required=True, help="Data type")

    # --- Gear ---
    p_gear = subparsers.add_parser("gear", help="Gear DOE (Parquet)")
    p_gear.add_argument("--n", type=int, default=100)
    p_gear.add_argument("--out", type=str, default="data/gear_doe_v1.parquet")
    p_gear.add_argument("--rpm", type=float, default=None, help="Fix RPM")
    p_gear.add_argument("--torque", type=float, default=None, help="Fix Torque")

    # --- Residual ---
    p_resid = subparsers.add_parser("residual", help="Residual surrogates (NPZ)")
    p_resid.add_argument("--n", type=int, default=200)
    p_resid.add_argument("--out", type=str, default="data/surrogate_v1/training_data.npz")
    p_resid.add_argument("--seed", type=int, default=42)

    # --- Scavenge ---
    p_scav = subparsers.add_parser("scavenge", help="Scavenge efficiency (NPZ)")
    p_scav.add_argument("--n", type=int, default=100)
    p_scav.add_argument("--out", type=str, default="data/surrogate_v1/scavenge")
    p_scav.add_argument("--use-sim", action="store_true", help="Run actual OpenFOAM sim")
    p_scav.add_argument("--template", type=str, default="templates/scavenge_case")

    # --- OpenFOAM ---
    p_of = subparsers.add_parser("openfoam", help="OpenFOAM breathing dataset (JSON)")
    p_of.add_argument("--template-dir", required=True)
    p_of.add_argument("--run-root", default="runs/openfoam_dataset")
    p_of.add_argument("--out", default="data/openfoam_nn/dataset.json")
    p_of.add_argument("--solver-cmd", default="pisoFoam")
    p_of.add_argument("--n", type=int, default=200)
    p_of.add_argument("--seed", type=int, default=42)
    # Feature ranges
    p_of.add_argument("--rpm-min", type=float, default=1000.0)
    p_of.add_argument("--rpm-max", type=float, default=7000.0)
    p_of.add_argument("--torque-min", type=float, default=20.0)
    p_of.add_argument("--torque-max", type=float, default=400.0)
    p_of.add_argument("--lambda-min", type=float, default=0.6)
    p_of.add_argument("--lambda-max", type=float, default=1.6)
    p_of.add_argument("--compression-min", type=float, default=30.0)
    p_of.add_argument("--compression-max", type=float, default=90.0)
    p_of.add_argument("--expansion-min", type=float, default=60.0)
    p_of.add_argument("--expansion-max", type=float, default=120.0)
    p_of.add_argument("--hr-center-min", type=float, default=0.0)
    p_of.add_argument("--hr-center-max", type=float, default=30.0)
    p_of.add_argument("--hr-width-min", type=float, default=10.0)
    p_of.add_argument("--hr-width-max", type=float, default=60.0)
    # Geometry / BC
    p_of.add_argument("--bore-mm", type=float, default=80.0)
    p_of.add_argument("--stroke-mm", type=float, default=90.0)
    p_of.add_argument("--intake-port-area-min", type=float, default=2.0e-4)
    p_of.add_argument("--intake-port-area-max", type=float, default=8.0e-4)
    p_of.add_argument("--exhaust-port-area-min", type=float, default=2.0e-4)
    p_of.add_argument("--exhaust-port-area-max", type=float, default=8.0e-4)
    p_of.add_argument("--p-manifold-min", type=float, default=30_000.0)
    p_of.add_argument("--p-manifold-max", type=float, default=250_000.0)
    p_of.add_argument("--p-back-min", type=float, default=80_000.0)
    p_of.add_argument("--p-back-max", type=float, default=200_000.0)
    p_of.add_argument("--overlap-min", type=float, default=0.0)
    p_of.add_argument("--overlap-max", type=float, default=80.0)
    p_of.add_argument("--intake-open-min", type=float, default=-60.0)
    p_of.add_argument("--intake-open-max", type=float, default=20.0)
    p_of.add_argument("--intake-close-min", type=float, default=20.0)
    p_of.add_argument("--intake-close-max", type=float, default=120.0)
    p_of.add_argument("--exhaust-open-min", type=float, default=-120.0)
    p_of.add_argument("--exhaust-open-max", type=float, default=-20.0)
    p_of.add_argument("--exhaust-close-min", type=float, default=-20.0)
    p_of.add_argument("--exhaust-close-max", type=float, default=60.0)
    # Solver
    p_of.add_argument("--solver-name", type=str, default="rhoPimpleFoam")
    p_of.add_argument("--endTime", type=float, default=0.01)
    p_of.add_argument("--deltaT", type=float, default=1e-4)
    p_of.add_argument("--writeInterval", type=int, default=100)
    p_of.add_argument("--metricWriteInterval", type=int, default=100)

    # --- STL ---
    p_stl = subparsers.add_parser("stl", help="Generate OpenFOAM Geometry STLs")
    p_stl.add_argument("--outdir", type=str, required=True)
    p_stl.add_argument("--bore-mm", type=float, default=80.0)
    p_stl.add_argument("--stroke-mm", type=float, default=90.0)
    p_stl.add_argument("--intake-port-area-m2", type=float, default=5e-4)
    p_stl.add_argument("--exhaust-port-area-m2", type=float, default=5e-4)

    args = parser.parse_args()

    dispatch = {
        "gear": generate_gear_workflow,
        "residual": generate_residual_workflow,
        "scavenge": generate_scavenge_workflow,
        "openfoam": generate_openfoam_workflow,
        "stl": generate_stl_workflow,
    }

    fn = dispatch.get(args.data_type)
    if fn is None:
        parser.print_help()
        return 1

    fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
