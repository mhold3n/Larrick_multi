"""Larrak2 Unified Run CLI.

Usage:
    python scripts/run.py pareto-grid     --pop 64 --gen 50
    python scripts/run.py pareto-staged   --pop 100 --gen 50 --promote 20
    python scripts/run.py active-learning --pop 64 --gen 30 --n_truth 10
    python scripts/run.py openfoam-doe    --template ... --n 500
"""

from __future__ import annotations

import argparse
import sys

from larrak2.cli.run_workflows import (
    run_active_learning_workflow,
    run_openfoam_doe_workflow,
    run_pareto_grid_workflow,
    run_pareto_staged_workflow,
)
from larrak2.analysis.workflows import diagnostic_workflow, sensitivity_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Larrak2 Unified Run CLI")
    subparsers = parser.add_subparsers(dest="run_type", required=True, help="Run type")

    # --- Pareto Grid ---
    pg = subparsers.add_parser("pareto-grid", help="Pareto over (rpm, torque) grid")
    pg.add_argument("--outdir-root", type=str, default="results_grid")
    pg.add_argument("--pop", type=int, default=64)
    pg.add_argument("--gen", type=int, default=50)
    pg.add_argument("--fidelity", type=int, default=2, choices=[0, 1, 2])
    pg.add_argument("--seed", type=int, default=42)
    pg.add_argument("--verbose", action="store_true")
    pg.add_argument("--bore-mm", type=float, default=80.0)
    pg.add_argument("--stroke-mm", type=float, default=90.0)
    pg.add_argument("--intake-port-area-m2", type=float, default=4.0e-4)
    pg.add_argument("--exhaust-port-area-m2", type=float, default=4.0e-4)
    pg.add_argument("--p-manifold-pa", type=float, default=101325.0)
    pg.add_argument("--p-back-pa", type=float, default=101325.0)
    pg.add_argument("--overlap-deg", type=float, default=0.0)
    pg.add_argument("--intake-open-deg", type=float, default=0.0)
    pg.add_argument("--intake-close-deg", type=float, default=0.0)
    pg.add_argument("--exhaust-open-deg", type=float, default=0.0)
    pg.add_argument("--exhaust-close-deg", type=float, default=0.0)
    pg.add_argument("--rpm-list", type=str, default="")
    pg.add_argument("--torque-list", type=str, default="")
    pg.add_argument("--rpm-min", type=float, default=1000.0)
    pg.add_argument("--rpm-max", type=float, default=7000.0)
    pg.add_argument("--rpm-n", type=int, default=6)
    pg.add_argument("--torque-min", type=float, default=50.0)
    pg.add_argument("--torque-max", type=float, default=400.0)
    pg.add_argument("--torque-n", type=int, default=6)

    # --- Pareto Staged ---
    ps = subparsers.add_parser("pareto-staged", help="Multi-fidelity staged Pareto")
    ps.add_argument("--pop", type=int, default=100)
    ps.add_argument("--gen", type=int, default=50)
    ps.add_argument("--promote", type=int, default=20)
    ps.add_argument("--rpm", type=float, default=3000.0)
    ps.add_argument("--torque", type=float, default=200.0)
    ps.add_argument("--seed", type=int, default=1)
    ps.add_argument("--outdir", type=str, default="results_staged")

    # --- Active Learning ---
    al = subparsers.add_parser("active-learning", help="Staged opt → uncertainty → truth sims")
    al.add_argument("--outdir", type=str, default="out/active_learning")
    al.add_argument("--pop", type=int, default=64)
    al.add_argument("--gen", type=int, default=30)
    al.add_argument("--promote", type=int, default=20)
    al.add_argument("--n_truth", type=int, default=10)
    al.add_argument("--dry_run", action="store_true")
    al.add_argument("--rpm", type=float, default=3000.0)
    al.add_argument("--torque", type=float, default=200.0)

    # --- OpenFOAM DOE ---
    od = subparsers.add_parser("openfoam-doe", help="Checkpointed OpenFOAM DOE")
    od.add_argument("--template", type=str, default="openfoam_templates/opposed_piston_rotary_valve_sliding_case")
    od.add_argument("--outdir", type=str, default="data/openfoam_doe")
    od.add_argument("--runs-root", type=str, default="runs/openfoam_doe")
    od.add_argument("--jsonl", type=str, default="data/openfoam_doe/results.jsonl")
    od.add_argument("--checkpoint", type=str, default="data/openfoam_doe/checkpoint.json")
    od.add_argument("--n", type=int, default=500)
    od.add_argument("--seed", type=int, default=42)
    od.add_argument("--checkpoint-every", type=int, default=10)
    od.add_argument("--solver", type=str, default="rhoPimpleFoam")
    od.add_argument("--docker-timeout-s", type=int, default=1800)
    od.add_argument("--snappy", action="store_true")
    od.add_argument("--rpm-min", type=float, default=1000.0)
    od.add_argument("--rpm-max", type=float, default=7000.0)
    od.add_argument("--torque-min", type=float, default=50.0)
    od.add_argument("--torque-max", type=float, default=400.0)
    od.add_argument("--lambda-min", type=float, default=0.6)
    od.add_argument("--lambda-max", type=float, default=1.6)
    od.add_argument("--bore-mm", type=float, default=80.0)
    od.add_argument("--stroke-mm", type=float, default=90.0)
    od.add_argument("--intake-port-area-min", type=float, default=2e-4)
    od.add_argument("--intake-port-area-max", type=float, default=8e-4)
    od.add_argument("--exhaust-port-area-min", type=float, default=2e-4)
    od.add_argument("--exhaust-port-area-max", type=float, default=8e-4)
    od.add_argument("--p-manifold-min", type=float, default=30_000.0)
    od.add_argument("--p-manifold-max", type=float, default=250_000.0)
    od.add_argument("--p-back-min", type=float, default=80_000.0)
    od.add_argument("--p-back-max", type=float, default=200_000.0)
    od.add_argument("--overlap-min", type=float, default=0.0)
    od.add_argument("--overlap-max", type=float, default=80.0)
    od.add_argument("--intake-open-min", type=float, default=-60.0)
    od.add_argument("--intake-open-max", type=float, default=20.0)
    od.add_argument("--intake-close-min", type=float, default=20.0)
    od.add_argument("--intake-close-max", type=float, default=120.0)
    od.add_argument("--exhaust-open-min", type=float, default=-120.0)
    od.add_argument("--exhaust-open-max", type=float, default=-20.0)
    od.add_argument("--exhaust-close-min", type=float, default=-20.0)
    od.add_argument("--exhaust-close-max", type=float, default=60.0)
    od.add_argument("--endTime", type=float, default=0.01)
    od.add_argument("--deltaT", type=float, default=1e-4)
    od.add_argument("--writeInterval", type=int, default=100)
    od.add_argument("--metricWriteInterval", type=int, default=1)

    # --- Sensitivity ---
    sens = subparsers.add_parser("sensitivity", help="Sensitivity & Constraint Analysis")
    # No args for now, uses hardcoded mid-bounds or diagnostics

    # --- Diagnostic ---
    diag = subparsers.add_parser("diagnostic", help="Diagnostic Pareto Run")
    diag.add_argument("--pop", type=int, default=64)
    diag.add_argument("--gen", type=int, default=50)
    diag.add_argument("--rpm", type=float, default=3000.0)
    diag.add_argument("--torque", type=float, default=200.0)
    diag.add_argument("--fidelity", type=int, default=1)
    diag.add_argument("--seed", type=int, default=123)
    diag.add_argument("--outdir", type=str, default="diagnostic_results")

    args = parser.parse_args()

    dispatch = {
        "pareto-grid": run_pareto_grid_workflow,
        "pareto-staged": run_pareto_staged_workflow,
        "active-learning": run_active_learning_workflow,
        "active-learning": run_active_learning_workflow,
        "openfoam-doe": run_openfoam_doe_workflow,
        "sensitivity": sensitivity_workflow,
        "diagnostic": diagnostic_workflow,
    }

    fn = dispatch.get(args.run_type)
    if fn is None:
        parser.print_help()
        return 1

    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
