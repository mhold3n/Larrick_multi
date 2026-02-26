"""Larrak2 Unified Run CLI.

Usage:
    python scripts/run.py pareto-grid     --pop 64 --gen 50
    python scripts/run.py pareto-staged   --pop 100 --gen 50 --promote 20
    python scripts/run.py active-learning --pop 64 --gen 30 --n_truth 10
    python scripts/run.py train-surrogates --single-condition
    python scripts/run.py openfoam-doe    --template ... --n 500
    python scripts/run.py dress-rehearsal --pop 16 --gen 5
"""

from __future__ import annotations

import argparse
import sys

from larrak2.analysis.workflows import diagnostic_workflow, sensitivity_workflow
from larrak2.cli.run_workflows import (
    run_active_learning_workflow,
    run_dress_rehearsal_workflow,
    run_openfoam_doe_workflow,
    run_orchestrate_workflow,
    run_pareto_grid_workflow,
    run_pareto_staged_workflow,
    run_train_surrogates_workflow,
)


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
    pg.add_argument("--openfoam-model-path", type=str, default="")
    pg.add_argument(
        "--calculix-stress-mode",
        type=str,
        default="nn",
        choices=["nn", "analytical"],
    )
    pg.add_argument("--calculix-model-path", type=str, default="")
    pg.add_argument(
        "--gear-loss-mode",
        type=str,
        default="physics",
        choices=["physics", "nn"],
    )
    pg.add_argument("--gear-loss-model-dir", type=str, default="")
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
    od.add_argument(
        "--template",
        type=str,
        default="openfoam_templates/opposed_piston_rotary_valve_sliding_case",
    )
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

    # --- Train Surrogates ---
    ts = subparsers.add_parser(
        "train-surrogates",
        help="Standalone OpenFOAM/CalculiX NN surrogate training job",
    )
    ts.add_argument("--outdir", type=str, default="outputs/surrogate_training")
    ts.add_argument("--openfoam-data", type=str, default="")
    ts.add_argument(
        "--openfoam-template",
        type=str,
        default="",
        help="OpenFOAM template directory used to generate training data when --openfoam-data is not provided",
    )
    ts.add_argument("--openfoam-runs-per-condition", type=int, default=4)
    ts.add_argument("--openfoam-lambda-min", type=float, default=0.6)
    ts.add_argument("--openfoam-lambda-max", type=float, default=1.6)
    ts.add_argument("--openfoam-solver", type=str, default="rhoPimpleFoam")
    ts.add_argument("--openfoam-docker-timeout-s", type=int, default=1800)
    ts.add_argument("--openfoam-outdir", type=str, default="models/openfoam_nn")
    ts.add_argument("--openfoam-name", type=str, default="openfoam_breathing.pt")
    ts.add_argument("--openfoam-epochs", type=int, default=120)
    ts.add_argument("--openfoam-lr", type=float, default=1e-3)
    ts.add_argument("--openfoam-hidden", type=str, default="64,64")
    ts.add_argument("--openfoam-weight-decay", type=float, default=0.0)
    ts.add_argument("--calculix-data", type=str, default="")
    ts.add_argument(
        "--calculix-template",
        type=str,
        default="",
        help="CalculiX .inp template used to generate training data when --calculix-data is not provided",
    )
    ts.add_argument("--calculix-runs-per-condition", type=int, default=4)
    ts.add_argument("--calculix-solver", type=str, default="ccx")
    ts.add_argument("--calculix-outdir", type=str, default="models/calculix_nn")
    ts.add_argument("--calculix-name", type=str, default="calculix_stress.pt")
    ts.add_argument("--calculix-epochs", type=int, default=120)
    ts.add_argument("--calculix-lr", type=float, default=1e-3)
    ts.add_argument("--calculix-hidden", type=str, default="64,64")
    ts.add_argument("--calculix-weight-decay", type=float, default=0.0)
    ts.add_argument("--rpm", type=float, default=3000.0)
    ts.add_argument("--torque", type=float, default=200.0)
    ts.add_argument(
        "--condition-sweep",
        dest="condition_sweep",
        action="store_true",
        help="Use operating-condition sweep for DOE-backed datasets",
    )
    ts.add_argument(
        "--single-condition",
        dest="condition_sweep",
        action="store_false",
        help="Use only --rpm/--torque for dataset generation",
    )
    ts.set_defaults(condition_sweep=True)
    ts.add_argument("--rpm-min", type=float, default=1200.0)
    ts.add_argument("--rpm-max", type=float, default=2800.0)
    ts.add_argument("--rpm-step", type=float, default=800.0)
    ts.add_argument("--torque-min", type=float, default=40.0)
    ts.add_argument("--torque-max", type=float, default=180.0)
    ts.add_argument("--torque-step", type=float, default=40.0)
    ts.add_argument("--seed", type=int, default=42)
    ts.add_argument("--verbose", action="store_true")
    ts.add_argument("--bore-mm", type=float, default=80.0)
    ts.add_argument("--stroke-mm", type=float, default=90.0)
    ts.add_argument("--intake-port-area-m2", type=float, default=4.0e-4)
    ts.add_argument("--exhaust-port-area-m2", type=float, default=4.0e-4)
    ts.add_argument("--p-manifold-pa", type=float, default=101325.0)
    ts.add_argument("--p-back-pa", type=float, default=101325.0)
    ts.add_argument("--overlap-deg", type=float, default=0.0)
    ts.add_argument("--intake-open-deg", type=float, default=0.0)
    ts.add_argument("--intake-close-deg", type=float, default=0.0)
    ts.add_argument("--exhaust-open-deg", type=float, default=0.0)
    ts.add_argument("--exhaust-close-deg", type=float, default=0.0)

    # --- Dress Rehearsal ---
    dr = subparsers.add_parser(
        "dress-rehearsal",
        help="Verify surrogate readiness, run optimization, and finish CEM validation gates",
    )
    dr.add_argument("--outdir", type=str, default="outputs/dress_rehearsal")
    dr.add_argument("--pop", type=int, default=16)
    dr.add_argument("--gen", type=int, default=5)
    dr.add_argument("--rpm", type=float, default=3000.0)
    dr.add_argument("--torque", type=float, default=200.0)
    dr.add_argument(
        "--condition-sweep",
        dest="condition_sweep",
        action="store_true",
        help="Run optimization over a coarse operating-condition grid",
    )
    dr.add_argument(
        "--single-condition",
        dest="condition_sweep",
        action="store_false",
        help="Run only the single --rpm/--torque operating condition",
    )
    dr.set_defaults(condition_sweep=True)
    dr.add_argument("--rpm-min", type=float, default=1200.0)
    dr.add_argument("--rpm-max", type=float, default=2800.0)
    dr.add_argument("--rpm-step", type=float, default=800.0)
    dr.add_argument("--torque-min", type=float, default=40.0)
    dr.add_argument("--torque-max", type=float, default=180.0)
    dr.add_argument("--torque-step", type=float, default=40.0)
    dr.add_argument("--fidelity", type=int, default=2, choices=[0, 1, 2])
    dr.add_argument(
        "--constraint-phase",
        type=str,
        default="explore",
        choices=["explore", "downselect"],
    )
    dr.add_argument(
        "--tolerance-constraint-mode",
        type=str,
        default="capability_floor",
        choices=["capability_floor", "stack_budget_max"],
    )
    dr.add_argument("--tolerance-threshold-mm", type=float, default=0.24)
    dr.add_argument(
        "--promotion-margin-min",
        type=float,
        default=-0.25,
        help="Minimum hard-constraint normalized margin for promotion",
    )
    dr.add_argument(
        "--promotion-pool-mult",
        type=int,
        default=4,
        help="Promotion shortlist multiplier before geometry-diversity selection",
    )
    dr.add_argument("--seed", type=int, default=42)
    dr.add_argument("--min-pareto", type=int, default=1)
    dr.add_argument("--cem-top", type=int, default=10)
    dr.add_argument("--cem-min-feasible", type=int, default=1)
    dr.add_argument(
        "--run-unit-tests",
        dest="run_unit_tests",
        action="store_true",
        help="Run unit tests as a required dress-rehearsal stage",
    )
    dr.add_argument(
        "--skip-unit-tests",
        dest="run_unit_tests",
        action="store_false",
        help="Skip unit-test stage (not recommended)",
    )
    dr.set_defaults(run_unit_tests=True)
    dr.add_argument(
        "--pytest-target",
        type=str,
        default="tests",
        help="Pytest target path/pattern used by dress-rehearsal",
    )
    dr.add_argument(
        "--pytest-args",
        type=str,
        default="-q",
        help="Additional pytest CLI args (quoted string)",
    )
    dr.add_argument(
        "--allow-gate-failure",
        dest="fail_on_gate",
        action="store_false",
        help="Do not fail the command if the CEM gate is not met",
    )
    dr.set_defaults(fail_on_gate=True)
    dr.add_argument("--verbose", action="store_true")
    dr.add_argument("--bore-mm", type=float, default=80.0)
    dr.add_argument("--stroke-mm", type=float, default=90.0)
    dr.add_argument("--intake-port-area-m2", type=float, default=4.0e-4)
    dr.add_argument("--exhaust-port-area-m2", type=float, default=4.0e-4)
    dr.add_argument("--p-manifold-pa", type=float, default=101325.0)
    dr.add_argument("--p-back-pa", type=float, default=101325.0)
    dr.add_argument("--overlap-deg", type=float, default=0.0)
    dr.add_argument("--intake-open-deg", type=float, default=0.0)
    dr.add_argument("--intake-close-deg", type=float, default=0.0)
    dr.add_argument("--exhaust-open-deg", type=float, default=0.0)
    dr.add_argument("--exhaust-close-deg", type=float, default=0.0)
    dr.add_argument(
        "--openfoam-model-path",
        type=str,
        default="",
        help="Override OpenFOAM NN artifact path used during optimization stage",
    )
    dr.add_argument(
        "--calculix-stress-mode",
        type=str,
        default="nn",
        choices=["nn", "analytical"],
        help="CalculiX stress evaluation mode for optimization stage",
    )
    dr.add_argument(
        "--calculix-model-path",
        type=str,
        default="",
        help="Override CalculiX NN artifact path used during optimization stage",
    )
    dr.add_argument(
        "--gear-loss-mode",
        type=str,
        default="physics",
        choices=["physics", "nn"],
        help="Gear-loss evaluation mode for optimization stage",
    )
    dr.add_argument(
        "--gear-loss-model-dir",
        type=str,
        default="",
        help="Gear-loss NN model directory when --gear-loss-mode=nn",
    )

    # --- Backend Orchestration ---
    orch = subparsers.add_parser(
        "orchestrate",
        help="Backend orchestration loop (no GUI)",
    )
    orch.add_argument("--outdir", type=str, default="outputs/orchestration")
    orch.add_argument("--rpm", type=float, default=3000.0)
    orch.add_argument("--torque", type=float, default=200.0)
    orch.add_argument("--seed", type=int, default=42)
    orch.add_argument("--sim-budget", type=int, default=32)
    orch.add_argument("--batch-size", type=int, default=16)
    orch.add_argument("--max-iterations", type=int, default=8)
    orch.add_argument(
        "--truth-dispatch-mode",
        type=str,
        choices=["off", "manual"],
        default="off",
    )
    orch.add_argument(
        "--truth-plan",
        type=str,
        default="",
        help="Path to JSON truth plan (required for --truth-dispatch-mode=manual)",
    )
    orch.add_argument(
        "--control-backend",
        type=str,
        choices=["file", "redis"],
        default="file",
    )
    orch.add_argument(
        "--provenance-backend",
        type=str,
        choices=["jsonl", "weaviate", "off"],
        default="jsonl",
    )
    orch.add_argument("--cache-path", type=str, default="")
    orch.add_argument("--multi-start", action="store_true")

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
        "train-surrogates": run_train_surrogates_workflow,
        "openfoam-doe": run_openfoam_doe_workflow,
        "dress-rehearsal": run_dress_rehearsal_workflow,
        "orchestrate": run_orchestrate_workflow,
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
