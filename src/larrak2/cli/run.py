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
    run_explore_exploit_workflow,
    run_openfoam_doe_workflow,
    run_orchestrate_workflow,
    run_pareto_grid_workflow,
    run_pareto_staged_workflow,
    run_promote_openfoam_artifact_workflow,
    run_train_stack_surrogate_workflow,
    run_train_surrogates_workflow,
    run_train_thermo_symbolic_workflow,
)
from larrak2.core.artifact_paths import (
    DEFAULT_CALCULIX_NN_DIR,
    DEFAULT_HIFI_SURROGATE_DIR,
    DEFAULT_OPENFOAM_NN_DIR,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Larrak2 Unified Run CLI")
    subparsers = parser.add_subparsers(dest="run_type", required=True, help="Run type")

    # --- Pareto Grid ---
    pg = subparsers.add_parser("pareto-grid", help="Pareto over (rpm, torque) grid")
    pg.add_argument("--outdir-root", type=str, default="outputs/pareto_grid")
    pg.add_argument("--pop", type=int, default=64)
    pg.add_argument("--gen", type=int, default=50)
    pg.add_argument("--algorithm", type=str, default="nsga3", choices=["nsga2", "nsga3"])
    pg.add_argument("--partitions", type=int, default=4)
    pg.add_argument("--nsga3-max-ref-dirs", type=int, default=192)
    pg.add_argument("--fidelity", type=int, default=2, choices=[0, 1, 2])
    pg.add_argument("--seed", type=int, default=42)
    pg.add_argument("--verbose", action="store_true")
    pg.add_argument(
        "--allow-nonproduction-paths",
        action="store_true",
        help="Allow non-production fallback paths and mark manifest as non-release",
    )
    pg.add_argument("--bore-mm", type=float, default=80.0)
    pg.add_argument("--stroke-mm", type=float, default=90.0)
    pg.add_argument("--intake-port-area-m2", type=float, default=4.0e-4)
    pg.add_argument("--exhaust-port-area-m2", type=float, default=4.0e-4)
    pg.add_argument("--p-manifold-pa", type=float, default=101325.0)
    pg.add_argument("--p-back-pa", type=float, default=101325.0)
    pg.add_argument("--compression-ratio", type=float, default=10.0)
    pg.add_argument(
        "--fuel-name",
        type=str,
        default="gasoline",
        choices=["gasoline", "ethanol", "methanol"],
    )
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
    pg.add_argument(
        "--thermo-model",
        type=str,
        default="two_zone_eq_v1",
        choices=["two_zone_eq_v1"],
    )
    pg.add_argument("--thermo-constants-path", type=str, default="")
    pg.add_argument("--thermo-anchor-manifest", type=str, default="")
    pg.add_argument("--thermo-chemistry-profile-path", type=str, default="")
    pg.add_argument(
        "--tribology-scuff-method",
        type=str,
        default="auto",
        choices=["auto", "flash", "integral"],
    )
    pg.add_argument("--strict-tribology-data", dest="strict_tribology_data", action="store_true")
    pg.add_argument(
        "--no-strict-tribology-data",
        dest="strict_tribology_data",
        action="store_false",
    )
    pg.set_defaults(strict_tribology_data=None)
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
    ps.add_argument("--outdir", type=str, default="outputs/pareto_staged")

    # --- Active Learning ---
    al = subparsers.add_parser("active-learning", help="Staged opt → uncertainty → truth sims")
    al.add_argument("--outdir", type=str, default="outputs/active_learning")
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
    od.add_argument("--outdir", type=str, default="outputs/openfoam_doe")
    od.add_argument("--runs-root", type=str, default="outputs/openfoam_doe/runs")
    od.add_argument("--jsonl", type=str, default="outputs/openfoam_doe/results.jsonl")
    od.add_argument("--checkpoint", type=str, default="outputs/openfoam_doe/checkpoint.json")
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

    # --- Promote OpenFOAM Artifact ---
    po = subparsers.add_parser(
        "promote-openfoam-artifact",
        help="Promote a staged real OpenFOAM artifact bundle to the canonical runtime path",
    )
    po.add_argument("--staged-dir", type=str, default="")
    po.add_argument("--bundle-root", type=str, default="outputs/openfoam_authority")
    po.add_argument("--canonical-dir", type=str, default=str(DEFAULT_OPENFOAM_NN_DIR))
    po.add_argument(
        "--backup-root",
        type=str,
        default="outputs/artifacts/surrogates/openfoam_nn/archive",
    )

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
    ts.add_argument("--openfoam-outdir", type=str, default=str(DEFAULT_OPENFOAM_NN_DIR))
    ts.add_argument("--openfoam-name", type=str, default="openfoam_breathing.pt")
    ts.add_argument("--openfoam-epochs", type=int, default=120)
    ts.add_argument("--openfoam-lr", type=float, default=1e-3)
    ts.add_argument("--openfoam-hidden", type=str, default="64,64")
    ts.add_argument("--openfoam-weight-decay", type=float, default=0.0)
    ts.add_argument(
        "--openfoam-data-provenance-kind",
        type=str,
        default="",
        choices=["", "synthetic_rehearsal", "doe_generated", "truth_records"],
    )
    ts.add_argument("--openfoam-authoritative-for-strict-f2", action="store_true")
    ts.add_argument("--openfoam-anchor-manifest", type=str, default="")
    ts.add_argument("--openfoam-truth-source-summary", type=str, default="")
    ts.add_argument(
        "--openfoam-authority-bundle-root", type=str, default="outputs/openfoam_authority"
    )
    ts.add_argument("--calculix-data", type=str, default="")
    ts.add_argument(
        "--calculix-template",
        type=str,
        default="",
        help="CalculiX .inp template used to generate training data when --calculix-data is not provided",
    )
    ts.add_argument("--calculix-runs-per-condition", type=int, default=4)
    ts.add_argument("--calculix-solver", type=str, default="ccx")
    ts.add_argument("--calculix-outdir", type=str, default=str(DEFAULT_CALCULIX_NN_DIR))
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

    # --- Train Stack Surrogate ---
    tss = subparsers.add_parser(
        "train-stack-surrogate",
        help="Train global surrogate stack artifact for symbolic CasADi refinement",
    )
    tss.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory for stack surrogate artifact (default: auto by --fidelity)",
    )
    tss.add_argument(
        "--name",
        type=str,
        default="",
        help="Artifact name (default: auto by --fidelity)",
    )
    tss.add_argument(
        "--dataset",
        type=str,
        default="",
        help="NPZ dataset path with X and Y arrays (overrides --pareto-dir)",
    )
    tss.add_argument(
        "--pareto-dir",
        type=str,
        default="",
        help="Pareto archive directory with pareto_X/F/G arrays used to build training data",
    )
    tss.add_argument("--fidelity", type=int, default=1, choices=[0, 1, 2])
    tss.add_argument("--rpm", type=float, default=3000.0)
    tss.add_argument("--torque", type=float, default=200.0)
    tss.add_argument("--hidden", type=str, default="128,128")
    tss.add_argument("--activation", type=str, default="relu", choices=["relu", "leaky_relu"])
    tss.add_argument("--leaky-relu-slope", type=float, default=0.01)
    tss.add_argument("--epochs", type=int, default=200)
    tss.add_argument("--lr", type=float, default=1e-3)
    tss.add_argument("--weight-decay", type=float, default=1e-6)
    tss.add_argument("--val-frac", type=float, default=0.2)
    tss.add_argument("--seed", type=int, default=42)
    tss.add_argument("--verbose", action="store_true")

    # --- Train Thermo Symbolic ---
    tts = subparsers.add_parser(
        "train-thermo-symbolic",
        help="Train thermo symbolic surrogate artifact for CasADi overlay",
    )
    tts.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory for thermo symbolic artifact (default: auto by --fidelity)",
    )
    tts.add_argument(
        "--name",
        type=str,
        default="",
        help="Artifact name (default: auto by --fidelity)",
    )
    tts.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional NPZ dataset path with X and Y arrays",
    )
    tts.add_argument(
        "--dataset-out",
        type=str,
        default="",
        help="Optional path to dump generated dataset NPZ",
    )
    tts.add_argument("--n-samples", type=int, default=256)
    tts.add_argument("--fidelity", type=int, default=1, choices=[0, 1, 2])
    tts.add_argument("--rpm", type=float, default=3000.0)
    tts.add_argument("--torque", type=float, default=200.0)
    tts.add_argument(
        "--objective-names",
        type=str,
        default="eta_comb_gap,eta_exp_gap,motion_law_penalty",
        help="Comma-separated objective names to model",
    )
    tts.add_argument(
        "--constraint-names",
        type=str,
        default="",
        help="Comma-separated constraint names to model (defaults to thermo constraints by fidelity)",
    )
    tts.add_argument("--val-frac", type=float, default=0.2)
    tts.add_argument("--seed", type=int, default=42)
    tts.add_argument(
        "--thermo-model",
        type=str,
        default="two_zone_eq_v1",
        choices=["two_zone_eq_v1"],
    )
    tts.add_argument("--thermo-constants-path", type=str, default="")
    tts.add_argument("--thermo-anchor-manifest", type=str, default="")
    tts.add_argument(
        "--surrogate-validation-mode",
        type=str,
        default="strict",
        choices=["strict", "warn", "off"],
    )
    tts.add_argument("--verbose", action="store_true")

    # --- Dress Rehearsal ---
    dr = subparsers.add_parser(
        "dress-rehearsal",
        help="Verify surrogate readiness, run optimization, and finish CEM validation gates",
    )
    dr.add_argument("--outdir", type=str, default="outputs/dress_rehearsal")
    dr.add_argument("--pop", type=int, default=16)
    dr.add_argument("--gen", type=int, default=5)
    dr.add_argument("--algorithm", type=str, default="nsga3", choices=["nsga2", "nsga3"])
    dr.add_argument("--partitions", type=int, default=4)
    dr.add_argument("--nsga3-max-ref-dirs", type=int, default=192)
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
        default="downselect",
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
    dr.add_argument(
        "--allow-nonproduction-paths",
        action="store_true",
        help="Allow non-production fallback paths and mark manifest as non-release",
    )
    dr.add_argument("--verbose", action="store_true")
    dr.add_argument("--bore-mm", type=float, default=80.0)
    dr.add_argument("--stroke-mm", type=float, default=90.0)
    dr.add_argument("--intake-port-area-m2", type=float, default=4.0e-4)
    dr.add_argument("--exhaust-port-area-m2", type=float, default=4.0e-4)
    dr.add_argument("--p-manifold-pa", type=float, default=101325.0)
    dr.add_argument("--p-back-pa", type=float, default=101325.0)
    dr.add_argument("--compression-ratio", type=float, default=10.0)
    dr.add_argument(
        "--fuel-name",
        type=str,
        default="gasoline",
        choices=["gasoline", "ethanol", "methanol"],
    )
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

    # --- Explore -> Exploit ---
    ee = subparsers.add_parser(
        "explore-exploit",
        help="Two-stage pipeline: Pareto explore then CasADi slice refinement",
    )
    ee.add_argument("--pareto-dir", type=str, default="outputs/diagnostic_results")
    ee.add_argument(
        "--explore-source",
        type=str,
        default="principles",
        choices=["principles", "archive"],
        help="Candidate source policy for exploit stage (principles synthesis or archive)",
    )
    ee.add_argument(
        "--principles-profile",
        type=str,
        default="iso_litvin_v2",
        help="Principles frontier profile id or JSON path",
    )
    ee.add_argument(
        "--principles-region-min-size",
        type=int,
        default=12,
        help="Minimum representative candidate count required for a principles region to be ready",
    )
    ee.add_argument(
        "--principles-frontier-min-size",
        dest="principles_region_min_size",
        type=int,
        help="Deprecated alias for --principles-region-min-size",
    )
    ee.add_argument(
        "--principles-seed-count",
        type=int,
        default=64,
        help="Deprecated compatibility input recorded in the manifest; default principles search is deterministic and seed-state driven",
    )
    ee.add_argument(
        "--principles-root-max-iter",
        type=int,
        default=80,
        help="Maximum iterations per deterministic principles continuation stage",
    )
    ee.add_argument(
        "--principles-export-archive-dir",
        type=str,
        default="",
        help="Optional output archive dir for synthesized principles frontier (defaults to <outdir>/principles_pareto)",
    )
    ee.add_argument(
        "--principles-alignment-mode",
        type=str,
        default="blend",
        choices=["blend", "proxy_only", "canonical_only"],
        help="How principles search mixes reduced-order proxy scoring and canonical-alignment scoring",
    )
    ee.add_argument(
        "--principles-canonical-alignment-fidelity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Canonical evaluator fidelity used by principles alignment scoring",
    )
    ee.add_argument("--outdir", type=str, default="outputs/explore_exploit")
    ee.add_argument("--run-explore", action="store_true")
    ee.add_argument("--pop", type=int, default=64)
    ee.add_argument("--gen", type=int, default=50)
    ee.add_argument("--algorithm", type=str, default="nsga3", choices=["nsga2", "nsga3"])
    ee.add_argument("--partitions", type=int, default=4)
    ee.add_argument("--nsga3-max-ref-dirs", type=int, default=192)
    ee.add_argument("--rpm", type=float, default=3000.0)
    ee.add_argument("--torque", type=float, default=200.0)
    ee.add_argument("--seed", type=int, default=42)
    ee.add_argument("--explore-fidelity", type=int, default=2, choices=[0, 1, 2])
    ee.add_argument("--hifi-fidelity", type=int, default=2, choices=[0, 1, 2])
    ee.add_argument(
        "--hifi-constraint-phase",
        type=str,
        default="downselect",
        choices=["explore", "downselect"],
        help="Constraint phase for high-fidelity evaluate/downselect stage",
    )
    ee.add_argument("--top-k", type=int, default=1)
    ee.add_argument(
        "--candidate-index",
        type=int,
        default=-1,
        help="Explicit index from selected candidate source; -1 means ranked top-k selection",
    )
    ee.add_argument("--rank-weights", type=str, default="1,1,1,1,1,1")
    ee.add_argument("--refine-indices", type=str, default="")
    ee.add_argument(
        "--mode",
        type=str,
        default="eps_constraint",
        choices=["weighted_sum", "eps_constraint"],
    )
    ee.add_argument(
        "--backend",
        type=str,
        default="casadi",
        choices=["casadi", "scipy"],
    )
    ee.add_argument(
        "--slice-method",
        type=str,
        default="sensitivity",
        choices=["sensitivity"],
    )
    ee.add_argument("--active-k", type=int, default=None)
    ee.add_argument("--min-per-group", type=int, default=1)
    ee.add_argument("--trust-radius", type=float, default=None)
    ee.add_argument("--max-iter", type=int, default=80)
    ee.add_argument("--tol", type=float, default=1e-6)
    ee.add_argument("--eps-margin", type=float, default=0.02)
    ee.add_argument("--skip-tribology", action="store_true")
    ee.add_argument(
        "--thermo-model",
        type=str,
        default="two_zone_eq_v1",
        choices=["two_zone_eq_v1"],
    )
    ee.add_argument("--thermo-constants-path", type=str, default="")
    ee.add_argument("--thermo-anchor-manifest", type=str, default="")
    ee.add_argument("--thermo-chemistry-profile-path", type=str, default="")
    ee.add_argument(
        "--thermo-symbolic-mode",
        type=str,
        default="strict",
        choices=["strict", "warn", "off"],
    )
    ee.add_argument(
        "--thermo-symbolic-artifact-path",
        type=str,
        default="",
        help="Thermo symbolic artifact path for symbolic overlay",
    )
    ee.add_argument(
        "--stack-model-path",
        type=str,
        default="",
        help="Stack surrogate artifact path for symbolic CasADi refinement",
    )
    ee.add_argument("--ipopt-max-iter", type=int, default=None)
    ee.add_argument("--ipopt-tol", type=float, default=None)
    ee.add_argument("--ipopt-linear-solver", type=str, default=None)
    ee.add_argument("--bore-mm", type=float, default=80.0)
    ee.add_argument("--stroke-mm", type=float, default=90.0)
    ee.add_argument("--intake-port-area-m2", type=float, default=4.0e-4)
    ee.add_argument("--exhaust-port-area-m2", type=float, default=4.0e-4)
    ee.add_argument("--p-manifold-pa", type=float, default=101325.0)
    ee.add_argument("--p-back-pa", type=float, default=101325.0)
    ee.add_argument("--compression-ratio", type=float, default=10.0)
    ee.add_argument(
        "--fuel-name",
        type=str,
        default="gasoline",
        choices=["gasoline", "ethanol", "methanol"],
    )
    ee.add_argument("--overlap-deg", type=float, default=0.0)
    ee.add_argument("--intake-open-deg", type=float, default=0.0)
    ee.add_argument("--intake-close-deg", type=float, default=0.0)
    ee.add_argument("--exhaust-open-deg", type=float, default=0.0)
    ee.add_argument("--exhaust-close-deg", type=float, default=0.0)
    ee.add_argument("--openfoam-model-path", type=str, default="")
    ee.add_argument(
        "--calculix-stress-mode",
        type=str,
        default="nn",
        choices=["nn", "analytical"],
    )
    ee.add_argument("--calculix-model-path", type=str, default="")
    ee.add_argument(
        "--gear-loss-mode",
        type=str,
        default="physics",
        choices=["physics", "nn"],
    )
    ee.add_argument("--gear-loss-model-dir", type=str, default="")
    ee.add_argument(
        "--tribology-scuff-method",
        type=str,
        default="auto",
        choices=["auto", "flash", "integral"],
    )
    ee.add_argument("--strict-tribology-data", dest="strict_tribology_data", action="store_true")
    ee.add_argument(
        "--no-strict-tribology-data",
        dest="strict_tribology_data",
        action="store_false",
    )
    ee.set_defaults(strict_tribology_data=None)
    ee.add_argument(
        "--enforce-contract-routing",
        action="store_true",
        help="Fail if observed edge engine_mode violates fidelity routing policy",
    )
    ee.add_argument(
        "--architecture-probe-mode",
        action="store_true",
        help="Emit explore-exploit manifest/contract artifacts even when downselect has no hard-feasible winner",
    )
    ee.add_argument(
        "--allow-nonproduction-paths",
        action="store_true",
        help="Allow non-production fallback paths and mark manifest as non-release",
    )
    ee.add_argument(
        "--surrogate-validation-mode",
        type=str,
        default="strict",
        choices=["strict", "warn", "off"],
    )
    ee.add_argument(
        "--strict-data",
        dest="strict_data",
        action="store_true",
        help="Enable strict data-path checks (default)",
    )
    ee.add_argument(
        "--no-strict-data",
        dest="strict_data",
        action="store_false",
        help="Disable strict data-path checks",
    )
    ee.set_defaults(strict_data=True)
    ee.add_argument(
        "--machining-mode",
        type=str,
        default="nn",
        choices=["nn", "analytical"],
    )
    ee.add_argument(
        "--machining-model-path",
        type=str,
        default="",
        help="Override machining NN artifact path",
    )
    ee.add_argument("--verbose", action="store_true")

    # --- Backend Orchestration ---
    orch = subparsers.add_parser(
        "orchestrate",
        help="Backend orchestration loop (no GUI)",
    )
    orch.add_argument("--outdir", type=str, default="outputs/orchestration")
    orch.add_argument("--rpm", type=float, default=3000.0)
    orch.add_argument("--torque", type=float, default=200.0)
    orch.add_argument("--fidelity", type=int, default=2, choices=[0, 1, 2])
    orch.add_argument("--bore-mm", type=float, default=80.0)
    orch.add_argument("--stroke-mm", type=float, default=90.0)
    orch.add_argument("--intake-port-area-m2", type=float, default=4.0e-4)
    orch.add_argument("--exhaust-port-area-m2", type=float, default=4.0e-4)
    orch.add_argument("--p-manifold-pa", type=float, default=101325.0)
    orch.add_argument("--p-back-pa", type=float, default=101325.0)
    orch.add_argument("--compression-ratio", type=float, default=10.0)
    orch.add_argument(
        "--fuel-name",
        type=str,
        default="gasoline",
        choices=["gasoline", "ethanol", "methanol"],
    )
    orch.add_argument(
        "--constraint-phase",
        type=str,
        default="downselect",
        choices=["explore", "downselect"],
        help="Constraint phase used for release-readiness gating",
    )
    orch.add_argument(
        "--enforce-contract-routing",
        action="store_true",
        help="Fail if observed edge engine_mode violates fidelity routing policy",
    )
    orch.add_argument("--seed", type=int, default=42)
    orch.add_argument("--sim-budget", type=int, default=32)
    orch.add_argument("--batch-size", type=int, default=16)
    orch.add_argument("--max-iterations", type=int, default=8)
    orch.add_argument(
        "--truth-dispatch-mode",
        type=str,
        choices=["off", "manual", "auto"],
        default="auto",
    )
    orch.add_argument(
        "--truth-plan",
        type=str,
        default="",
        help="Path to JSON truth plan (required for --truth-dispatch-mode=manual)",
    )
    orch.add_argument("--truth-auto-top-k", type=int, default=2)
    orch.add_argument("--truth-auto-min-uncertainty", type=float, default=0.0)
    orch.add_argument("--truth-auto-min-feasibility", type=float, default=0.0)
    orch.add_argument("--truth-auto-min-pred-quantile", type=float, default=0.0)
    orch.add_argument(
        "--hifi-model-dir",
        type=str,
        default=str(DEFAULT_HIFI_SURROGATE_DIR),
        help="HiFi surrogate model directory",
    )
    orch.add_argument(
        "--allow-heuristic-surrogate-fallback",
        action="store_true",
        help="Explicit non-production override to allow heuristic surrogate fallback",
    )
    orch.add_argument(
        "--allow-nonproduction-paths",
        action="store_true",
        help="Allow non-production fallback paths and mark manifest as non-release",
    )
    orch.add_argument(
        "--surrogate-validation-mode",
        type=str,
        default="strict",
        choices=["strict", "warn", "off"],
    )
    orch.add_argument(
        "--thermo-symbolic-mode",
        type=str,
        default="strict",
        choices=["strict", "warn", "off"],
    )
    orch.add_argument(
        "--thermo-symbolic-artifact-path",
        type=str,
        default="",
        help="Thermo symbolic artifact path for CasADi thermo overlay",
    )
    orch.add_argument(
        "--stack-model-path",
        type=str,
        default="",
        help="Stack surrogate artifact path for CasADi symbolic refinement",
    )
    orch.add_argument("--ipopt-max-iter", type=int, default=None)
    orch.add_argument("--ipopt-tol", type=float, default=None)
    orch.add_argument("--ipopt-linear-solver", type=str, default=None)
    orch.add_argument(
        "--thermo-constants-path",
        type=str,
        default="",
        help="Override thermo constants path for orchestration evaluations",
    )
    orch.add_argument(
        "--thermo-anchor-manifest",
        type=str,
        default="",
        help="Override thermo anchor manifest path for orchestration evaluations",
    )
    orch.add_argument(
        "--thermo-chemistry-profile-path",
        type=str,
        default="",
        help="Override hybrid chemistry profile path for orchestration evaluations",
    )
    orch.add_argument(
        "--strict-data",
        dest="strict_data",
        action="store_true",
        help="Enable strict data-path checks (default)",
    )
    orch.add_argument(
        "--no-strict-data",
        dest="strict_data",
        action="store_false",
        help="Disable strict data-path checks",
    )
    orch.set_defaults(strict_data=True)
    orch.add_argument("--strict-tribology-data", dest="strict_tribology_data", action="store_true")
    orch.add_argument(
        "--no-strict-tribology-data",
        dest="strict_tribology_data",
        action="store_false",
    )
    orch.set_defaults(strict_tribology_data=None)
    orch.add_argument(
        "--tribology-scuff-method",
        type=str,
        default="auto",
        choices=["auto", "flash", "integral"],
    )
    orch.add_argument(
        "--machining-mode",
        type=str,
        default="nn",
        choices=["nn", "analytical"],
    )
    orch.add_argument(
        "--machining-model-path",
        type=str,
        default="",
        help="Override machining NN artifact path",
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
    subparsers.add_parser("sensitivity", help="Sensitivity & Constraint Analysis")
    # No args for now, uses hardcoded mid-bounds or diagnostics

    # --- Diagnostic ---
    diag = subparsers.add_parser("diagnostic", help="Diagnostic Pareto Run")
    diag.add_argument("--pop", type=int, default=64)
    diag.add_argument("--gen", type=int, default=50)
    diag.add_argument("--rpm", type=float, default=3000.0)
    diag.add_argument("--torque", type=float, default=200.0)
    diag.add_argument("--fidelity", type=int, default=1)
    diag.add_argument("--seed", type=int, default=123)
    diag.add_argument("--outdir", type=str, default="outputs/diagnostic_results")

    args = parser.parse_args()

    dispatch = {
        "pareto-grid": run_pareto_grid_workflow,
        "pareto-staged": run_pareto_staged_workflow,
        "active-learning": run_active_learning_workflow,
        "train-surrogates": run_train_surrogates_workflow,
        "train-stack-surrogate": run_train_stack_surrogate_workflow,
        "train-thermo-symbolic": run_train_thermo_symbolic_workflow,
        "openfoam-doe": run_openfoam_doe_workflow,
        "promote-openfoam-artifact": run_promote_openfoam_artifact_workflow,
        "dress-rehearsal": run_dress_rehearsal_workflow,
        "explore-exploit": run_explore_exploit_workflow,
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
