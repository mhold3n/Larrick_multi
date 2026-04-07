"""Unified Training CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from larrak_runtime.core.artifact_paths import (
    DEFAULT_CALCULIX_NN_DIR,
    DEFAULT_GEAR_LOSS_NN_DIR,
    DEFAULT_OPENFOAM_NN_DIR,
    DEFAULT_SURROGATE_V1_DIR,
)
from larrak_simulation.training.workflows import (
    train_calculix_workflow,
    train_gear_gbr_workflow,
    train_gear_nn_workflow,
    train_openfoam_workflow,
    train_residual_workflow,
    train_scavenge_gbr_workflow,
)


def _add_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validation-regime",
        default="",
        choices=[
            "",
            "chemistry",
            "spray",
            "reacting_flow",
            "reacting-flow",
            "closed_cylinder",
            "closed-cylinder",
            "full_handoff",
            "full-handoff",
            "suite",
        ],
        help="Optional simulation-validation preflight to run before training.",
    )
    parser.add_argument(
        "--validation-config",
        default="",
        help="Path to the simulation-validation config JSON for the preflight gate.",
    )
    parser.add_argument(
        "--validation-outdir",
        default="",
        help="Optional output directory for validation artifacts.",
    )


def _maybe_run_validation_preflight(args: argparse.Namespace) -> None:
    regime = str(getattr(args, "validation_regime", "")).strip()
    config = str(getattr(args, "validation_config", "")).strip()
    outdir_raw = str(getattr(args, "validation_outdir", "")).strip()

    if not regime and not config:
        return
    if not regime or not config:
        raise ValueError(
            "Training validation preflight requires both --validation-regime and "
            "--validation-config."
        )

    from larrak2.cli.validate_simulation import run_validation_preflight

    outdir = outdir_raw or str(
        Path("outputs") / "validation" / "pretrain" / str(getattr(args, "model_type", "unknown"))
    )
    print(
        f"Running simulation-validation preflight: regime={regime} config={config} outdir={outdir}"
    )
    code = run_validation_preflight(regime, config_path=config, outdir=outdir)
    if code != 0:
        raise RuntimeError(
            f"Training blocked by simulation validation preflight ({regime}). "
            f"See artifacts in {outdir}."
        )


def main():
    parser = argparse.ArgumentParser(description="Larrak2 Unified Training CLI")

    subparsers = parser.add_subparsers(dest="model_type", required=True, help="Model Type")

    # Common Arguments Factory
    def add_common_args(p):
        p.add_argument("--data", required=True, help="Input data path")
        p.add_argument("--outdir", default="outputs/artifacts/surrogates", help="Output directory")
        p.add_argument("--seed", type=int, default=42, help="Random seed")

    def add_nn_args(p):
        p.add_argument("--epochs", type=int, default=1000)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--hidden", type=str, default=None, help="Hidden layers (e.g. 64,64 or 64)")
        p.add_argument("--weight-decay", type=float, default=0.0)

    # 1. OpenFOAM NN
    p_of = subparsers.add_parser("openfoam", help="OpenFOAM Neural Network Surrogate")
    add_common_args(p_of)
    add_nn_args(p_of)
    p_of.add_argument("--name", default="openfoam_breathing.pt")
    p_of.add_argument(
        "--data-provenance-kind",
        type=str,
        default="synthetic_rehearsal",
        choices=["synthetic_rehearsal", "doe_generated", "truth_records"],
    )
    p_of.add_argument("--authoritative-for-strict-f2", action="store_true")
    p_of.add_argument("--anchor-manifest", default="")
    p_of.add_argument("--truth-source-summary", default="")
    p_of.add_argument("--authority-bundle-root", default="outputs/openfoam_authority")
    p_of.add_argument("--authority-run-id", default="")
    p_of.add_argument("--source-metadata-json", default="")
    p_of.add_argument("--doe-template-path", default="")
    _add_validation_args(p_of)
    p_of.set_defaults(outdir=str(DEFAULT_OPENFOAM_NN_DIR))

    # 1b. CalculiX NN
    p_ccx = subparsers.add_parser("calculix", help="CalculiX Stress Neural Network Surrogate")
    add_common_args(p_ccx)
    add_nn_args(p_ccx)
    p_ccx.add_argument("--name", default="calculix_stress.pt")
    _add_validation_args(p_ccx)
    p_ccx.set_defaults(outdir=str(DEFAULT_CALCULIX_NN_DIR))

    # 2. Gear NN
    p_gear_nn = subparsers.add_parser("gear-nn", help="Gear Loss Neural Network")
    add_common_args(p_gear_nn)
    add_nn_args(p_gear_nn)
    _add_validation_args(p_gear_nn)
    p_gear_nn.set_defaults(outdir=str(DEFAULT_GEAR_LOSS_NN_DIR))

    # 3. Gear GBR
    p_gear_gbr = subparsers.add_parser("gear-gbr", help="Gear GBR Ensemble")
    add_common_args(p_gear_gbr)
    _add_validation_args(p_gear_gbr)
    p_gear_gbr.set_defaults(outdir=str(DEFAULT_SURROGATE_V1_DIR))

    # 4. Scavenge GBR
    p_scav = subparsers.add_parser("scavenge", help="Scavenge GBR Ensemble")
    add_common_args(p_scav)
    _add_validation_args(p_scav)
    p_scav.set_defaults(outdir=str(DEFAULT_SURROGATE_V1_DIR))

    # 5. Residual
    p_resid = subparsers.add_parser("residual", help="Residual Surrogate (Efficiency/Loss)")
    add_common_args(p_resid)
    _add_validation_args(p_resid)
    p_resid.set_defaults(outdir=str(DEFAULT_SURROGATE_V1_DIR))

    args = parser.parse_args()
    _maybe_run_validation_preflight(args)

    # Dispatch
    if args.model_type == "openfoam":
        train_openfoam_workflow(args)
    elif args.model_type == "calculix":
        train_calculix_workflow(args)
    elif args.model_type == "gear-nn":
        train_gear_nn_workflow(args)
    elif args.model_type == "gear-gbr":
        train_gear_gbr_workflow(args)
    elif args.model_type == "scavenge":
        train_scavenge_gbr_workflow(args)
    elif args.model_type == "residual":
        train_residual_workflow(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
