"""Unified Training CLI."""

from __future__ import annotations

import argparse
import sys

from larrak2.training.workflows import (
    train_gear_gbr_workflow,
    train_gear_nn_workflow,
    train_openfoam_workflow,
    train_residual_workflow,
    train_scavenge_gbr_workflow,
)


def main():
    parser = argparse.ArgumentParser(description="Larrak2 Unified Training CLI")

    subparsers = parser.add_subparsers(dest="model_type", required=True, help="Model Type")

    # Common Arguments Factory
    def add_common_args(p):
        p.add_argument("--data", required=True, help="Input data path")
        p.add_argument("--outdir", default="models", help="Output directory")
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

    # 2. Gear NN
    p_gear_nn = subparsers.add_parser("gear-nn", help="Gear Loss Neural Network")
    add_common_args(p_gear_nn)
    add_nn_args(p_gear_nn)

    # 3. Gear GBR
    p_gear_gbr = subparsers.add_parser("gear-gbr", help="Gear GBR Ensemble")
    add_common_args(p_gear_gbr)

    # 4. Scavenge GBR
    p_scav = subparsers.add_parser("scavenge", help="Scavenge GBR Ensemble")
    add_common_args(p_scav)

    # 5. Residual
    p_resid = subparsers.add_parser("residual", help="Residual Surrogate (Efficiency/Loss)")
    add_common_args(p_resid)

    args = parser.parse_args()

    # Dispatch
    if args.model_type == "openfoam":
        train_openfoam_workflow(args)
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
