"""Optimization-only CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

from .run_workflows import (
    run_explore_exploit_workflow,
    run_pareto_grid_workflow,
    run_pareto_staged_workflow,
)


def _add_common_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bore-mm", type=float, default=80.0)
    parser.add_argument("--stroke-mm", type=float, default=90.0)
    parser.add_argument("--intake-port-area-m2", type=float, default=4.0e-4)
    parser.add_argument("--exhaust-port-area-m2", type=float, default=4.0e-4)
    parser.add_argument("--p-manifold-pa", type=float, default=101325.0)
    parser.add_argument("--p-back-pa", type=float, default=101325.0)
    parser.add_argument("--compression-ratio", type=float, default=10.0)
    parser.add_argument(
        "--fuel-name",
        type=str,
        default="gasoline",
        choices=["gasoline", "ethanol", "methanol"],
    )
    parser.add_argument("--overlap-deg", type=float, default=0.0)
    parser.add_argument("--intake-open-deg", type=float, default=0.0)
    parser.add_argument("--intake-close-deg", type=float, default=0.0)
    parser.add_argument("--exhaust-open-deg", type=float, default=0.0)
    parser.add_argument("--exhaust-close-deg", type=float, default=0.0)
    parser.add_argument("--openfoam-model-path", type=str, default="")
    parser.add_argument(
        "--calculix-stress-mode",
        type=str,
        default="nn",
        choices=["nn", "analytical"],
    )
    parser.add_argument("--calculix-model-path", type=str, default="")
    parser.add_argument(
        "--gear-loss-mode",
        type=str,
        default="physics",
        choices=["physics", "nn"],
    )
    parser.add_argument("--gear-loss-model-dir", type=str, default="")
    parser.add_argument(
        "--thermo-model",
        type=str,
        default="two_zone_eq_v1",
        choices=["two_zone_eq_v1"],
    )
    parser.add_argument("--thermo-constants-path", type=str, default="")
    parser.add_argument("--thermo-anchor-manifest", type=str, default="")
    parser.add_argument("--thermo-chemistry-profile-path", type=str, default="")
    parser.add_argument(
        "--tribology-scuff-method",
        type=str,
        default="auto",
        choices=["auto", "flash", "integral"],
    )
    parser.add_argument("--strict-data", dest="strict_data", action="store_true")
    parser.add_argument("--no-strict-data", dest="strict_data", action="store_false")
    parser.add_argument(
        "--strict-tribology-data", dest="strict_tribology_data", action="store_true"
    )
    parser.add_argument(
        "--no-strict-tribology-data", dest="strict_tribology_data", action="store_false"
    )
    parser.add_argument(
        "--surrogate-validation-mode",
        type=str,
        default="strict",
        choices=["strict", "warn", "off"],
    )
    parser.add_argument("--machining-mode", type=str, default="nn", choices=["nn", "analytical"])
    parser.add_argument("--machining-model-path", type=str, default="")
    parser.set_defaults(strict_data=True)
    parser.set_defaults(strict_tribology_data=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Larrak standalone optimization CLI")
    subparsers = parser.add_subparsers(dest="run_type", required=True)

    pg = subparsers.add_parser("pareto-grid", help="Pareto over an (rpm, torque) grid")
    pg.add_argument("--outdir-root", type=str, default="outputs/pareto_grid")
    pg.add_argument("--pop", type=int, default=64)
    pg.add_argument("--gen", type=int, default=50)
    pg.add_argument("--algorithm", type=str, default="nsga3", choices=["nsga2", "nsga3"])
    pg.add_argument("--partitions", type=int, default=4)
    pg.add_argument("--nsga3-max-ref-dirs", type=int, default=192)
    pg.add_argument("--fidelity", type=int, default=2, choices=[0, 1, 2])
    pg.add_argument("--seed", type=int, default=42)
    pg.add_argument("--verbose", action="store_true")
    pg.add_argument("--allow-nonproduction-paths", action="store_true")
    pg.add_argument("--rpm-list", type=str, default="")
    pg.add_argument("--torque-list", type=str, default="")
    pg.add_argument("--rpm-min", type=float, default=1000.0)
    pg.add_argument("--rpm-max", type=float, default=7000.0)
    pg.add_argument("--rpm-n", type=int, default=6)
    pg.add_argument("--torque-min", type=float, default=50.0)
    pg.add_argument("--torque-max", type=float, default=400.0)
    pg.add_argument("--torque-n", type=int, default=6)
    _add_common_engine_args(pg)

    ps = subparsers.add_parser("pareto-staged", help="Multi-fidelity staged Pareto")
    ps.add_argument("--pop", type=int, default=100)
    ps.add_argument("--gen", type=int, default=50)
    ps.add_argument("--promote", type=int, default=20)
    ps.add_argument("--rpm", type=float, default=3000.0)
    ps.add_argument("--torque", type=float, default=200.0)
    ps.add_argument("--seed", type=int, default=1)
    ps.add_argument("--outdir", type=str, default="outputs/pareto_staged")

    ee = subparsers.add_parser(
        "explore-exploit",
        help="Two-stage pipeline: Pareto/principles explore then slice refinement",
    )
    ee.add_argument("--pareto-dir", type=str, default="outputs/diagnostic_results")
    ee.add_argument(
        "--explore-source",
        type=str,
        default="principles",
        choices=["principles", "archive"],
    )
    ee.add_argument("--principles-profile", type=str, default="iso_litvin_v2")
    ee.add_argument("--principles-region-min-size", type=int, default=12)
    ee.add_argument("--principles-frontier-min-size", dest="principles_region_min_size", type=int)
    ee.add_argument("--principles-seed-count", type=int, default=64)
    ee.add_argument("--principles-root-max-iter", type=int, default=80)
    ee.add_argument("--principles-export-archive-dir", type=str, default="")
    ee.add_argument(
        "--principles-alignment-mode",
        type=str,
        default="blend",
        choices=["blend", "proxy_only", "canonical_only"],
    )
    ee.add_argument(
        "--principles-canonical-alignment-fidelity", type=int, default=1, choices=[0, 1, 2]
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
    )
    ee.add_argument("--top-k", type=int, default=1)
    ee.add_argument("--candidate-index", type=int, default=-1)
    ee.add_argument("--rank-weights", type=str, default="1,1,1,1,1,1")
    ee.add_argument("--refine-indices", type=str, default="")
    ee.add_argument(
        "--mode", type=str, default="eps_constraint", choices=["weighted_sum", "eps_constraint"]
    )
    ee.add_argument("--backend", type=str, default="casadi", choices=["casadi", "scipy"])
    ee.add_argument("--slice-method", type=str, default="sensitivity", choices=["sensitivity"])
    ee.add_argument("--active-k", type=int, default=None)
    ee.add_argument("--min-per-group", type=int, default=1)
    ee.add_argument("--trust-radius", type=float, default=None)
    ee.add_argument("--max-iter", type=int, default=80)
    ee.add_argument("--tol", type=float, default=1e-6)
    ee.add_argument("--eps-margin", type=float, default=0.02)
    ee.add_argument("--skip-tribology", action="store_true")
    ee.add_argument(
        "--thermo-symbolic-mode",
        type=str,
        default="strict",
        choices=["strict", "warn", "off"],
    )
    ee.add_argument("--thermo-symbolic-artifact-path", type=str, default="")
    ee.add_argument("--stack-model-path", type=str, default="")
    ee.add_argument("--ipopt-max-iter", type=int, default=None)
    ee.add_argument("--ipopt-tol", type=float, default=None)
    ee.add_argument("--ipopt-linear-solver", type=str, default=None)
    ee.add_argument("--allow-nonproduction-paths", action="store_true")
    ee.add_argument("--verbose", action="store_true")
    ee.add_argument("--architecture-probe-mode", action="store_true")
    ee.add_argument("--enforce-contract-routing", action="store_true")
    _add_common_engine_args(ee)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "pareto-grid": run_pareto_grid_workflow,
        "pareto-staged": run_pareto_staged_workflow,
        "explore-exploit": run_explore_exploit_workflow,
    }
    return dispatch[args.run_type](args)


if __name__ == "__main__":
    sys.exit(main())
