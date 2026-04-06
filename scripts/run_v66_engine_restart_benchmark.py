#!/usr/bin/env python3
"""Run the canonical v66 engine-restart benchmark from data/simulation_validation/v66_engine_restart_recipe.json.

This is the single supported entry point for v66 multitable benchmark runs: do not hand-assemble
`larrak-validate-sim engine-restart-benchmark` flags.

After any edit under openfoam_custom_solvers/larrakEngineFoam/, force a Docker wmake once via
`--refresh-custom-solver` or by setting `refresh_custom_solver` to true in the recipe for that run
(then set it back to false). New runtimeChemistryTable entries such as `rbfDiagEnvelopeScaleHO2`
require a rebuilt solver binary to take effect.

Usage:
  python scripts/run_v66_engine_restart_benchmark.py
  python scripts/run_v66_engine_restart_benchmark.py --outdir outputs/diagnostics/engine_restart_benchmark_live_parallel_v70x_chem323_my_run
  python scripts/run_v66_engine_restart_benchmark.py --recipe path/to/custom_recipe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_recipe(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Recipe must be a JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        default="data/simulation_validation/v66_engine_restart_recipe.json",
        help="Path to v66 recipe JSON (repo-relative or absolute)",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Override recipe outdir (required if recipe still contains v66_recipe_placeholder)",
    )
    parser.add_argument(
        "--refresh-custom-solver",
        action="store_true",
        help="Override recipe: rebuild custom larrakEngineFoam in Docker once (use after C++ changes)",
    )
    args = parser.parse_args()
    root = _repo_root()
    recipe_path = Path(args.recipe)
    if not recipe_path.is_absolute():
        recipe_path = (root / recipe_path).resolve()
    recipe = _load_recipe(recipe_path)

    outdir = str(args.outdir).strip() or str(recipe.get("outdir", "") or "").strip()
    if not outdir:
        print(
            "error: set outdir in the recipe JSON or pass --outdir "
            "(e.g. outputs/diagnostics/engine_restart_benchmark_live_parallel_vNNx_chem323_purpose/)",
            file=sys.stderr,
        )
        return 2
    if "v66_recipe_placeholder" in outdir and not str(args.outdir).strip():
        print(
            "error: replace v66_recipe_placeholder in the recipe outdir or pass --outdir",
            file=sys.stderr,
        )
        return 2

    strategy = str(recipe.get("strategy_config", "")).strip()
    tuned = str(recipe.get("tuned_params", "")).strip()
    handoff = str(recipe.get("handoff_artifact", "")).strip()
    run_dir = str(recipe.get("run_dir", "")).strip()
    profiles = list(recipe.get("profiles", []) or [])
    if not profiles:
        print("error: recipe must define profiles (non-empty list)", file=sys.stderr)
        return 2

    window_angle_deg = float(recipe.get("window_angle_deg", 0.01))
    docker_timeout_s = int(recipe.get("docker_timeout_s", 1800))
    solver_name = str(recipe.get("solver_name", "larrakEngineFoam")).strip() or "larrakEngineFoam"
    package_label = str(recipe.get("package_label", "") or "")
    refresh_runtime_tables = bool(recipe.get("refresh_runtime_tables", False))
    continue_across = bool(recipe.get("continue_across_remaining_stages", False))
    refresh_custom_solver = bool(recipe.get("refresh_custom_solver", False))
    if args.refresh_custom_solver:
        refresh_custom_solver = True
    docker_image = recipe.get("docker_image")
    docker_bin = recipe.get("docker_bin")

    tuning_block = dict(recipe.get("tuning_characterization") or {})
    tuning_characterization = None
    if bool(tuning_block.get("enabled")):
        knob_schema = str(
            tuning_block.get("knob_schema_path", "")
            or "data/simulation_validation/tuning_knob_schema_chem323_ignition_entry_v1.json"
        ).strip()
        table_cfg = str(tuning_block.get("table_config_path", "") or "").strip()
        experiments_jsonl = str(tuning_block.get("experiments_jsonl", "") or "").strip()
        if not table_cfg or not experiments_jsonl:
            print(
                "error: tuning_characterization.enabled requires table_config_path and experiments_jsonl",
                file=sys.stderr,
            )
            return 2
        strat_override = str(tuning_block.get("strategy_config_path", "") or "").strip()
        strategy_for_hashes = strat_override or strategy
        profile_tc = str(tuning_block.get("profile_name", "") or "").strip()
        tuning_characterization = {
            "enabled": True,
            "experiments_jsonl": experiments_jsonl,
            "knob_schema_path": knob_schema,
            "table_config_path": table_cfg,
            "strategy_config_path": strategy_for_hashes,
            "repo_root": str(root),
        }
        if profile_tc:
            tuning_characterization["profile_name"] = profile_tc

    from larrak2.simulation_validation.engine_restart_benchmark import (
        benchmark_engine_restart_profiles,
    )

    summary = benchmark_engine_restart_profiles(
        run_dir=root / run_dir,
        tuned_params_path=root / tuned,
        handoff_artifact_path=root / handoff,
        outdir=root / outdir,
        profiles=profiles,
        window_angle_deg=window_angle_deg,
        solver_name=solver_name,
        docker_timeout_s=docker_timeout_s,
        runtime_strategy_config=root / strategy,
        package_label=package_label,
        docker_image=str(docker_image) if docker_image else None,
        docker_bin=str(docker_bin) if docker_bin else None,
        refresh_runtime_tables=refresh_runtime_tables,
        continue_across_remaining_stages=continue_across,
        refresh_custom_solver=refresh_custom_solver,
        tuning_characterization=tuning_characterization,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
