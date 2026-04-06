from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _new_modules(before: set[str]) -> set[str]:
    return set(sys.modules) - before


def test_runtime_package_imports_without_training_analysis_or_simulation() -> None:
    before = set(sys.modules)
    importlib.import_module("larrak_runtime")
    importlib.import_module("larrak_runtime.architecture")
    importlib.import_module("larrak_runtime.surrogate.stack")

    added = _new_modules(before)
    blocked = (
        "larrak2.training",
        "larrak2.analysis",
        "larrak2.simulation_validation",
        "larrak_runtime.training",
        "larrak_runtime.analysis",
        "larrak_runtime.simulation_validation",
    )
    assert not any(name.startswith(blocked) for name in added)


def test_optimization_package_imports_without_training_analysis_or_simulation() -> None:
    before = set(sys.modules)
    importlib.import_module("larrak_optimization")
    importlib.import_module("larrak_optimization.cli.run")
    importlib.import_module("larrak_optimization.pipelines.principles_frontier")
    importlib.import_module("larrak_optimization.pipelines.explore_exploit")

    added = _new_modules(before)
    blocked = (
        "larrak2.training",
        "larrak2.analysis",
        "larrak2.simulation_validation",
        "larrak_optimization.training",
        "larrak_optimization.analysis",
        "larrak_optimization.simulation_validation",
    )
    assert not any(name.startswith(blocked) for name in added)


def test_current_orchestration_bridge_uses_extracted_optimizer() -> None:
    from larrak2.orchestration import orchestrator as orchestrator_mod
    from larrak2.orchestration.adapters import solver_adapter as solver_adapter_mod

    assert orchestrator_mod.evaluate_production_gate.__module__.startswith("larrak_optimization")
    assert solver_adapter_mod.CasadiSolverAdapter.__module__ == (
        "larrak_optimization.integrations.orchestration"
    )
    assert solver_adapter_mod.SimpleSolverAdapter.__module__ == (
        "larrak_optimization.integrations.orchestration"
    )


def test_compatibility_shims_route_to_extracted_packages() -> None:
    from larrak2.adapters import casadi_refine as casadi_refine_mod
    from larrak2.adapters import pymoo_problem as pymoo_problem_mod
    from larrak2.cli import run_pareto as run_pareto_mod
    from larrak2.pipelines import principles_frontier as principles_frontier_mod

    assert casadi_refine_mod.refine_candidate.__module__.startswith("larrak_optimization")
    assert pymoo_problem_mod.ParetoProblem.__module__.startswith("larrak_optimization")
    assert run_pareto_mod.main.__module__.startswith("larrak_optimization")
    assert principles_frontier_mod.synthesize_principles_frontier.__module__.startswith(
        "larrak_optimization"
    )


def test_standalone_cli_parser_exposes_three_optimization_frames() -> None:
    from larrak_optimization.cli.run import build_parser

    parser = build_parser()

    pg = parser.parse_args(["pareto-grid", "--rpm-list", "3000", "--torque-list", "200"])
    assert pg.run_type == "pareto-grid"

    ps = parser.parse_args(["pareto-staged", "--rpm", "3000", "--torque", "200"])
    assert ps.run_type == "pareto-staged"

    ee = parser.parse_args(["explore-exploit", "--explore-source", "principles"])
    assert ee.run_type == "explore-exploit"
    assert ee.explore_source == "principles"


def test_standalone_package_manifests_are_present() -> None:
    runtime_manifest = Path("packages/larrak-runtime/pyproject.toml")
    optimization_manifest = Path("packages/larrak-optimization/pyproject.toml")

    assert runtime_manifest.exists()
    assert optimization_manifest.exists()

    assert 'name = "larrak-runtime"' in runtime_manifest.read_text(encoding="utf-8")
    assert 'name = "larrak-optimization"' in optimization_manifest.read_text(encoding="utf-8")
