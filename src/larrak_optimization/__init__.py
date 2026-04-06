"""Standalone optimization package extracted from the monorepo."""

from importlib import import_module
from typing import Any

__all__ = [
    "BALANCED_THRESHOLD_PROFILE",
    "CandidateEntry",
    "CandidateStore",
    "PrinciplesFrontierResult",
    "STRICT_PRODUCTION_PROFILE",
    "evaluate_production_gate",
    "load_principles_profile",
    "required_pareto_min",
    "run_two_stage_pipeline",
    "synthesize_principles_frontier",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CandidateEntry",
        "CandidateStore",
        "BALANCED_THRESHOLD_PROFILE",
        "STRICT_PRODUCTION_PROFILE",
        "evaluate_production_gate",
        "required_pareto_min",
    }:
        mod = import_module("larrak_optimization.optimization")
        return getattr(mod, name)
    if name in {
        "PrinciplesFrontierResult",
        "load_principles_profile",
        "synthesize_principles_frontier",
    }:
        mod = import_module("larrak_optimization.pipelines.principles_frontier")
        return getattr(mod, name)
    if name == "run_two_stage_pipeline":
        mod = import_module("larrak_optimization.pipelines.explore_exploit")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
