"""CLI entrypoints for the standalone optimization package."""

from .run import main
from .run_pareto import main as run_pareto_main

__all__ = ["main", "run_pareto_main"]
