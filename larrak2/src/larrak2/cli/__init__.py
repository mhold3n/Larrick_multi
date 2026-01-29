"""CLI modules for running optimization and evaluation."""

from .run_pareto import main as run_pareto_main
from .run_single import main as run_single_main

__all__ = ["run_pareto_main", "run_single_main"]
