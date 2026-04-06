"""Integration shims for consumers outside the optimization package."""

from .orchestration import CasadiSolverAdapter, SimpleSolverAdapter

__all__ = ["CasadiSolverAdapter", "SimpleSolverAdapter"]
