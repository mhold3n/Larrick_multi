"""Compatibility shim to the extracted optimization orchestration adapter."""

from __future__ import annotations

from larrak_optimization.integrations.orchestration import *  # noqa: F403
from larrak_optimization.integrations.orchestration import (
    CasadiSolverAdapter,
    SimpleSolverAdapter,
)

__all__ = ["CasadiSolverAdapter", "SimpleSolverAdapter"]
