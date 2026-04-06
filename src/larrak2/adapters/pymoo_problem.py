"""Compatibility shim for the extracted optimization pymoo adapter."""

from __future__ import annotations

from typing import Any

from larrak_optimization.adapters import pymoo_problem as _impl

ParetoProblem = _impl.ParetoProblem
create_problem = _impl.create_problem


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


__all__ = ["ParetoProblem", "create_problem"]
