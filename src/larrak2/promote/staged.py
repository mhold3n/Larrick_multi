"""Compatibility shim for extracted staged Pareto workflow."""

from __future__ import annotations

from typing import Any

from larrak_optimization.promote import staged as _impl

StagedWorkflow = _impl.StagedWorkflow


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


__all__ = ["StagedWorkflow"]
