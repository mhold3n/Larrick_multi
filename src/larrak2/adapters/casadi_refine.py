"""Compatibility shim for the extracted optimization CasADi refinement adapter."""

from __future__ import annotations

from typing import Any

from larrak_optimization.adapters import casadi_refine as _impl

RefinementMode = _impl.RefinementMode
RefinementResult = _impl.RefinementResult
refine_candidate = _impl.refine_candidate


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


__all__ = ["RefinementMode", "RefinementResult", "refine_candidate"]
