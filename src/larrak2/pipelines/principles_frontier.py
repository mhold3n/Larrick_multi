"""Compatibility shim for the extracted principles frontier pipeline."""

from __future__ import annotations

from typing import Any

from larrak_optimization.pipelines import principles_frontier as _impl

PrinciplesFrontierResult = _impl.PrinciplesFrontierResult
load_principles_profile = _impl.load_principles_profile
synthesize_principles_frontier = _impl.synthesize_principles_frontier


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


__all__ = [
    "PrinciplesFrontierResult",
    "load_principles_profile",
    "synthesize_principles_frontier",
]
