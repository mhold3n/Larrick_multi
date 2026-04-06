"""Compatibility shim for the extracted optimization Pareto CLI."""

from __future__ import annotations

from typing import Any

from larrak_optimization.cli import run_pareto as _impl

main = _impl.main


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


__all__ = ["main"]
