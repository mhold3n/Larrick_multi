"""Compatibility shim for the extracted principles search pipeline."""

from __future__ import annotations

from typing import Any

from larrak_optimization.pipelines import principles_search as _impl


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)
