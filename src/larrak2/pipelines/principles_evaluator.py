"""Compatibility shim for the extracted principles evaluator."""

from __future__ import annotations

from typing import Any

from larrak_optimization.pipelines import principles_evaluator as _impl


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)
