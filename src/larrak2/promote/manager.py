"""Compatibility shim for extracted promotion management."""

from __future__ import annotations

from typing import Any

from larrak_optimization.promote import manager as _impl


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)
