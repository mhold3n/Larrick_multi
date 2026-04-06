"""Compatibility shim for extracted promotion archive helpers."""

from __future__ import annotations

from typing import Any

from larrak_optimization.promote import archive as _impl


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)
