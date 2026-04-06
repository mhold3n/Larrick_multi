"""Iteration callback stubs for compatibility with legacy interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IPOPTIterationCallback:
    """No-op callback placeholder used by downstream code."""

    step: int = 1

    def update_bounds(self, *_args, **_kwargs) -> None:
        return
