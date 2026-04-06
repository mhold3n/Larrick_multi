"""Diagnostics utilities for IPOPT solves."""

from __future__ import annotations

from typing import Any


def summarize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Extract stable diagnostics fields from CasADi solver stats."""
    return {
        "success": bool(stats.get("success", False)),
        "return_status": str(stats.get("return_status", "")),
        "iter_count": int(stats.get("iter_count", 0) or 0),
        "t_proc_total": float(stats.get("t_proc_total", 0.0) or 0.0),
    }
