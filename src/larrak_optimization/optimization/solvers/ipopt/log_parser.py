"""Helpers to parse IPOPT status text."""

from __future__ import annotations


def is_success_status(status: str) -> bool:
    """Return True for common IPOPT success statuses."""
    s = (status or "").lower()
    good = (
        "solve_succeeded",
        "solved_to_acceptable_level",
        "optimal solution found",
        "success",
    )
    return any(token in s for token in good)
