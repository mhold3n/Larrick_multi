"""Optimization utilities migrated from legacy monolith."""

from __future__ import annotations

from .candidate_store import CandidateEntry, CandidateStore
from .production_gate import (
    BALANCED_THRESHOLD_PROFILE,
    STRICT_PRODUCTION_PROFILE,
    evaluate_production_gate,
    required_pareto_min,
)

__all__ = [
    "BALANCED_THRESHOLD_PROFILE",
    "CandidateEntry",
    "CandidateStore",
    "STRICT_PRODUCTION_PROFILE",
    "evaluate_production_gate",
    "required_pareto_min",
]
