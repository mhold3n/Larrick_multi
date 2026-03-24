"""Canonical-regime simulation validation suite.

Replaces the monolithic thermo-anchor gate with a five-regime validation
ladder: chemistry → spray → reacting_flow → closed_cylinder → full_handoff.
"""

from __future__ import annotations

from .models import (
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationMetricResult,
    ValidationMetricSpec,
    ValidationRunManifest,
    ValidationSuiteManifest,
    ValidationSuiteProfile,
)
from .regimes import CanonicalRegime

__all__ = [
    "CanonicalRegime",
    "ValidationCaseSpec",
    "ValidationDatasetManifest",
    "ValidationMetricResult",
    "ValidationMetricSpec",
    "ValidationRunManifest",
    "ValidationSuiteProfile",
    "ValidationSuiteManifest",
]
