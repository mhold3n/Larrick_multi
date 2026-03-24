"""Base class for regime runners and shared metric evaluation."""

from __future__ import annotations

import abc
from typing import Any

from ..models import (
    ComparisonMode,
    RegimeStatus,
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationMetricResult,
    ValidationMetricSpec,
    ValidationRunManifest,
)


def evaluate_metric(
    spec: ValidationMetricSpec,
    measured: float,
    simulated: float,
) -> ValidationMetricResult:
    """Evaluate a single metric against its tolerance band."""
    if spec.comparison_mode == ComparisonMode.RELATIVE:
        denom = abs(measured) if abs(measured) > 1e-12 else 1e-12
        error = abs(simulated - measured) / denom
    elif spec.comparison_mode == ComparisonMode.BAND:
        error = abs(simulated - measured)
    else:  # ABSOLUTE
        error = abs(simulated - measured)

    passed = error <= spec.tolerance_band

    return ValidationMetricResult(
        metric_id=spec.metric_id,
        measured_value=measured,
        simulated_value=simulated,
        error=error,
        tolerance_used=spec.tolerance_band,
        passed=passed,
        source_type=spec.source_type,
        units=spec.units,
    )


class BaseRegimeRunner(abc.ABC):
    """Abstract base for regime-specific validation runners."""

    @property
    @abc.abstractmethod
    def regime_name(self) -> str:
        """Return canonical regime name string."""

    @abc.abstractmethod
    def run(
        self,
        dataset: ValidationDatasetManifest,
        case_spec: ValidationCaseSpec,
        simulation_data: dict[str, Any],
    ) -> ValidationRunManifest:
        """Execute validation for this regime."""

    def _build_run_manifest(
        self,
        case_spec: ValidationCaseSpec,
        metric_results: list[ValidationMetricResult],
        solver_artifacts: dict[str, str] | None = None,
        messages: list[str] | None = None,
    ) -> ValidationRunManifest:
        """Helper to create a run manifest from metric results."""
        all_passed = all(r.passed for r in metric_results)
        status = RegimeStatus.PASSED if all_passed else RegimeStatus.FAILED
        if not metric_results:
            status = RegimeStatus.NOT_RUN

        return ValidationRunManifest(
            regime=self.regime_name,
            case_spec=case_spec,
            metric_results=metric_results,
            solver_artifacts=solver_artifacts or {},
            status=status,
            messages=messages or [],
        )
