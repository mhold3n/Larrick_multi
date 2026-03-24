"""Core data models for the simulation-validation suite."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Source types
# ---------------------------------------------------------------------------


class SourceType(str, enum.Enum):
    """How a validation target was obtained."""

    MEASURED = "measured"
    SYNTHETIC = "synthetic"
    DERIVED_CONSTRAINT = "derived_constraint"


class ComparisonMode(str, enum.Enum):
    """How a metric is compared against the target."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    BAND = "band"


class RegimeStatus(str, enum.Enum):
    """Outcome status for a regime within the suite."""

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED_BY_PREREQUISITE = "blocked_by_prerequisite"
    NOT_RUN = "not_run"


# ---------------------------------------------------------------------------
# Metric specifications
# ---------------------------------------------------------------------------


@dataclass
class ValidationMetricSpec:
    """Definition of a single validation metric."""

    metric_id: str
    units: str
    comparison_mode: ComparisonMode
    tolerance_band: float
    source_type: SourceType
    required: bool = True
    description: str = ""


@dataclass
class ValidationMetricResult:
    """Result of evaluating a single validation metric."""

    metric_id: str
    measured_value: float
    simulated_value: float
    error: float
    tolerance_used: float
    passed: bool
    source_type: SourceType
    units: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------


@dataclass
class ValidationDatasetManifest:
    """Registry entry for a validation dataset."""

    dataset_id: str
    regime: str
    fuel_family: str
    source_type: SourceType
    provenance: dict[str, Any] = field(default_factory=dict)
    operating_bounds: dict[str, Any] = field(default_factory=dict)
    metrics: list[ValidationMetricSpec] = field(default_factory=list)
    measured_anchor_ids: list[str] = field(default_factory=list)
    governing_basis: str = ""
    literature_reference: str = ""
    standard_reference: str = ""

    def validate_provenance(self) -> list[str]:
        """Check provenance rules; return list of error messages (empty = OK)."""
        errors: list[str] = []
        if self.source_type == SourceType.SYNTHETIC:
            if not self.measured_anchor_ids:
                errors.append(
                    f"Dataset '{self.dataset_id}': synthetic target must record measured_anchor_ids"
                )
            if not self.governing_basis:
                errors.append(
                    f"Dataset '{self.dataset_id}': synthetic target must record governing_basis"
                )
        return errors


# ---------------------------------------------------------------------------
# Case spec
# ---------------------------------------------------------------------------


@dataclass
class ValidationCaseSpec:
    """Specification of a single validation run case."""

    case_id: str
    regime: str
    operating_point: dict[str, float] = field(default_factory=dict)
    geometry_revision: str = ""
    motion_profile_revision: str = ""
    solver_config: dict[str, Any] = field(default_factory=dict)
    dataset_binding: str = ""


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


@dataclass
class ValidationRunManifest:
    """Results of running validation for one regime."""

    regime: str
    case_spec: ValidationCaseSpec
    metric_results: list[ValidationMetricResult] = field(default_factory=list)
    solver_artifacts: dict[str, str] = field(default_factory=dict)
    status: RegimeStatus = RegimeStatus.NOT_RUN
    blocked_by: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == RegimeStatus.PASSED

    def compute_status(self) -> RegimeStatus:
        """Derive status from metric results."""
        if self.status == RegimeStatus.BLOCKED_BY_PREREQUISITE:
            return self.status
        required_results = [r for r in self.metric_results if True]  # all are evaluated
        if not required_results:
            return RegimeStatus.NOT_RUN
        if all(r.passed for r in required_results):
            return RegimeStatus.PASSED
        return RegimeStatus.FAILED


# ---------------------------------------------------------------------------
# Suite manifest
# ---------------------------------------------------------------------------


@dataclass
class ValidationSuiteProfile:
    """Ordered regime profile and prerequisite graph for a validation suite."""

    suite_id: str = "canonical_v1"
    regime_order: list[str] = field(default_factory=list)
    prerequisites: dict[str, list[str]] = field(default_factory=dict)
    description: str = ""

    def normalized_regime_order(self) -> list[str]:
        """Return the declared regime order or the canonical order by default."""
        from .regimes import CanonicalRegime

        return list(self.regime_order) if self.regime_order else CanonicalRegime.ordered_names()


@dataclass
class RegimeScoreboardEntry:
    """One row in the ordered regime scoreboard."""

    regime: str
    status: RegimeStatus
    n_metrics_total: int = 0
    n_metrics_passed: int = 0
    n_metrics_failed: int = 0
    blocked_by: list[str] = field(default_factory=list)
    first_failing_metric: str = ""


@dataclass
class ValidationSuiteManifest:
    """Aggregate manifest over an ordered validation-suite profile."""

    suite_id: str = "canonical_v1"
    regime_order: list[str] = field(default_factory=list)
    prerequisites: dict[str, list[str]] = field(default_factory=dict)
    regime_results: dict[str, ValidationRunManifest] = field(default_factory=dict)
    scoreboard: list[RegimeScoreboardEntry] = field(default_factory=list)
    overall_passed: bool = False
    first_blocking_regime: str = ""
    first_blocking_metric_group: str = ""
    unblock_criteria: dict[str, list[str]] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def build_scoreboard(self) -> None:
        """Populate scoreboard from regime_results."""
        entries: list[RegimeScoreboardEntry] = []
        ordered_regimes = list(self.regime_order)
        if not ordered_regimes:
            from .regimes import CanonicalRegime

            ordered_regimes = CanonicalRegime.ordered_names()

        for regime_name in ordered_regimes:
            run = self.regime_results.get(regime_name)
            if run is None:
                entries.append(
                    RegimeScoreboardEntry(
                        regime=regime_name,
                        status=RegimeStatus.NOT_RUN,
                    )
                )
                continue
            n_total = len(run.metric_results)
            n_passed = sum(1 for r in run.metric_results if r.passed)
            n_failed = n_total - n_passed
            first_fail = ""
            for r in run.metric_results:
                if not r.passed:
                    first_fail = r.metric_id
                    break
            entries.append(
                RegimeScoreboardEntry(
                    regime=regime_name,
                    status=run.status,
                    n_metrics_total=n_total,
                    n_metrics_passed=n_passed,
                    n_metrics_failed=n_failed,
                    blocked_by=list(run.blocked_by),
                    first_failing_metric=first_fail,
                )
            )

        self.scoreboard = entries
        all_passed = all(e.status == RegimeStatus.PASSED for e in entries)
        self.overall_passed = all_passed

        # Identify first blocker
        self.first_blocking_regime = ""
        self.first_blocking_metric_group = ""
        for e in entries:
            if e.status == RegimeStatus.FAILED:
                self.first_blocking_regime = e.regime
                self.first_blocking_metric_group = e.first_failing_metric
                break
