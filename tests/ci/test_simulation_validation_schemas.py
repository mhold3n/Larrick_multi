"""Unit tests — schema validation for all validation manifest and dataset record types."""

from __future__ import annotations

import pytest

from larrak2.simulation_validation.models import (
    ComparisonMode,
    RegimeStatus,
    SourceType,
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationMetricResult,
    ValidationMetricSpec,
    ValidationRunManifest,
    ValidationSuiteManifest,
)
from larrak2.simulation_validation.regimes import CanonicalRegime

# ---------------------------------------------------------------------------
# CanonicalRegime
# ---------------------------------------------------------------------------


class TestCanonicalRegime:
    def test_ordered_returns_five(self):
        ordered = CanonicalRegime.ordered()
        assert len(ordered) == 5
        assert ordered[0] == CanonicalRegime.CHEMISTRY
        assert ordered[-1] == CanonicalRegime.FULL_HANDOFF

    def test_regime_values(self):
        assert CanonicalRegime.CHEMISTRY.value == "chemistry"
        assert CanonicalRegime.SPRAY.value == "spray"
        assert CanonicalRegime.REACTING_FLOW.value == "reacting_flow"
        assert CanonicalRegime.CLOSED_CYLINDER.value == "closed_cylinder"
        assert CanonicalRegime.FULL_HANDOFF.value == "full_handoff"


# ---------------------------------------------------------------------------
# ValidationMetricSpec
# ---------------------------------------------------------------------------


class TestValidationMetricSpec:
    def test_required_fields(self):
        spec = ValidationMetricSpec(
            metric_id="ign_delay_800K",
            units="ms",
            comparison_mode=ComparisonMode.ABSOLUTE,
            tolerance_band=0.5,
            source_type=SourceType.MEASURED,
        )
        assert spec.metric_id == "ign_delay_800K"
        assert spec.required is True  # default

    def test_optional_flag(self):
        spec = ValidationMetricSpec(
            metric_id="gas_velocity",
            units="m/s",
            comparison_mode=ComparisonMode.RELATIVE,
            tolerance_band=0.10,
            source_type=SourceType.MEASURED,
            required=False,
        )
        assert spec.required is False


# ---------------------------------------------------------------------------
# ValidationMetricResult
# ---------------------------------------------------------------------------


class TestValidationMetricResult:
    def test_passed_result(self):
        r = ValidationMetricResult(
            metric_id="flame_speed",
            measured_value=0.35,
            simulated_value=0.34,
            error=0.01,
            tolerance_used=0.05,
            passed=True,
            source_type=SourceType.MEASURED,
            units="m/s",
        )
        assert r.passed is True
        assert r.error == pytest.approx(0.01)

    def test_failed_result(self):
        r = ValidationMetricResult(
            metric_id="flame_speed",
            measured_value=0.35,
            simulated_value=0.50,
            error=0.15,
            tolerance_used=0.05,
            passed=False,
            source_type=SourceType.MEASURED,
        )
        assert r.passed is False


# ---------------------------------------------------------------------------
# ValidationDatasetManifest
# ---------------------------------------------------------------------------


class TestValidationDatasetManifest:
    def test_measured_passes_provenance(self):
        ds = ValidationDatasetManifest(
            dataset_id="shock_tube_trf",
            regime="chemistry",
            fuel_family="gasoline",
            source_type=SourceType.MEASURED,
        )
        assert ds.validate_provenance() == []

    def test_synthetic_without_anchors_fails(self):
        ds = ValidationDatasetManifest(
            dataset_id="synthetic_ign",
            regime="chemistry",
            fuel_family="gasoline",
            source_type=SourceType.SYNTHETIC,
            measured_anchor_ids=[],
            governing_basis="",
        )
        errors = ds.validate_provenance()
        assert len(errors) == 2
        assert "measured_anchor_ids" in errors[0]
        assert "governing_basis" in errors[1]

    def test_synthetic_with_anchors_passes(self):
        ds = ValidationDatasetManifest(
            dataset_id="synthetic_ign",
            regime="chemistry",
            fuel_family="gasoline",
            source_type=SourceType.SYNTHETIC,
            measured_anchor_ids=["shock_tube_trf"],
            governing_basis="Arrhenius rate extrapolation",
        )
        assert ds.validate_provenance() == []


# ---------------------------------------------------------------------------
# ValidationCaseSpec
# ---------------------------------------------------------------------------


class TestValidationCaseSpec:
    def test_basic_construction(self):
        cs = ValidationCaseSpec(
            case_id="chem_case_1",
            regime="chemistry",
            operating_point={"T": 800.0, "P_bar": 20.0},
        )
        assert cs.regime == "chemistry"
        assert cs.operating_point["T"] == 800.0


# ---------------------------------------------------------------------------
# ValidationRunManifest
# ---------------------------------------------------------------------------


class TestValidationRunManifest:
    def test_passed_status(self):
        run = ValidationRunManifest(
            regime="chemistry",
            case_spec=ValidationCaseSpec(case_id="c1", regime="chemistry"),
            metric_results=[
                ValidationMetricResult(
                    metric_id="m1",
                    measured_value=1.0,
                    simulated_value=1.0,
                    error=0.0,
                    tolerance_used=0.1,
                    passed=True,
                    source_type=SourceType.MEASURED,
                ),
            ],
            status=RegimeStatus.PASSED,
        )
        assert run.passed is True

    def test_blocked_status(self):
        run = ValidationRunManifest(
            regime="reacting_flow",
            case_spec=ValidationCaseSpec(case_id="rf1", regime="reacting_flow"),
            status=RegimeStatus.BLOCKED_BY_PREREQUISITE,
            blocked_by=["chemistry"],
        )
        assert run.passed is False
        assert run.blocked_by == ["chemistry"]

    def test_compute_status_from_metrics(self):
        run = ValidationRunManifest(
            regime="spray",
            case_spec=ValidationCaseSpec(case_id="s1", regime="spray"),
            metric_results=[
                ValidationMetricResult(
                    metric_id="m1",
                    measured_value=1.0,
                    simulated_value=1.0,
                    error=0.0,
                    tolerance_used=0.1,
                    passed=True,
                    source_type=SourceType.MEASURED,
                ),
                ValidationMetricResult(
                    metric_id="m2",
                    measured_value=1.0,
                    simulated_value=2.0,
                    error=1.0,
                    tolerance_used=0.1,
                    passed=False,
                    source_type=SourceType.MEASURED,
                ),
            ],
        )
        assert run.compute_status() == RegimeStatus.FAILED


# ---------------------------------------------------------------------------
# ValidationSuiteManifest
# ---------------------------------------------------------------------------


class TestValidationSuiteManifest:
    def test_build_scoreboard_all_passed(self):
        suite = ValidationSuiteManifest()
        for regime in CanonicalRegime.ordered():
            suite.regime_results[regime.value] = ValidationRunManifest(
                regime=regime.value,
                case_spec=ValidationCaseSpec(case_id=f"{regime.value}_c1", regime=regime.value),
                metric_results=[
                    ValidationMetricResult(
                        metric_id=f"{regime.value}_m1",
                        measured_value=1.0,
                        simulated_value=1.0,
                        error=0.0,
                        tolerance_used=0.1,
                        passed=True,
                        source_type=SourceType.MEASURED,
                    ),
                ],
                status=RegimeStatus.PASSED,
            )
        suite.build_scoreboard()
        assert suite.overall_passed is True
        assert len(suite.scoreboard) == 5
        assert suite.first_blocking_regime == ""

    def test_build_scoreboard_with_failure(self):
        suite = ValidationSuiteManifest()
        suite.regime_results["chemistry"] = ValidationRunManifest(
            regime="chemistry",
            case_spec=ValidationCaseSpec(case_id="c1", regime="chemistry"),
            metric_results=[
                ValidationMetricResult(
                    metric_id="ign_delay",
                    measured_value=1.0,
                    simulated_value=2.0,
                    error=1.0,
                    tolerance_used=0.1,
                    passed=False,
                    source_type=SourceType.MEASURED,
                ),
            ],
            status=RegimeStatus.FAILED,
        )
        suite.build_scoreboard()
        assert suite.overall_passed is False
        assert suite.first_blocking_regime == "chemistry"
        assert suite.first_blocking_metric_group == "ign_delay"
