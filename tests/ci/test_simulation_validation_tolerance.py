"""Unit tests — tolerance resolution from dataset records, not hardcoded constants."""

from __future__ import annotations

from larrak2.simulation_validation.models import (
    ComparisonMode,
    SourceType,
    ValidationMetricSpec,
)
from larrak2.simulation_validation.runners.base import evaluate_metric


class TestToleranceFromDatasetRecords:
    """Verify that tolerance bands are resolved from metric specs, not hardcoded."""

    def test_absolute_tolerance_from_spec(self):
        spec = ValidationMetricSpec(
            metric_id="ignition_delay_800K",
            units="ms",
            comparison_mode=ComparisonMode.ABSOLUTE,
            tolerance_band=0.5,
            source_type=SourceType.MEASURED,
        )
        result = evaluate_metric(spec, measured=2.0, simulated=2.3)
        assert result.tolerance_used == 0.5
        assert result.passed is True  # |2.3 - 2.0| = 0.3 <= 0.5

    def test_absolute_tolerance_fails(self):
        spec = ValidationMetricSpec(
            metric_id="ignition_delay_800K",
            units="ms",
            comparison_mode=ComparisonMode.ABSOLUTE,
            tolerance_band=0.1,
            source_type=SourceType.MEASURED,
        )
        result = evaluate_metric(spec, measured=2.0, simulated=2.3)
        assert result.tolerance_used == 0.1
        assert result.passed is False  # |2.3 - 2.0| = 0.3 > 0.1

    def test_relative_tolerance_from_spec(self):
        spec = ValidationMetricSpec(
            metric_id="flame_speed",
            units="m/s",
            comparison_mode=ComparisonMode.RELATIVE,
            tolerance_band=0.10,
            source_type=SourceType.MEASURED,
        )
        result = evaluate_metric(spec, measured=0.40, simulated=0.43)
        assert result.tolerance_used == 0.10
        # |0.43 - 0.40| / |0.40| = 0.075 <= 0.10
        assert result.passed is True

    def test_relative_tolerance_fails(self):
        spec = ValidationMetricSpec(
            metric_id="flame_speed",
            units="m/s",
            comparison_mode=ComparisonMode.RELATIVE,
            tolerance_band=0.05,
            source_type=SourceType.MEASURED,
        )
        result = evaluate_metric(spec, measured=0.40, simulated=0.43)
        # |0.43 - 0.40| / |0.40| = 0.075 > 0.05
        assert result.passed is False

    def test_band_comparison_from_spec(self):
        spec = ValidationMetricSpec(
            metric_id="ca50",
            units="deg",
            comparison_mode=ComparisonMode.BAND,
            tolerance_band=2.0,
            source_type=SourceType.MEASURED,
        )
        result = evaluate_metric(spec, measured=10.0, simulated=11.5)
        assert result.passed is True  # 1.5 <= 2.0

    def test_different_datasets_use_different_tolerances(self):
        """Each dataset defines its own tolerances, not a single global constant."""
        spec_tight = ValidationMetricSpec(
            metric_id="pressure_peak",
            units="bar",
            comparison_mode=ComparisonMode.ABSOLUTE,
            tolerance_band=0.5,
            source_type=SourceType.MEASURED,
        )
        spec_loose = ValidationMetricSpec(
            metric_id="pressure_peak",
            units="bar",
            comparison_mode=ComparisonMode.ABSOLUTE,
            tolerance_band=5.0,
            source_type=SourceType.MEASURED,
        )
        r_tight = evaluate_metric(spec_tight, measured=50.0, simulated=52.0)
        r_loose = evaluate_metric(spec_loose, measured=50.0, simulated=52.0)
        assert r_tight.passed is False  # 2.0 > 0.5
        assert r_loose.passed is True  # 2.0 <= 5.0

    def test_near_zero_measured_relative(self):
        """Relative comparison near zero uses safe denominator."""
        spec = ValidationMetricSpec(
            metric_id="residual",
            units="",
            comparison_mode=ComparisonMode.RELATIVE,
            tolerance_band=0.10,
            source_type=SourceType.DERIVED_CONSTRAINT,
        )
        result = evaluate_metric(spec, measured=1e-15, simulated=1e-14)
        # Should not divide by zero
        assert result.error >= 0
