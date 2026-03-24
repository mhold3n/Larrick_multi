"""Integration test — spray runner against fixture ECN-style cases."""

from __future__ import annotations

from larrak2.simulation_validation.models import (
    ComparisonMode,
    RegimeStatus,
    SourceType,
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationMetricSpec,
)
from larrak2.simulation_validation.runners.spray import SprayRunner


def _ecn_spray_g_dataset() -> ValidationDatasetManifest:
    return ValidationDatasetManifest(
        dataset_id="ecn_spray_g",
        regime="spray",
        fuel_family="gasoline",
        source_type=SourceType.MEASURED,
        provenance={"source": "ECN", "injector": "Spray G"},
        operating_bounds={"P_amb_bar": 6, "T_amb_K": 573},
        metrics=[
            ValidationMetricSpec(
                metric_id="liquid_penetration_mm",
                units="mm",
                comparison_mode=ComparisonMode.ABSOLUTE,
                tolerance_band=2.0,
                source_type=SourceType.MEASURED,
            ),
            ValidationMetricSpec(
                metric_id="vapor_penetration_mm",
                units="mm",
                comparison_mode=ComparisonMode.ABSOLUTE,
                tolerance_band=5.0,
                source_type=SourceType.MEASURED,
            ),
            ValidationMetricSpec(
                metric_id="droplet_smd_um",
                units="um",
                comparison_mode=ComparisonMode.RELATIVE,
                tolerance_band=0.15,
                source_type=SourceType.MEASURED,
            ),
            ValidationMetricSpec(
                metric_id="gas_velocity_m_s",
                units="m/s",
                comparison_mode=ComparisonMode.RELATIVE,
                tolerance_band=0.20,
                source_type=SourceType.MEASURED,
                required=False,
            ),
        ],
    )


class TestSprayRunnerIntegration:
    def test_all_pass(self):
        runner = SprayRunner()
        ds = _ecn_spray_g_dataset()
        case = ValidationCaseSpec(case_id="spray_pass", regime="spray")
        sim = {
            "liquid_penetration_mm": 18.5,
            "liquid_penetration_mm_measured": 18.0,
            "vapor_penetration_mm": 52.0,
            "vapor_penetration_mm_measured": 50.0,
            "droplet_smd_um": 12.5,
            "droplet_smd_um_measured": 12.0,
            "gas_velocity_m_s": 25.0,
            "gas_velocity_m_s_measured": 24.0,
        }
        run = runner.run(ds, case, sim)
        assert run.status == RegimeStatus.PASSED

    def test_optional_metric_skipped(self):
        runner = SprayRunner()
        ds = _ecn_spray_g_dataset()
        case = ValidationCaseSpec(case_id="spray_optional", regime="spray")
        sim = {
            "liquid_penetration_mm": 18.5,
            "liquid_penetration_mm_measured": 18.0,
            "vapor_penetration_mm": 52.0,
            "vapor_penetration_mm_measured": 50.0,
            "droplet_smd_um": 12.5,
            "droplet_smd_um_measured": 12.0,
            # gas_velocity not provided — optional
        }
        run = runner.run(ds, case, sim)
        assert run.status == RegimeStatus.PASSED
        assert any("optional" in m.lower() for m in run.messages)

    def test_acceptance_outputs(self):
        runner = SprayRunner()
        ds = _ecn_spray_g_dataset()
        case = ValidationCaseSpec(case_id="spray_outputs", regime="spray")
        sim = {
            "liquid_penetration_mm": 18.5,
            "liquid_penetration_mm_measured": 18.0,
            "vapor_penetration_mm": 52.0,
            "vapor_penetration_mm_measured": 50.0,
            "droplet_smd_um": 12.5,
            "droplet_smd_um_measured": 12.0,
            "spray_provenance": {"injector": "Spray G", "institution": "SNL"},
        }
        run = runner.run(ds, case, sim)
        outputs = runner.build_acceptance_outputs(run)
        assert "liquid_penetration" in outputs
        assert "vapor_penetration" in outputs
        assert "droplet_comparisons" in outputs
