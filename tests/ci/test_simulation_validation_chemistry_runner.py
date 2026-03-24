"""Integration test — chemistry runner against fixture datasets."""

from __future__ import annotations

from larrak2.simulation_validation.models import (
    ComparisonMode,
    RegimeStatus,
    SourceType,
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationMetricSpec,
)
from larrak2.simulation_validation.runners.chemistry import ChemistryRunner


def _chemistry_dataset() -> ValidationDatasetManifest:
    return ValidationDatasetManifest(
        dataset_id="shock_tube_trf_gasoline",
        regime="chemistry",
        fuel_family="gasoline",
        source_type=SourceType.MEASURED,
        provenance={"source": "shock_tube", "facility": "test_lab"},
        operating_bounds={"T_min": 700, "T_max": 1400, "P_bar_min": 10, "P_bar_max": 50},
        metrics=[
            ValidationMetricSpec(
                metric_id="ignition_delay_800K",
                units="ms",
                comparison_mode=ComparisonMode.ABSOLUTE,
                tolerance_band=0.5,
                source_type=SourceType.MEASURED,
            ),
            ValidationMetricSpec(
                metric_id="ignition_delay_1000K",
                units="ms",
                comparison_mode=ComparisonMode.ABSOLUTE,
                tolerance_band=0.3,
                source_type=SourceType.MEASURED,
            ),
            ValidationMetricSpec(
                metric_id="species_CO2_peak",
                units="mol_frac",
                comparison_mode=ComparisonMode.RELATIVE,
                tolerance_band=0.10,
                source_type=SourceType.MEASURED,
            ),
            ValidationMetricSpec(
                metric_id="flame_speed_phi10",
                units="m/s",
                comparison_mode=ComparisonMode.ABSOLUTE,
                tolerance_band=0.05,
                source_type=SourceType.MEASURED,
            ),
        ],
    )


class TestChemistryRunnerIntegration:
    def test_all_metrics_pass(self):
        runner = ChemistryRunner()
        dataset = _chemistry_dataset()
        case_spec = ValidationCaseSpec(
            case_id="chem_case_pass",
            regime="chemistry",
            operating_point={"T": 800, "P_bar": 20, "phi": 1.0},
        )
        sim = {
            "ignition_delay_800K": 2.1,
            "ignition_delay_800K_measured": 2.0,
            "ignition_delay_1000K": 0.8,
            "ignition_delay_1000K_measured": 0.9,
            "species_CO2_peak": 0.12,
            "species_CO2_peak_measured": 0.11,
            "flame_speed_phi10": 0.38,
            "flame_speed_phi10_measured": 0.36,
            "mechanism_provenance": {"name": "TRF_v2", "n_species": 48},
        }
        run = runner.run(dataset, case_spec, sim)
        assert run.status == RegimeStatus.PASSED
        assert len(run.metric_results) == 4
        assert all(r.passed for r in run.metric_results)

    def test_metric_failure(self):
        runner = ChemistryRunner()
        dataset = _chemistry_dataset()
        case_spec = ValidationCaseSpec(
            case_id="chem_case_fail",
            regime="chemistry",
        )
        sim = {
            "ignition_delay_800K": 5.0,
            "ignition_delay_800K_measured": 2.0,
            "ignition_delay_1000K": 0.8,
            "ignition_delay_1000K_measured": 0.9,
            "species_CO2_peak": 0.12,
            "species_CO2_peak_measured": 0.11,
            "flame_speed_phi10": 0.36,
            "flame_speed_phi10_measured": 0.36,
        }
        run = runner.run(dataset, case_spec, sim)
        assert run.status == RegimeStatus.FAILED
        failed = [r for r in run.metric_results if not r.passed]
        assert len(failed) >= 1
        assert failed[0].metric_id == "ignition_delay_800K"

    def test_missing_metric_data(self):
        runner = ChemistryRunner()
        dataset = _chemistry_dataset()
        case_spec = ValidationCaseSpec(case_id="chem_incomplete", regime="chemistry")
        sim = {
            "ignition_delay_800K": 2.1,
            "ignition_delay_800K_measured": 2.0,
            # missing others
        }
        run = runner.run(dataset, case_spec, sim)
        assert run.status == RegimeStatus.FAILED
        assert any("Missing" in m for m in run.messages)

    def test_acceptance_outputs(self):
        runner = ChemistryRunner()
        dataset = _chemistry_dataset()
        case_spec = ValidationCaseSpec(case_id="chem_outputs", regime="chemistry")
        sim = {
            "ignition_delay_800K": 2.1,
            "ignition_delay_800K_measured": 2.0,
            "ignition_delay_1000K": 0.85,
            "ignition_delay_1000K_measured": 0.9,
            "species_CO2_peak": 0.115,
            "species_CO2_peak_measured": 0.11,
            "flame_speed_phi10": 0.37,
            "flame_speed_phi10_measured": 0.36,
            "mechanism_provenance": {"name": "TRF_v2"},
        }
        run = runner.run(dataset, case_spec, sim)
        outputs = runner.build_acceptance_outputs(run)
        assert "ignition_delay_error_table" in outputs
        assert "species_profile_comparisons" in outputs
        assert "flame_speed_comparisons" in outputs
        assert "mechanism_provenance" in outputs
