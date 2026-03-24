"""CLI coverage for chemistry offline-cache generation."""

from __future__ import annotations

import json
from pathlib import Path

from larrak2.cli.validate_simulation import main as validate_simulation_main
from larrak2.simulation_validation.models import (
    RegimeStatus,
    SourceType,
    ValidationCaseSpec,
    ValidationMetricResult,
    ValidationRunManifest,
)


def test_chemistry_cache_cli_forces_native_cache_build(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    cache_path = tmp_path / "chemistry_cache.json"

    config = {
        "suite_id": "gas_combustion_v1",
        "regimes": {
            "chemistry": {
                "dataset": {
                    "dataset_id": "chem_fixture",
                    "source_type": "measured",
                    "fuel_family": "gasoline",
                    "metrics": [
                        {
                            "metric_id": "ignition_delay_796K_15bar_phi0p66",
                            "units": "ms",
                            "comparison_mode": "absolute",
                            "tolerance_band": 15.0,
                            "source_type": "measured",
                            "required": True,
                        }
                    ],
                },
                "case_spec": {
                    "case_id": "chemistry_suite_case",
                    "regime": "chemistry",
                    "solver_config": {
                        "simulation_adapter": {
                            "kind": "chemistry",
                            "backend": "auto",
                            "offline_results_path": str(cache_path),
                            "offline_results_only": True,
                            "mechanism_file": "Docs/Detailed gasoline surrogate/ChemDetailed.inp.txt",
                        }
                    },
                },
                "simulation_data": {
                    "ignition_delay_796K_15bar_phi0p66_measured": 109.0,
                },
            }
        },
    }
    config_path = tmp_path / "gas_suite.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def _fake_run_single_regime(regime, dataset, case_spec, simulation_data, **kwargs):
        _ = regime, dataset, simulation_data, kwargs
        captured["case_spec"] = case_spec
        return ValidationRunManifest(
            regime="chemistry",
            case_spec=ValidationCaseSpec(case_id="chemistry_suite_case", regime="chemistry"),
            metric_results=[
                ValidationMetricResult(
                    metric_id="ignition_delay_796K_15bar_phi0p66",
                    measured_value=109.0,
                    simulated_value=109.0,
                    error=0.0,
                    tolerance_used=15.0,
                    passed=True,
                    source_type=SourceType.MEASURED,
                    units="ms",
                )
            ],
            solver_artifacts={"chemistry_offline_results": str(cache_path)},
            status=RegimeStatus.PASSED,
        )

    monkeypatch.setattr(
        "larrak2.simulation_validation.suite.run_single_regime",
        _fake_run_single_regime,
    )
    monkeypatch.setattr(
        "larrak2.cli.validate_simulation._write_regime_artifacts",
        lambda **kwargs: (tmp_path / "manifest.json", tmp_path / "summary.md"),
    )

    code = validate_simulation_main(
        [
            "chemistry-cache",
            "--config",
            str(config_path),
            "--outdir",
            str(tmp_path / "artifacts"),
        ]
    )

    assert code == 0
    case_spec = captured["case_spec"]
    assert isinstance(case_spec, ValidationCaseSpec)
    adapter = case_spec.solver_config["simulation_adapter"]
    assert adapter["backend"] == "native_cantera"
    assert adapter["offline_results_only"] is False
    assert adapter["offline_results_path"] == str(cache_path)
