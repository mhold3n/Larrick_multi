from __future__ import annotations

import json
from pathlib import Path

from larrak2.simulation_validation.engine_calibration import (
    build_engine_calibration_report,
    compare_engine_calibration_reports,
    derive_cantera_preignition_handoff_bundle,
    derive_two_zone_cantera_preignition_handoff_bundle,
    propose_engine_tuning_params,
    propose_preignition_handoff_bundle,
)


def test_build_engine_calibration_report_reads_targets_and_engine_state(tmp_path: Path) -> None:
    suite_config = tmp_path / "suite.json"
    suite_config.write_text(
        json.dumps(
            {
                "suite_id": "gas_combustion_v1",
                "regimes": {
                    "spray": {
                        "simulation_data": {
                            "liquid_penetration_max_mm_sprayG_measured": 30.0,
                            "vapor_spreading_angle_deg_sprayG_measured": 90.0,
                            "droplet_smd_um_sprayG_z15mm_measured": 11.0,
                        }
                    },
                    "reacting_flow": {
                        "case_spec": {"operating_point": {"pressure_bar": 15.0}},
                        "simulation_data": {
                            "gas_temperature_K_iso_octane_reacting_measured": 1810.0,
                            "bulk_velocity_m_s_iso_octane_reacting_measured": 44.0,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    engine_dir = tmp_path / "engine_case"
    engine_dir.mkdir()
    (engine_dir / "logSummary.0.010.dat").write_text(
        "\n".join(
            [
                "# crankAngleDeg meanPressurePa meanTemperatureK meanVelocityMagnitude",
                "-72.0 120000.0 550.0 12.0",
            ]
        ),
        encoding="utf-8",
    )
    (engine_dir / "openfoam_metrics.json").write_text(
        json.dumps({"residual_fraction": 0.2, "trapped_mass": 4.1e-4}),
        encoding="utf-8",
    )

    report = build_engine_calibration_report(
        suite_config_path=suite_config,
        engine_case_dir=engine_dir,
    )

    assert report["suite_id"] == "gas_combustion_v1"
    assert report["spray_targets"]["liquid_penetration_max_mm_sprayG"] == 30.0
    assert report["reacting_targets"]["gas_temperature_K_iso_octane_reacting"] == 1810.0
    assert report["engine_latest_summary"]["mean_temperature_K"] == 550.0
    assert report["recommended_engine_seed_updates"]["handoff_velocity_m_s"] == 44.0
    assert report["recommended_engine_seed_updates"]["mixture_homogeneity_index"] == 0.75
    assert report["gap_report"]["mean_temperature_to_reacting_target_K"] == -1260.0


def test_propose_engine_tuning_params_moves_baseline_toward_targets() -> None:
    report = {
        "engine_latest_summary": {
            "mean_pressure_Pa": 112954.0,
            "mean_temperature_K": 377.847,
            "mean_velocity_magnitude_m_s": 584.407,
        },
        "engine_metrics": {
            "residual_fraction": 0.0,
        },
        "reacting_targets": {
            "gas_temperature_K_iso_octane_reacting": 1810.0,
        },
        "recommended_engine_seed_updates": {
            "handoff_velocity_m_s": 44.0,
            "mixture_homogeneity_index": 0.75,
        },
    }
    baseline = {
        "p_manifold_Pa": 120000.0,
        "p_back_Pa": 101325.0,
        "T_intake_K": 300.0,
        "T_residual_K": 900.0,
        "intake_port_area_m2": 4.0e-4,
        "exhaust_port_area_m2": 4.0e-4,
        "handoff_velocity_m_s": 44.0,
    }

    tuned = propose_engine_tuning_params(
        calibration_report=report,
        baseline_params=baseline,
    )

    assert tuned["p_manifold_Pa"] > baseline["p_manifold_Pa"]
    assert tuned["p_back_Pa"] > tuned["p_manifold_Pa"]
    assert tuned["T_residual_K"] >= baseline["T_residual_K"]
    assert tuned["T_residual_K"] <= 1200.0
    assert tuned["intake_port_area_m2"] > baseline["intake_port_area_m2"]
    assert tuned["exhaust_port_area_m2"] > baseline["exhaust_port_area_m2"]
    assert tuned["handoff_velocity_m_s"] < baseline["handoff_velocity_m_s"]
    assert tuned["residual_fraction_seed"] >= 0.1
    assert tuned["intake_open_deg"] == -125.0
    assert tuned["exhaust_close_deg"] == 85.0


def test_compare_engine_calibration_reports_reports_gap_shrinkage() -> None:
    baseline = {
        "gap_report": {
            "mean_temperature_to_reacting_target_K": -1400.0,
            "mean_velocity_to_reacting_target_m_s": 500.0,
        }
    }
    tuned = {
        "gap_report": {
            "mean_temperature_to_reacting_target_K": -1100.0,
            "mean_velocity_to_reacting_target_m_s": 600.0,
        }
    }

    comparison = compare_engine_calibration_reports(baseline, tuned)

    assert comparison["mean_temperature_to_reacting_target_K"] == 300.0
    assert comparison["mean_velocity_to_reacting_target_m_s"] == -100.0


def test_propose_preignition_handoff_bundle_compresses_engine_seed() -> None:
    report = {
        "engine_latest_summary": {
            "mean_pressure_Pa": 200472.0,
            "mean_temperature_K": 724.26,
        },
        "engine_metrics": {
            "mass_weighted_temperature_K": 666.10,
            "residual_fraction": 0.3076,
        },
    }
    tuned = {
        "p_manifold_Pa": 199449.6206,
        "T_intake_K": 420.0,
        "residual_fraction_seed": 0.125,
    }

    handoff = propose_preignition_handoff_bundle(
        calibration_report=report,
        tuned_params=tuned,
    )

    assert handoff["bundle_id"] == "preignition_seed_v1"
    assert handoff["stage_marker"] == "pre_ignition_closed_valve"
    assert handoff["pressure_Pa"] == 902124.0
    assert handoff["temperature_K"] == 950.0
    assert handoff["residual_fraction"] == 0.3
    assert handoff["cycle_coordinate_deg"] == -10.0


def test_derive_cantera_preignition_handoff_bundle_uses_cantera_state(
    monkeypatch,
) -> None:
    report = {
        "engine_latest_summary": {
            "mean_pressure_Pa": 200472.0,
            "mean_temperature_K": 724.26,
        },
        "engine_metrics": {
            "mass_weighted_temperature_K": 666.10,
            "residual_fraction": 0.3076,
        },
    }
    tuned = {
        "p_manifold_Pa": 199449.6206,
        "T_intake_K": 420.0,
        "residual_fraction_seed": 0.125,
    }

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_calibration._resolve_cantera_mechanism_from_package_manifest",
        lambda package_manifest_path: (
            {"package_id": "chem323_reduced_v2512", "package_hash": "hash"},
            Path("/tmp/chem323.yaml"),
        ),
    )

    class _FakeGas:
        def __init__(self) -> None:
            self._T = 666.1
            self._P = 200472.0
            self._X = {"IC8H18": 0.014, "O2": 0.205, "N2": 0.743, "CO2": 0.02, "H2O": 0.018}
            self.n_species = len(self._X)
            self.cp_mass = 1200.0

        @property
        def TPX(self):
            return self._T, self._P, self._X

        @TPX.setter
        def TPX(self, value):
            self._T, self._P, self._X = value
            self.n_species = len(self._X)

        @property
        def TP(self):
            return self._T, self._P

        @TP.setter
        def TP(self, value):
            self._T, self._P = value

        @property
        def T(self):
            return self._T

        @property
        def P(self):
            return self._P

        @property
        def s(self):
            return 1.0

        @property
        def SP(self):
            return self.s, self._P

        @SP.setter
        def SP(self, value):
            _, pressure = value
            self._P = pressure
            self._T = 950.0

        @property
        def X(self):
            return [self._X[name] for name in self.species_names]

        @X.setter
        def X(self, value):
            if isinstance(value, dict):
                self._X = dict(value)
            self.n_species = len(self._X)

        @property
        def species_names(self):
            return list(self._X.keys())

        def species_name(self, index: int) -> str:
            return self.species_names[index]

    class _FakeReactor:
        def __init__(self, gas) -> None:
            self.thermo = gas

    class _FakeNet:
        def __init__(self, reactors) -> None:
            self.reactor = reactors[0]
            self.time = 0.0

        def step(self):
            self.time += 5.0e-5
            gas = self.reactor.thermo
            gas._T += 40.0
            gas._P += 3.0e4
            gas._X["OH"] = gas._X.get("OH", 0.0) + 1.0e-4
            gas._X["CO"] = gas._X.get("CO", 0.0) + 2.0e-4

    class _FakeCt:
        def Solution(self, path: str, transport_model=None):
            return _FakeGas()

        def IdealGasReactor(self, gas):
            return _FakeReactor(gas)

        def ReactorNet(self, reactors):
            return _FakeNet(reactors)

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_calibration._load_cantera_module",
        lambda: _FakeCt(),
    )

    handoff, diagnostic = derive_cantera_preignition_handoff_bundle(
        calibration_report=report,
        tuned_params=tuned,
    )

    assert handoff["bundle_id"] == "preignition_cantera_seed_v1"
    assert handoff["pressure_Pa"] > 902000.0
    assert handoff["temperature_K"] >= 950.0
    assert "OH" in handoff["species_mole_fractions"]
    assert handoff["total_energy_J"] > 0.0
    assert diagnostic["base_pressure_Pa"] == 200472.0
    assert diagnostic["base_temperature_K"] == 666.10
    assert diagnostic["package_id"] == "chem323_reduced_v2512"
    assert diagnostic["estimated_ignition_time_s"] is not None


def test_derive_two_zone_cantera_preignition_handoff_bundle_uses_eval_candidate(
    monkeypatch,
) -> None:
    report = {
        "engine_latest_summary": {
            "mean_pressure_Pa": 200472.0,
            "mean_temperature_K": 724.26,
        },
        "engine_metrics": {
            "mass_weighted_temperature_K": 666.10,
            "residual_fraction": 0.3076,
            "trapped_mass": 0.0139,
        },
        "recommended_engine_seed_updates": {
            "handoff_velocity_m_s": 44.0,
        },
    }
    tuned = {
        "rpm": 1800.0,
        "torque": 80.0,
        "p_manifold_Pa": 199449.6206,
        "p_back_Pa": 210000.0,
        "T_intake_K": 420.0,
        "residual_fraction_seed": 0.125,
        "intake_port_area_m2": 4.5e-4,
        "exhaust_port_area_m2": 4.5e-4,
        "intake_open_deg": -125.0,
        "intake_close_deg": -85.0,
        "exhaust_open_deg": 35.0,
        "exhaust_close_deg": 85.0,
        "engine_end_angle_deg": 20.0,
    }

    class _FakeEval:
        def __init__(self) -> None:
            self.diag = {
                "thermo": {
                    "ignition_stage": {
                        "spark_absolute_deg": 352.0,
                        "ignition_delay_spark_s": 0.002,
                        "ignitability_margin": -0.5,
                    },
                    "mixture_preparation": {
                        "delivered_vapor_fraction": 0.92,
                        "mixture_homogeneity": 0.81,
                    },
                }
            }

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_calibration.evaluate_candidate",
        lambda x, ctx: _FakeEval(),
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_calibration.candidate_openfoam_params",
        lambda candidate, ctx, eval_diag=None: {
            "lambda_af": 1.0,
            "engine_end_angle_deg": 20.0,
        },
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_calibration.candidate_openfoam_handoff_bundle",
        lambda candidate, ctx, openfoam_params, eval_diag=None: {
            "bundle_id": "base_truth_seed",
            "mechanism_id": "chem323_reduced_v2512",
            "pressure_Pa": 150000.0,
            "temperature_K": 410.0,
            "species_mole_fractions": {
                "IC8H18": 0.02,
                "O2": 0.21,
                "N2": 0.73,
                "CO2": 0.02,
                "H2O": 0.02,
            },
            "cycle_coordinate_deg": -170.0,
            "residual_fraction": 0.12,
            "total_mass_kg": 4.0e-4,
        },
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_calibration._resolve_cantera_mechanism_from_package_manifest",
        lambda package_manifest_path: (
            {"package_id": "chem323_reduced_v2512", "package_hash": "hash"},
            Path("/tmp/chem323.yaml"),
        ),
    )

    class _FakeGas:
        def __init__(self) -> None:
            self._T = 1000.0
            self._P = 900000.0
            self._X = {
                "IC8H18": 0.014,
                "O2": 0.205,
                "N2": 0.743,
                "CO2": 0.02,
                "H2O": 0.018,
            }
            self.n_species = len(self._X)
            self.cp_mass = 1200.0

        @property
        def TPX(self):
            return self._T, self._P, self._X

        @TPX.setter
        def TPX(self, value):
            self._T, self._P, self._X = value
            self.n_species = len(self._X)

        @property
        def TP(self):
            return self._T, self._P

        @TP.setter
        def TP(self, value):
            self._T, self._P = value

        @property
        def T(self):
            return self._T

        @property
        def P(self):
            return self._P

        @property
        def X(self):
            return [self._X[name] for name in self.species_names]

        @X.setter
        def X(self, value):
            if isinstance(value, dict):
                self._X = dict(value)
            self.n_species = len(self._X)

        @property
        def species_names(self):
            return list(self._X.keys())

        def species_name(self, index: int) -> str:
            return self.species_names[index]

    class _FakeReactor:
        def __init__(self, gas) -> None:
            self.thermo = gas

    class _FakeNet:
        def __init__(self, reactors) -> None:
            self.reactor = reactors[0]
            self.time = 0.0

        def step(self):
            self.time += 5.0e-5
            gas = self.reactor.thermo
            gas._T += 30.0
            gas._P += 2.5e4
            gas._X["OH"] = gas._X.get("OH", 0.0) + 8.0e-5

    class _FakeCt:
        def Solution(self, path: str, transport_model=None):
            return _FakeGas()

        def IdealGasReactor(self, gas):
            return _FakeReactor(gas)

        def ReactorNet(self, reactors):
            return _FakeNet(reactors)

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_calibration._load_cantera_module",
        lambda: _FakeCt(),
    )

    handoff, diagnostic = derive_two_zone_cantera_preignition_handoff_bundle(
        candidate_x=[60.0] * 27,
        calibration_report=report,
        tuned_params=tuned,
    )

    assert handoff["bundle_id"] == "preignition_two_zone_cantera_seed_v1"
    assert handoff["cycle_coordinate_deg"] == -10.0
    assert handoff["stage_marker"] == "pre_ignition_two_zone_cantera"
    assert handoff["pressure_Pa"] > 9.0e5
    assert handoff["temperature_K"] > 1000.0
    assert handoff["velocity_m_s"] == 0.0
    assert handoff["total_mass_kg"] == 0.0139
    assert set(handoff["species_mole_fractions"]) <= {"IC8H18", "O2", "N2", "CO2", "H2O"}
    assert "OH" in diagnostic["full_cantera_species_mole_fractions"]
    assert diagnostic["diagnostic_seed"]["seed_origin"] == "two_zone_ignition_stage_v1"
    assert diagnostic["diagnostic_seed"]["candidate_id"] == "truth_00"
    assert diagnostic["initial_temperature_boost_K"] >= 90.0
