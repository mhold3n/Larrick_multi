from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from larrak2.simulation_validation.engine_restart_benchmark import (
    benchmark_engine_restart_profiles,
)
from larrak2.simulation_validation.tuning_characterization_study import (
    TUNING_MANIFEST_BASENAME,
    load_experiments_jsonl,
)


def _write_package(package_dir: Path, *, package_id: str, package_hash: str) -> None:
    package_dir.mkdir(parents=True)
    (package_dir / "reactions").write_text("reactions\n", encoding="utf-8")
    (package_dir / "thermo.compressibleGas").write_text("thermo\n", encoding="utf-8")
    (package_dir / "transportProperties").write_text("transport\n", encoding="utf-8")
    (package_dir / "package_manifest.json").write_text(
        json.dumps(
            {
                "package_id": package_id,
                "package_hash": package_hash,
                "species_count": 312,
                "reaction_count": 2469,
                "fuel_species": "IC8H18",
            }
        ),
        encoding="utf-8",
    )


def _write_runtime_table(table_dir: Path, *, table_id: str, table_hash: str) -> None:
    table_dir.mkdir(parents=True)
    (table_dir / "runtimeChemistryTable").write_text("table\n", encoding="utf-8")
    (table_dir / "runtime_chemistry_table_manifest.json").write_text(
        json.dumps(
            {
                "table_id": table_id,
                "generated_file_hashes": {"runtimeChemistryTable": table_hash},
            }
        ),
        encoding="utf-8",
    )


def _write_log_summary(
    run_dir: Path,
    *,
    time_s: float,
    crank_angle_deg: float,
    mean_pressure_Pa: float,
    mean_temperature_K: float,
    mean_velocity_magnitude_m_s: float,
) -> None:
    (run_dir / f"logSummary.{time_s}.dat").write_text(
        "\n".join(
            [
                "# crankAngleDeg meanPressurePa meanTemperatureK meanVelocityMagnitude",
                f"{crank_angle_deg}\t{mean_pressure_Pa}\t{mean_temperature_K}\t{mean_velocity_magnitude_m_s}",
            ]
        ),
        encoding="utf-8",
    )


def test_benchmark_engine_restart_profiles_clones_checkpoint_and_scores_profiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_package = tmp_path / "chem323_reduced"
    _write_package(runtime_package, package_id="chem323_reduced_v2512", package_hash="runtime-hash")

    strategy_path = tmp_path / "strategy.json"
    strategy_path.write_text(
        json.dumps(
            {
                "strategy_id": "engine_runtime_mechanism_ladder_v1",
                "runtime_package": {
                    "label": "chem323_runtime",
                    "package_dir": str(runtime_package),
                },
                "checkpoint_packages": [
                    {
                        "label": "chem323_checkpoint_reference",
                        "package_dir": str(runtime_package),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tuned_params_path = tmp_path / "tuned_params.json"
    tuned_params_path.write_text(
        json.dumps({"rpm": 1800.0, "torque": 80.0}),
        encoding="utf-8",
    )
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps({"handoff_bundle": {"cycle_coordinate_deg": -10.0}}),
        encoding="utf-8",
    )

    run_dir = tmp_path / "engine_run"
    for name in ("0", "constant", "system", "chemistry"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "constant" / "engineGeometry").write_text(
        "\n".join(
            [
                "cycleEndAngleDeg 0;",
                "minTemperatureK 300;",
                "maxTemperatureK 1700;",
                "maxThermoDeltaTK 8;",
                "minPressurePa 25000;",
                "minDensityKgM3 0.08;",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "system" / "controlDict").write_text(
        "\n".join(
            [
                "startFrom latestTime;",
                "endTime 0.0003;",
                "deltaT 1e-6;",
                "maxCo 0.02;",
                "maxDeltaT 1e-6;",
                "writeInterval 1;",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "constant" / "chemistryProperties").write_text(
        "\n".join(
            [
                "chemistry on;",
                "initialChemicalTimeStep 1e-8;",
                "odeCoeffs",
                "{",
                "    absTol 1e-14;",
                "    relTol 1e-9;",
                "}",
                "reduction",
                "{",
                "    active on;",
                "}",
                "tabulation",
                "{",
                "    active off;",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "constant" / "combustionProperties").write_text(
        "\n".join(
            [
                "active yes;",
                "PaSRCoeffs",
                "{",
                "    Cmix 0.01;",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "engine_stage_manifest.json").write_text(
        json.dumps(
            {
                "profile": "closed_valve_ignition_v1",
                "stages": [
                    {"name": "settle_flow", "ok": True, "end_angle_deg": -9.25},
                    {"name": "chemistry_seed", "ok": True, "end_angle_deg": -8.5},
                    {"name": "chemistry_spinup", "ok": True, "end_angle_deg": -7.0},
                    {"name": "ignition_release", "ok": False, "end_angle_deg": -4.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "engine_stage_resume_summary.json").write_text(
        json.dumps(
            {
                "remaining_stages": ["ignition_release", "early_burn"],
                "current_stage": "ignition_release",
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "0.1000").mkdir()
    _write_log_summary(
        run_dir,
        time_s=0.1,
        crank_angle_deg=-6.98,
        mean_pressure_Pa=9.30e5,
        mean_temperature_K=1015.0,
        mean_velocity_magnitude_m_s=1120.0,
    )

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline._ensure_custom_solver",
        lambda self, log_file=None: {},
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline.docker_preflight",
        lambda self, log_file=None: {
            "ok": True,
            "docker_bin": "/usr/local/bin/docker",
            "failure_class": "",
            "message": "",
            "candidate_paths": ["/usr/local/bin/docker"],
            "docker_autostart_attempted": False,
            "docker_autostart_succeeded": False,
        },
    )

    def _fake_run(self, run_dir: Path, *, custom_solver_dirs, log_name: str):
        if "fast_runtime" in log_name:
            time_s = 0.10006
            angle = -6.968
            pressure = 9.34e5
            temp = 1017.0
            vel = 1123.0
        else:
            time_s = 0.10004
            angle = -6.972
            pressure = 9.325e5
            temp = 1016.0
            vel = 1121.0
        _write_log_summary(
            run_dir,
            time_s=time_s,
            crank_angle_deg=angle,
            mean_pressure_Pa=pressure,
            mean_temperature_K=temp,
            mean_velocity_magnitude_m_s=vel,
        )
        (run_dir / f"{self.solver_cmd}.benchmark.log").write_text(
            "Clipped pressure floor: 10 values\n"
            "Clipped density floor: 2 values\n"
            "Clipped thermo energy state before correction: 5 values\n"
            "Limited thermo correction window with maxThermoDeltaTK=8 on 20 values\n",
            encoding="utf-8",
        )
        (run_dir / "runtimeChemistryCoverageCorpus.json").write_text(
            json.dumps(
                {
                    "state_variables": ["Temperature", "Pressure"],
                    "rows": [
                        {
                            "coverage_bucket_key": [12, 34],
                            "raw_state": {"Temperature": 1016.0, "Pressure": 9.33e5},
                            "transformed_state": {"Temperature": 1016.0, "Pressure": 5.97},
                            "query_count": 2,
                            "table_hit_count": 2,
                            "coverage_reject_count": 0,
                            "trust_reject_count": 0,
                            "worst_reject_variable": None,
                            "worst_reject_excess": 0.0,
                            "nearest_sample_distance_min": 0.0,
                            "nearest_sample_distance_max": 1.0e-8,
                            "stage_names": ["ignition_entry"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / f"runtimeChemistrySummary.{time_s}.dat").write_text(
            "# tableQueryCells tableHitCells fallbackTimesteps interpolationCacheHits coverageRejectCells trustRegionRejectCells\n"
            "4\t4\t0\t0\t0\t0\n",
            encoding="utf-8",
        )
        return True, ""

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline._run_solver_with_custom_dirs_log",
        _fake_run,
    )

    summary = benchmark_engine_restart_profiles(
        run_dir=run_dir,
        tuned_params_path=tuned_params_path,
        handoff_artifact_path=handoff_path,
        outdir=tmp_path / "benchmark_out",
        profiles=["fast_runtime", "low_clamp"],
        runtime_strategy_config=str(strategy_path),
    )

    assert len(summary["profiles"]) == 2
    fast_runtime = next(
        item for item in summary["profiles"] if item["profile_name"] == "fast_runtime"
    )
    assert fast_runtime["runtime_mode"] == "lookupTableStrict"
    assert fast_runtime["docker_bin"] == "/usr/local/bin/docker"
    assert fast_runtime["docker_preflight_ok"] is True
    assert fast_runtime["docker_autostart_attempted"] is False
    assert fast_runtime["docker_autostart_succeeded"] is False
    assert fast_runtime["runtime_coverage_corpus_path"].endswith(
        "runtimeChemistryCoverageCorpus.json"
    )
    assert fast_runtime["runtime_coverage_corpus_npz_path"].endswith(
        "runtimeChemistryCoverageCorpus.npz"
    )
    npz_path = Path(fast_runtime["runtime_coverage_corpus_npz_path"])
    assert npz_path.is_file()
    loaded = np.load(npz_path)
    assert "high_fidelity_trust_reject" in loaded.files
    assert loaded["high_fidelity_trust_reject"].shape[0] == loaded["raw_states"].shape[0]
    assert summary["recommendation"]["fastest_profile"] in {"fast_runtime", "low_clamp", None}
    assert summary["runtime_package_id"] == "chem323_reduced_v2512"
    assert (tmp_path / "benchmark_out" / "engine_restart_benchmark_summary.json").exists()


def test_benchmark_engine_restart_profiles_tuning_characterization_logging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    knob_schema_path = (
        repo_root / "data/simulation_validation/tuning_knob_schema_chem323_ignition_entry_v1.json"
    )
    assert knob_schema_path.is_file()

    table_wrap = tmp_path / "tuning_table_wrap.json"
    expected_knobs = {
        "rbf_envelope_scale": 1.11,
        "rbf_neighbor_count": 10,
        "rbf_epsilon": 1.25,
        "rbf_diag_envelope_scale_ho2": 1.33,
        "max_samples": 450,
    }
    table_wrap.write_text(
        json.dumps(
            {
                "runtime_chemistry_table": {
                    "adaptive_sampling": {
                        "rbf_envelope_scale": expected_knobs["rbf_envelope_scale"],
                        "rbf_neighbor_count": expected_knobs["rbf_neighbor_count"],
                        "rbf_epsilon": expected_knobs["rbf_epsilon"],
                        "max_samples": expected_knobs["max_samples"],
                    },
                    "rbf_diag_envelope_scale_ho2": expected_knobs["rbf_diag_envelope_scale_ho2"],
                }
            }
        ),
        encoding="utf-8",
    )

    runtime_package = tmp_path / "chem323_reduced"
    _write_package(runtime_package, package_id="chem323_reduced_v2512", package_hash="runtime-hash")

    strategy_path = tmp_path / "strategy.json"
    strategy_path.write_text(
        json.dumps(
            {
                "strategy_id": "engine_runtime_mechanism_ladder_v1",
                "runtime_package": {
                    "label": "chem323_runtime",
                    "package_dir": str(runtime_package),
                },
                "checkpoint_packages": [
                    {
                        "label": "chem323_checkpoint_reference",
                        "package_dir": str(runtime_package),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tuned_params_path = tmp_path / "tuned_params.json"
    tuned_params_path.write_text(
        json.dumps({"rpm": 1800.0, "torque": 80.0}),
        encoding="utf-8",
    )
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps({"handoff_bundle": {"cycle_coordinate_deg": -10.0}}),
        encoding="utf-8",
    )

    run_dir = tmp_path / "engine_run"
    for name in ("0", "constant", "system", "chemistry"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "constant" / "engineGeometry").write_text(
        "\n".join(
            [
                "cycleEndAngleDeg 0;",
                "minTemperatureK 300;",
                "maxTemperatureK 1700;",
                "maxThermoDeltaTK 8;",
                "minPressurePa 25000;",
                "minDensityKgM3 0.08;",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "system" / "controlDict").write_text(
        "\n".join(
            [
                "startFrom latestTime;",
                "endTime 0.0003;",
                "deltaT 1e-6;",
                "maxCo 0.02;",
                "maxDeltaT 1e-6;",
                "writeInterval 1;",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "constant" / "chemistryProperties").write_text(
        "\n".join(
            [
                "chemistry on;",
                "initialChemicalTimeStep 1e-8;",
                "odeCoeffs",
                "{",
                "    absTol 1e-14;",
                "    relTol 1e-9;",
                "}",
                "reduction",
                "{",
                "    active on;",
                "}",
                "tabulation",
                "{",
                "    active off;",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "constant" / "combustionProperties").write_text(
        "\n".join(
            [
                "active yes;",
                "PaSRCoeffs",
                "{",
                "    Cmix 0.01;",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "engine_stage_manifest.json").write_text(
        json.dumps(
            {
                "profile": "closed_valve_ignition_v1",
                "stages": [
                    {"name": "settle_flow", "ok": True, "end_angle_deg": -9.25},
                    {"name": "chemistry_seed", "ok": True, "end_angle_deg": -8.5},
                    {"name": "chemistry_spinup", "ok": True, "end_angle_deg": -7.0},
                    {"name": "ignition_release", "ok": False, "end_angle_deg": -4.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "engine_stage_resume_summary.json").write_text(
        json.dumps(
            {
                "remaining_stages": ["ignition_release", "early_burn"],
                "current_stage": "ignition_release",
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "0.1000").mkdir()
    _write_log_summary(
        run_dir,
        time_s=0.1,
        crank_angle_deg=-6.98,
        mean_pressure_Pa=9.30e5,
        mean_temperature_K=1015.0,
        mean_velocity_magnitude_m_s=1120.0,
    )

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline._ensure_custom_solver",
        lambda self, log_file=None: {},
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline.docker_preflight",
        lambda self, log_file=None: {
            "ok": True,
            "docker_bin": "/usr/local/bin/docker",
            "failure_class": "",
            "message": "",
            "candidate_paths": ["/usr/local/bin/docker"],
            "docker_autostart_attempted": False,
            "docker_autostart_succeeded": False,
        },
    )

    def _fake_run(self, run_dir: Path, *, custom_solver_dirs, log_name: str):
        time_s = 0.10006
        angle = -6.968
        pressure = 9.34e5
        temp = 1017.0
        vel = 1123.0
        _write_log_summary(
            run_dir,
            time_s=time_s,
            crank_angle_deg=angle,
            mean_pressure_Pa=pressure,
            mean_temperature_K=temp,
            mean_velocity_magnitude_m_s=vel,
        )
        (run_dir / f"{self.solver_cmd}.benchmark.log").write_text(
            "Clipped pressure floor: 10 values\n"
            "Clipped density floor: 2 values\n"
            "Clipped thermo energy state before correction: 5 values\n"
            "Limited thermo correction window with maxThermoDeltaTK=8 on 20 values\n",
            encoding="utf-8",
        )
        (run_dir / "runtimeChemistryCoverageCorpus.json").write_text(
            json.dumps(
                {
                    "state_variables": ["Temperature", "Pressure"],
                    "rows": [
                        {
                            "coverage_bucket_key": [12, 34],
                            "raw_state": {"Temperature": 1016.0, "Pressure": 9.33e5},
                            "transformed_state": {"Temperature": 1016.0, "Pressure": 5.97},
                            "query_count": 2,
                            "table_hit_count": 2,
                            "coverage_reject_count": 0,
                            "trust_reject_count": 0,
                            "worst_reject_variable": None,
                            "worst_reject_excess": 0.0,
                            "nearest_sample_distance_min": 0.0,
                            "nearest_sample_distance_max": 1.0e-8,
                            "stage_names": ["ignition_entry"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / f"runtimeChemistrySummary.{time_s}.dat").write_text(
            "# tableQueryCells tableHitCells fallbackTimesteps interpolationCacheHits coverageRejectCells trustRegionRejectCells\n"
            "4\t4\t0\t0\t0\t0\n",
            encoding="utf-8",
        )
        return True, ""

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline._run_solver_with_custom_dirs_log",
        _fake_run,
    )

    outdir = tmp_path / "benchmark_out_tuning"
    experiments_jsonl = tmp_path / "experiments.jsonl"
    summary = benchmark_engine_restart_profiles(
        run_dir=run_dir,
        tuned_params_path=tuned_params_path,
        handoff_artifact_path=handoff_path,
        outdir=outdir,
        profiles=["fast_runtime"],
        runtime_strategy_config=str(strategy_path),
        tuning_characterization={
            "enabled": True,
            "experiments_jsonl": str(experiments_jsonl),
            "knob_schema_path": str(knob_schema_path),
            "table_config_path": str(table_wrap),
            "strategy_config_path": str(strategy_path),
            "repo_root": str(repo_root),
        },
    )

    tc = summary.get("tuning_characterization") or {}
    assert tc.get("logged") is True
    assert tc.get("error") == ""
    assert Path(tc["experiments_jsonl"]) == experiments_jsonl.resolve()

    manifest_path = outdir / TUNING_MANIFEST_BASENAME
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["knobs"] == expected_knobs

    rows = load_experiments_jsonl(experiments_jsonl)
    assert len(rows) == 1
    assert rows[0]["knobs_trusted"] is True
    assert rows[0]["knobs"] == expected_knobs
    assert rows[0]["knob_schema_id"] == "chem323_ignition_entry_rbf_v1"

    summary_disk = json.loads(
        (outdir / "engine_restart_benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert summary_disk.get("tuning_characterization", {}).get("logged") is True


def test_benchmark_engine_restart_profiles_continues_across_stages_and_records_stage_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_package = tmp_path / "chem323_reduced"
    _write_package(runtime_package, package_id="chem323_reduced_v2512", package_hash="runtime-hash")
    base_table = tmp_path / "tables" / "base"
    entry_table = tmp_path / "tables" / "entry"
    ramp_table = tmp_path / "tables" / "ramp"
    _write_runtime_table(base_table, table_id="chem323_engine_ignition_v2", table_hash="base-hash")
    _write_runtime_table(
        entry_table,
        table_id="chem323_engine_ignition_entry_v1",
        table_hash="entry-hash",
    )
    _write_runtime_table(
        ramp_table,
        table_id="chem323_engine_ignition_ramp_v1",
        table_hash="ramp-hash",
    )

    entry_config = tmp_path / "entry.json"
    entry_config.write_text(json.dumps({"runtime_chemistry_table": {"table_id": "entry"}}))
    ramp_config = tmp_path / "ramp.json"
    ramp_config.write_text(json.dumps({"runtime_chemistry_table": {"table_id": "ramp"}}))

    strategy_path = tmp_path / "strategy.json"
    strategy_path.write_text(
        json.dumps(
            {
                "strategy_id": "engine_runtime_mechanism_multitable_v1",
                "runtime_package": {
                    "label": "chem323_runtime",
                    "package_dir": str(runtime_package),
                    "runtime_table_dir": str(base_table),
                    "stage_runtime_tables": {
                        "ignition_entry": {
                            "runtime_table_dir": str(entry_table),
                            "runtime_table_config_path": str(entry_config),
                        },
                        "ignition_ramp": {
                            "runtime_table_dir": str(ramp_table),
                            "runtime_table_config_path": str(ramp_config),
                        },
                    },
                },
                "checkpoint_packages": [],
            }
        ),
        encoding="utf-8",
    )

    tuned_params_path = tmp_path / "tuned_params.json"
    tuned_params_path.write_text(json.dumps({"rpm": 1800.0}), encoding="utf-8")
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps({"handoff_bundle": {"cycle_coordinate_deg": -10.0}}),
        encoding="utf-8",
    )

    run_dir = tmp_path / "engine_run"
    for name in ("0", "constant", "system", "chemistry"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "constant" / "engineGeometry").write_text(
        "\n".join(
            [
                "cycleEndAngleDeg 0;",
                "minTemperatureK 300;",
                "maxTemperatureK 1700;",
                "maxThermoDeltaTK 8;",
                "minPressurePa 25000;",
                "minDensityKgM3 0.08;",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "system" / "controlDict").write_text(
        "\n".join(
            [
                "startFrom latestTime;",
                "endTime 0.0003;",
                "deltaT 1e-6;",
                "maxCo 0.02;",
                "maxDeltaT 1e-6;",
                "writeInterval 1;",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "constant" / "chemistryProperties").write_text(
        "\n".join(
            [
                "chemistry on;",
                "tabulation",
                "{",
                "    active off;",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "constant" / "combustionProperties").write_text("active yes;\n", encoding="utf-8")
    (run_dir / "engine_stage_manifest.json").write_text(
        json.dumps({"profile": "closed_valve_ignition_multitable_v1", "stages": []}),
        encoding="utf-8",
    )
    (run_dir / "engine_stage_resume_summary.json").write_text(
        json.dumps({"remaining_stages": ["ignition_entry", "ignition_ramp"], "results": []}),
        encoding="utf-8",
    )
    (run_dir / "0.1000").mkdir()
    _write_log_summary(
        run_dir,
        time_s=0.1,
        crank_angle_deg=-6.98,
        mean_pressure_Pa=9.30e5,
        mean_temperature_K=1015.0,
        mean_velocity_magnitude_m_s=1120.0,
    )

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline._ensure_custom_solver",
        lambda self, log_file=None: {},
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline.docker_preflight",
        lambda self, log_file=None: {
            "ok": True,
            "docker_bin": "/usr/local/bin/docker",
            "failure_class": "",
            "message": "",
            "candidate_paths": ["/usr/local/bin/docker"],
            "docker_autostart_attempted": False,
            "docker_autostart_succeeded": False,
        },
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.emit_engine_progress_artifacts",
        lambda **kwargs: None,
    )

    def _fake_remaining(self, *, base_params, manifest):
        del base_params, manifest
        common = {
            "deltaT": 1.0e-6,
            "maxCo": 0.02,
            "maxDeltaT": 1.0e-6,
            "writeInterval": 1,
        }
        return [
            {"name": "ignition_entry", "end_angle_deg": -6.9, **common},
            {"name": "ignition_ramp", "end_angle_deg": -6.7, **common},
            {"name": "ignition_branch", "end_angle_deg": -6.3, **common},
        ]

    def _fake_solver(self, run_dir: Path, *, custom_solver_dirs, log_name: str):
        del custom_solver_dirs
        (run_dir / log_name).write_text("End\n", encoding="utf-8")
        return True, ""

    def _fake_stage_completion_status(self, run_dir: Path, *, base_params, stage):
        del run_dir, base_params
        return {"latest_angle_deg": float(stage["end_angle_deg"]), "within_tolerance": False}

    def _fake_progress_summary(*, engine_case_dir: Path, **kwargs):
        del kwargs
        return {
            "latest_checkpoint": {"time_s": 0.10004, "crank_angle_deg": -6.68},
            "numeric_stability": {},
            "runtime_chemistry": {
                "coverage_reject_cells": 0.0,
                "fallback_timesteps": 0.0,
                "trust_region_reject_cells": 0.0,
                "table_hit_fraction": 0.95,
                "runtime_summary_count": 2.0,
            },
            "first_offending_variables": [],
            "miss_counts_by_variable": {},
            "max_out_of_bound_by_variable": {},
        }

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline._remaining_engine_stages",
        _fake_remaining,
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline._run_solver_with_custom_dirs_log",
        _fake_solver,
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline._stage_completion_status",
        _fake_stage_completion_status,
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline._apply_engine_stage_settings",
        lambda self, run_dir, *, base_params, stage: None,
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.build_engine_progress_summary",
        _fake_progress_summary,
    )

    summary = benchmark_engine_restart_profiles(
        run_dir=run_dir,
        tuned_params_path=tuned_params_path,
        handoff_artifact_path=handoff_path,
        outdir=tmp_path / "benchmark_multitable_out",
        profiles=["chem323_lookup_strict"],
        window_angle_deg=0.25,
        runtime_strategy_config=str(strategy_path),
        continue_across_remaining_stages=True,
    )

    profile = summary["profiles"][0]
    assert profile["resolved_profile"] == "closed_valve_ignition_multitable_v1"
    assert profile["executed_stage_names"] == ["ignition_entry", "ignition_ramp"]
    assert profile["executed_stage_count"] == 2
    assert profile["docker_autostart_attempted"] is False
    assert profile["docker_autostart_succeeded"] is False
    assert profile["executed_stage_runtime_tables"][0]["runtime_table_id"] == (
        "chem323_engine_ignition_entry_v1"
    )
    assert profile["executed_stage_runtime_tables"][1]["runtime_table_id"] == (
        "chem323_engine_ignition_ramp_v1"
    )


def test_benchmark_engine_restart_profiles_reports_docker_cli_preflight_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_package = tmp_path / "chem323_reduced"
    _write_package(runtime_package, package_id="chem323_reduced_v2512", package_hash="runtime-hash")

    strategy_path = tmp_path / "strategy.json"
    strategy_path.write_text(
        json.dumps(
            {
                "strategy_id": "engine_runtime_mechanism_ladder_v1",
                "runtime_package": {
                    "label": "chem323_runtime",
                    "package_dir": str(runtime_package),
                },
                "checkpoint_packages": [],
            }
        ),
        encoding="utf-8",
    )

    tuned_params_path = tmp_path / "tuned_params.json"
    tuned_params_path.write_text(json.dumps({"rpm": 1800.0}), encoding="utf-8")
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(
        json.dumps({"handoff_bundle": {"cycle_coordinate_deg": -10.0}}),
        encoding="utf-8",
    )

    run_dir = tmp_path / "engine_run"
    for name in ("0", "constant", "system", "chemistry"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "constant" / "engineGeometry").write_text("cycleEndAngleDeg 0;\n", encoding="utf-8")
    (run_dir / "system" / "controlDict").write_text("endTime 0.0003;\n", encoding="utf-8")
    (run_dir / "constant" / "chemistryProperties").write_text("chemistry on;\n", encoding="utf-8")
    (run_dir / "constant" / "combustionProperties").write_text("active yes;\n", encoding="utf-8")
    (run_dir / "engine_stage_manifest.json").write_text(
        json.dumps({"profile": "closed_valve_ignition_v1", "stages": []}),
        encoding="utf-8",
    )
    (run_dir / "engine_stage_resume_summary.json").write_text(
        json.dumps({"remaining_stages": ["ignition_entry"], "results": []}),
        encoding="utf-8",
    )
    (run_dir / "0.1000").mkdir()
    _write_log_summary(
        run_dir,
        time_s=0.1,
        crank_angle_deg=-6.98,
        mean_pressure_Pa=9.30e5,
        mean_temperature_K=1015.0,
        mean_velocity_magnitude_m_s=1120.0,
    )

    def _fake_preflight(self, log_file=None):
        assert self.docker.cfg.docker_bin == "/usr/local/bin/docker"
        return {
            "ok": False,
            "docker_bin": "",
            "failure_class": "docker_cli_missing",
            "message": "Docker CLI not found. Tried: docker, /usr/local/bin/docker",
            "candidate_paths": ["docker", "/usr/local/bin/docker"],
            "docker_autostart_attempted": False,
            "docker_autostart_succeeded": False,
        }

    monkeypatch.setattr(
        "larrak2.simulation_validation.engine_restart_benchmark.OpenFoamPipeline.docker_preflight",
        _fake_preflight,
    )

    summary = benchmark_engine_restart_profiles(
        run_dir=run_dir,
        tuned_params_path=tuned_params_path,
        handoff_artifact_path=handoff_path,
        outdir=tmp_path / "benchmark_out",
        profiles=["fast_runtime"],
        runtime_strategy_config=str(strategy_path),
        docker_bin="/usr/local/bin/docker",
    )

    profile = summary["profiles"][0]
    assert profile["solver_ok"] is False
    assert profile["stage_result"] == "docker_cli_missing"
    assert profile["failed_stage_name"] == ""
    assert profile["launch_failure_class"] == "docker_cli_missing"
    assert "/usr/local/bin/docker" in profile["launch_failure_candidates"]
    assert profile["docker_preflight_ok"] is False
    assert profile["docker_autostart_attempted"] is False
    assert profile["docker_autostart_succeeded"] is False


def test_solver_authority_miss_writer_keeps_seedable_reject_fields() -> None:
    source_text = Path(
        "/Users/maxholden/GitHub/Larrick_multi/openfoam_custom_solvers/larrakEngineFoam/larrakEngineFoam.C"
    ).read_text(encoding="utf-8")

    for required_key in (
        "failure_class",
        "failure_branch_id",
        "reject_variable",
        "reject_excess",
        "reject_nearest_sample_distance",
        "reject_state",
        "noteFirstTrustReject",
    ):
        assert required_key in source_text
