from __future__ import annotations

import json
from pathlib import Path

from larrak2.simulation_validation.engine_restart_benchmark import (
    benchmark_engine_restart_profiles,
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
    assert summary["recommendation"]["fastest_profile"] in {"fast_runtime", "low_clamp", None}
    assert summary["runtime_package_id"] == "chem323_reduced_v2512"
    assert (tmp_path / "benchmark_out" / "engine_restart_benchmark_summary.json").exists()
