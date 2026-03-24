from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.simulation_validation.engine_results import (
    build_engine_results,
    emit_engine_results_artifact,
)


def _write_scalar_field(path: Path, *, name: str, value: float, location: str) -> None:
    path.write_text(
        "\n".join(
            [
                "FoamFile",
                "{",
                "    format      ascii;",
                "    class       volScalarField;",
                f'    location    "{location}";',
                f"    object      {name};",
                "}",
                "",
                "dimensions      [0 0 0 0 0 0 0];",
                "",
                f"internalField   uniform {value};",
                "",
            ]
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
    stem = str(time_s)
    (run_dir / f"logSummary.{stem}.dat").write_text(
        "\n".join(
            [
                "# crankAngleDeg meanPressurePa meanTemperatureK meanVelocityMagnitude",
                f"{crank_angle_deg} {mean_pressure_Pa} {mean_temperature_K} {mean_velocity_magnitude_m_s}",
            ]
        ),
        encoding="utf-8",
    )


def test_build_engine_results_computes_closed_cylinder_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "engine_case"
    run_dir.mkdir()

    traces = [
        (0.0, -180.0, 100000.0, 600.0, 20.0, 1.20e-3),
        (0.1, -90.0, 180000.0, 650.0, 25.0, 9.0e-4),
        (0.2, 0.0, 500000.0, 1100.0, 40.0, 6.0e-4),
        (0.3, 90.0, 230000.0, 850.0, 30.0, 9.5e-4),
        (0.4, 180.0, 110000.0, 650.0, 18.0, 1.25e-3),
    ]
    for time_s, angle, pressure, temperature, velocity, volume in traces:
        time_dir = run_dir / ("0" if time_s == 0.0 else str(time_s))
        time_dir.mkdir(parents=True, exist_ok=True)
        _write_scalar_field(time_dir / "V", name="V", value=volume, location=time_dir.name)
        _write_log_summary(
            run_dir,
            time_s=time_s,
            crank_angle_deg=angle,
            mean_pressure_Pa=pressure,
            mean_temperature_K=temperature,
            mean_velocity_magnitude_m_s=velocity,
        )

    (run_dir / "engine_stage_manifest.json").write_text(
        json.dumps(
            {
                "profile": "closed_valve_ignition_v1",
                "stages": [
                    {"name": "chemistry_seed", "end_angle_deg": -8.5, "ok": True},
                    {"name": "chemistry_spinup", "end_angle_deg": -7.0, "ok": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "larrakEngineFoam.log").write_text(
        "\n".join(
            [
                "Clipped pressure floor: 3 values to minPressurePa=25000",
                "Clipped density floor: 2 values to minDensityKgM3=0.08",
                "Clipped thermo energy state before correction: 4 values into [Tmin, Tmax] = [300, 1350]",
            ]
        ),
        encoding="utf-8",
    )
    metrics = {
        "trapped_mass": 1.1e-3,
        "residual_fraction": 0.12,
        "trapped_o2_mass": 2.4e-4,
        "scavenging_efficiency": 0.88,
    }

    results = build_engine_results(
        engine_case_dir=run_dir,
        params={"bore_mm": 80.0, "stroke_mm": 90.0},
        engine_metrics=metrics,
        solver_name="larrakEngineFoam",
        handoff_bundle_id="seed_bundle",
        run_ok=True,
        stage="complete",
    )

    assert results["trace_point_count"] == 5
    assert results["metrics"]["peak_pressure_Pa"] == pytest.approx(500000.0)
    assert results["metrics"]["peak_pressure_crank_angle_deg"] == pytest.approx(0.0)
    assert results["metrics"]["imep_Pa"] is not None
    assert results["metrics"]["ca10_deg"] is not None
    assert (
        results["metrics"]["ca10_deg"]
        < results["metrics"]["ca50_deg"]
        < results["metrics"]["ca90_deg"]
    )
    assert results["metrics"]["trapped_mass"] == pytest.approx(1.1e-3)
    assert results["metrics"]["residual_fraction"] == pytest.approx(0.12)
    assert results["metrics"]["trapped_o2_mass"] == pytest.approx(2.4e-4)
    assert results["numeric_stability"]["pressure_floor_hits"] == 3
    assert results["numeric_stability"]["density_floor_hits"] == 2
    assert results["stage_boundaries"][0]["name"] == "chemistry_seed"


def test_emit_engine_results_artifact_writes_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "engine_case"
    run_dir.mkdir()
    (run_dir / "0").mkdir()
    _write_scalar_field(run_dir / "0" / "V", name="V", value=1.0e-3, location="0")
    _write_log_summary(
        run_dir,
        time_s=0.0,
        crank_angle_deg=-10.0,
        mean_pressure_Pa=150000.0,
        mean_temperature_K=700.0,
        mean_velocity_magnitude_m_s=10.0,
    )

    payload = emit_engine_results_artifact(
        engine_case_dir=run_dir,
        params={"bore_mm": 80.0, "stroke_mm": 90.0},
        engine_metrics={"trapped_mass": 9.0e-4},
        solver_name="larrakEngineFoam",
    )

    artifact_path = run_dir / "engine_results.json"
    assert artifact_path.exists()
    loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert loaded["solver_name"] == "larrakEngineFoam"
    assert payload["metrics"]["trapped_mass"] == pytest.approx(9.0e-4)
