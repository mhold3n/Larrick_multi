from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.pipelines.openfoam import OpenFoamPipeline


def _write_uniform_scalar_field(path: Path, *, name: str, value: float, location: str) -> None:
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
    mean_pressure_Pa: float = 1.0e5,
    mean_temperature_K: float = 700.0,
    mean_velocity_magnitude_m_s: float = 10.0,
) -> None:
    (run_dir / f"logSummary.{time_s}.dat").write_text(
        "\n".join(
            [
                "# crankAngleDeg meanPressurePa meanTemperatureK meanVelocityMagnitude",
                f"{crank_angle_deg} {mean_pressure_Pa} {mean_temperature_K} {mean_velocity_magnitude_m_s}",
            ]
        ),
        encoding="utf-8",
    )


def test_default_engine_execute_stages_package_and_handoff(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / "chem323_reduced"
    package_dir.mkdir(parents=True)
    (package_dir / "reactions").write_text("reactions\n", encoding="utf-8")
    (package_dir / "thermo.compressibleGas").write_text("thermo\n", encoding="utf-8")
    (package_dir / "transportProperties").write_text("transport\n", encoding="utf-8")
    (package_dir / "package_manifest.json").write_text(
        json.dumps(
            {
                "package_id": "chem323_reduced_v2512",
                "package_hash": "package-hash",
            }
        ),
        encoding="utf-8",
    )

    solver_bin = tmp_path / "solver-bin"
    solver_bin.mkdir()
    (solver_bin / "larrakEngineFoam").write_text("binary\n", encoding="utf-8")

    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
        chemistry_package_dir=package_dir,
        custom_solver_source_dir=tmp_path / "solver-src",
        custom_solver_cache_root=tmp_path / "solver-cache",
    )

    monkeypatch.setattr(
        pipeline,
        "_ensure_custom_solver",
        lambda log_file=None: {
            "binary_path": str(solver_bin / "larrakEngineFoam"),
            "source_hash": "solver-hash",
        },
    )
    monkeypatch.setattr(pipeline, "run_meshing", lambda run_dir: (True, ""))

    observed: dict[str, object] = {}

    def _fake_run_solver_with_custom_dirs(run_dir: Path, *, custom_solver_dirs):
        observed["custom_solver_dirs"] = list(custom_solver_dirs or [])
        return True, ""

    monkeypatch.setattr(pipeline, "run_solver_with_custom_dirs", _fake_run_solver_with_custom_dirs)
    monkeypatch.setattr(
        pipeline,
        "_ensure_case_metrics",
        lambda run_dir, params: {
            "trapped_mass": 1.0e-3,
            "scavenging_efficiency": 0.92,
            "residual_fraction": 0.08,
            "trapped_o2_mass": 2.0e-4,
        },
    )

    run_dir = tmp_path / "engine_case"
    result = pipeline.execute(
        run_dir=run_dir,
        params={
            "rpm": 1800.0,
            "torque": 80.0,
            "lambda_af": 1.0,
            "p_manifold_Pa": 120000.0,
            "p_back_Pa": 101325.0,
            "T_intake_K": 305.0,
            "T_residual_K": 900.0,
            "endTime": 3.0e-4,
            "deltaT": 1.0e-4,
            "writeInterval": 1,
            "solver_name": "larrakEngineFoam",
        },
        handoff_bundle={
            "bundle_id": "seed_bundle",
            "mechanism_id": "chem323_reduced_v2512",
            "pressure_Pa": 150000.0,
            "temperature_K": 410.0,
            "species_mole_fractions": {
                "IC8H18": 0.02,
                "O2": 0.21,
                "N2": 0.75,
                "CO2": 0.02,
            },
            "cycle_coordinate_deg": -170.0,
            "residual_fraction": 0.12,
        },
    )

    assert (run_dir / "constant" / "reactions").read_text(encoding="utf-8") == "reactions\n"
    assert (run_dir / "chemistry" / "package_manifest.json").is_file()
    assert "{{" not in (run_dir / "0" / "O2").read_text(encoding="utf-8")
    assert "{{" not in (run_dir / "constant" / "engineGeometry").read_text(encoding="utf-8")
    assert "{{" not in (run_dir / "constant" / "dynamicMeshDict").read_text(encoding="utf-8")
    assert "{{" not in (run_dir / "constant" / "thermophysicalProperties").read_text(
        encoding="utf-8"
    )
    assert "{{" not in (run_dir / "system" / "createBafflesDict").read_text(encoding="utf-8")
    assert "{{" not in (run_dir / "system" / "blockMeshDict").read_text(encoding="utf-8")
    assert "{{" not in (run_dir / "system" / "snappyHexMeshDict").read_text(encoding="utf-8")
    control_dict = (run_dir / "system" / "controlDict").read_text(encoding="utf-8")
    assert "{{" not in control_dict
    assert "adjustTimeStep  yes;" in control_dict
    assert "maxCo           1.0;" in control_dict
    assert "maxDeltaT       0.0001;" in control_dict
    thermo_props = (run_dir / "constant" / "thermophysicalProperties").read_text(encoding="utf-8")
    temp_field = (run_dir / "0" / "T").read_text(encoding="utf-8")
    residual_field = (run_dir / "0" / "residualTracer").read_text(encoding="utf-8")
    o2_field = (run_dir / "0" / "O2").read_text(encoding="utf-8")
    snappy_dict = (run_dir / "system" / "snappyHexMeshDict").read_text(encoding="utf-8")
    block_mesh = (run_dir / "system" / "blockMeshDict").read_text(encoding="utf-8")
    assert "energy          sensibleInternalEnergy;" in thermo_props
    assert "value uniform 450.0;" in temp_field
    assert "type        inletOutlet;" in temp_field
    assert "internalField   uniform 0.12;" in residual_field
    assert "type            inletOutlet;" in o2_field
    assert "inletValue      uniform 0.21750134247500277;" in o2_field
    assert "intakeManifold.stl" in snappy_dict
    assert "exhaustManifoldLeft.stl" in snappy_dict
    assert "exhaustManifoldRight.stl" in snappy_dict
    assert "locationInMesh (0 0 0.0675);" in snappy_dict
    assert "(-0.12 -0.16 0)" in block_mesh
    assert "(0.12 0.16 0.18)" in block_mesh
    assert observed["custom_solver_dirs"] == [solver_bin]
    assert result["handoff_bundle_id"] == "seed_bundle"
    assert result["openfoam_chemistry_package_id"] == "chem323_reduced_v2512"
    assert result["custom_solver_source_hash"] == "solver-hash"


def test_default_engine_execute_can_disable_chemistry_for_breathing_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "chem323_reduced"
    package_dir.mkdir(parents=True)
    (package_dir / "reactions").write_text("reactions\n", encoding="utf-8")
    (package_dir / "thermo.compressibleGas").write_text("thermo\n", encoding="utf-8")
    (package_dir / "transportProperties").write_text("transport\n", encoding="utf-8")
    (package_dir / "package_manifest.json").write_text(
        json.dumps({"package_id": "chem323_reduced_v2512", "package_hash": "package-hash"}),
        encoding="utf-8",
    )

    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
        chemistry_package_dir=package_dir,
        custom_solver_source_dir=tmp_path / "solver-src",
        custom_solver_cache_root=tmp_path / "solver-cache",
    )
    monkeypatch.setattr(pipeline, "_ensure_custom_solver", lambda log_file=None: {})
    monkeypatch.setattr(pipeline, "run_meshing", lambda run_dir: (True, ""))
    monkeypatch.setattr(
        pipeline, "run_solver_with_custom_dirs", lambda run_dir, *, custom_solver_dirs: (True, "")
    )
    monkeypatch.setattr(
        pipeline,
        "_ensure_case_metrics",
        lambda run_dir, params: {
            "trapped_mass": 1.0e-3,
            "scavenging_efficiency": 0.9,
            "residual_fraction": 0.1,
            "trapped_o2_mass": 1.5e-4,
        },
    )

    run_dir = tmp_path / "engine_case"
    result = pipeline.execute(
        run_dir=run_dir,
        params={
            "rpm": 1800.0,
            "torque": 80.0,
            "lambda_af": 1.0,
            "chemistry_enabled": False,
            "combustion_enabled": False,
            "mesh_nx": 12,
            "mesh_ny": 12,
            "mesh_nz": 18,
            "surface_level": 0,
            "interface_level": 0,
            "endTime": 3.0e-4,
            "deltaT": 1.0e-4,
            "writeInterval": 1,
            "solver_name": "larrakEngineFoam",
        },
    )

    chemistry_props = (run_dir / "constant" / "chemistryProperties").read_text(encoding="utf-8")
    combustion_props = (run_dir / "constant" / "combustionProperties").read_text(encoding="utf-8")
    block_mesh = (run_dir / "system" / "blockMeshDict").read_text(encoding="utf-8")
    snappy_dict = (run_dir / "system" / "snappyHexMeshDict").read_text(encoding="utf-8")
    thermo_props = (run_dir / "constant" / "thermophysicalProperties").read_text(encoding="utf-8")
    temp_field = (run_dir / "0" / "T").read_text(encoding="utf-8")

    assert "chemistry       off;" in chemistry_props
    assert "active               no;" in combustion_props
    assert "energy          sensibleInternalEnergy;" in thermo_props
    assert "(12 12 18)" in block_mesh
    assert "cylinder { level (0 0); }" in snappy_dict
    assert "level           (0 0);" in snappy_dict
    assert "uniform 375.0" in temp_field
    assert result["ok"] is True


def test_default_engine_execute_respects_package_dir_override_in_params(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_package = tmp_path / "chem323_reduced"
    runtime_package.mkdir(parents=True)
    (runtime_package / "reactions").write_text("runtime\n", encoding="utf-8")
    (runtime_package / "thermo.compressibleGas").write_text("thermo-runtime\n", encoding="utf-8")
    (runtime_package / "transportProperties").write_text("transport-runtime\n", encoding="utf-8")
    (runtime_package / "package_manifest.json").write_text(
        json.dumps({"package_id": "chem323_reduced_v2512", "package_hash": "runtime-hash"}),
        encoding="utf-8",
    )

    checkpoint_package = tmp_path / "chem679_reduced"
    checkpoint_package.mkdir(parents=True)
    (checkpoint_package / "reactions").write_text("checkpoint\n", encoding="utf-8")
    (checkpoint_package / "thermo.compressibleGas").write_text(
        "thermo-checkpoint\n", encoding="utf-8"
    )
    (checkpoint_package / "transportProperties").write_text(
        "transport-checkpoint\n", encoding="utf-8"
    )
    (checkpoint_package / "package_manifest.json").write_text(
        json.dumps({"package_id": "chem679_reduced_v2512", "package_hash": "checkpoint-hash"}),
        encoding="utf-8",
    )

    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
        chemistry_package_dir=runtime_package,
        custom_solver_source_dir=tmp_path / "solver-src",
        custom_solver_cache_root=tmp_path / "solver-cache",
    )
    monkeypatch.setattr(pipeline, "_ensure_custom_solver", lambda log_file=None: {})
    monkeypatch.setattr(pipeline, "run_meshing", lambda run_dir: (True, ""))
    monkeypatch.setattr(
        pipeline, "run_solver_with_custom_dirs", lambda run_dir, *, custom_solver_dirs: (True, "")
    )
    monkeypatch.setattr(
        pipeline,
        "_ensure_case_metrics",
        lambda run_dir, params: {
            "trapped_mass": 1.0e-3,
            "scavenging_efficiency": 0.9,
            "residual_fraction": 0.1,
            "trapped_o2_mass": 1.5e-4,
        },
    )

    run_dir = tmp_path / "engine_case"
    result = pipeline.execute(
        run_dir=run_dir,
        params={
            "rpm": 1800.0,
            "torque": 80.0,
            "lambda_af": 1.0,
            "openfoam_chemistry_package_dir": str(checkpoint_package),
            "endTime": 3.0e-4,
            "deltaT": 1.0e-4,
            "writeInterval": 1,
            "solver_name": "larrakEngineFoam",
        },
    )

    assert (run_dir / "constant" / "reactions").read_text(encoding="utf-8") == "checkpoint\n"
    assert result["openfoam_chemistry_package_id"] == "chem679_reduced_v2512"
    assert result["openfoam_chemistry_package_hash"] == "checkpoint-hash"


def test_fast_runtime_stage_profile_enables_tabulation_and_larger_timesteps() -> None:
    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
    )

    stages = pipeline._default_engine_stage_sequence(  # noqa: SLF001
        {
            "engine_stage_profile": "closed_valve_ignition_fast_runtime_v1",
            "engine_start_angle_deg": -10.0,
            "engine_end_angle_deg": 0.0,
            "engine_max_temperature_K": 1700.0,
        }
    )

    assert [stage["name"] for stage in stages] == [
        "settle_flow",
        "chemistry_seed",
        "chemistry_spinup",
        "ignition_release",
        "early_burn",
    ]
    assert stages[1]["chemistry_tabulation_enabled"] is True
    assert stages[2]["chemistry_tabulation_enabled"] is True
    assert stages[1]["runtime_chemistry_mode"] == "lookupTableStrict"
    assert stages[2]["runtime_chemistry_mode"] == "lookupTableStrict"
    assert stages[3]["deltaT"] > 2.5e-7
    assert stages[3]["maxCo"] > 0.03


def test_low_clamp_stage_profile_uses_tighter_timestep_and_higher_temp_headroom() -> None:
    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
    )

    stages = pipeline._default_engine_stage_sequence(  # noqa: SLF001
        {
            "engine_stage_profile": "closed_valve_ignition_low_clamp_v1",
            "engine_start_angle_deg": -10.0,
            "engine_end_angle_deg": 0.0,
            "engine_max_temperature_K": 1700.0,
        }
    )

    ignition_release = next(stage for stage in stages if stage["name"] == "ignition_release")
    chemistry_spinup = next(stage for stage in stages if stage["name"] == "chemistry_spinup")
    assert chemistry_spinup["deltaT"] < 2.5e-7
    assert chemistry_spinup["runtime_chemistry_mode"] == "fullReducedKinetics"
    assert ignition_release["deltaT"] < 2.5e-7
    assert ignition_release["engine_max_temperature_K"] >= 1700.0
    assert ignition_release["maxCo"] < 0.03


def test_default_engine_execute_stages_runtime_chemistry_table_when_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "chem323_reduced"
    package_dir.mkdir(parents=True)
    (package_dir / "reactions").write_text("reactions\n", encoding="utf-8")
    (package_dir / "thermo.compressibleGas").write_text("thermo\n", encoding="utf-8")
    (package_dir / "transportProperties").write_text("transport\n", encoding="utf-8")
    (package_dir / "package_manifest.json").write_text(
        json.dumps({"package_id": "chem323_reduced_v2512", "package_hash": "package-hash"}),
        encoding="utf-8",
    )

    table_dir = tmp_path / "runtime_table"
    table_dir.mkdir(parents=True)
    (table_dir / "runtimeChemistryTable").write_text("table\n", encoding="utf-8")
    (table_dir / "runtime_chemistry_jacobian.json").write_text("{}", encoding="utf-8")
    (table_dir / "runtime_chemistry_table_manifest.json").write_text(
        json.dumps(
            {
                "table_id": "chem323_engine_ignition_v1",
                "fallback_policy": "fullReducedKinetics",
                "interpolation_method": "nearest",
                "max_untracked_mass_fraction": 0.01,
                "generated_file_hashes": {
                    "runtimeChemistryTable": "table-hash",
                    "runtime_chemistry_jacobian": "jacobian-hash",
                },
            }
        ),
        encoding="utf-8",
    )

    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
        chemistry_package_dir=package_dir,
        runtime_chemistry_table_dir=table_dir,
        custom_solver_source_dir=tmp_path / "solver-src",
        custom_solver_cache_root=tmp_path / "solver-cache",
    )
    monkeypatch.setattr(pipeline, "_ensure_custom_solver", lambda log_file=None: {})
    monkeypatch.setattr(pipeline, "run_meshing", lambda run_dir: (True, ""))
    monkeypatch.setattr(
        pipeline, "run_solver_with_custom_dirs", lambda run_dir, *, custom_solver_dirs: (True, "")
    )
    monkeypatch.setattr(
        pipeline,
        "_ensure_case_metrics",
        lambda run_dir, params: {
            "trapped_mass": 1.0e-3,
            "scavenging_efficiency": 0.9,
            "residual_fraction": 0.1,
            "trapped_o2_mass": 1.5e-4,
        },
    )

    run_dir = tmp_path / "engine_case"
    result = pipeline.execute(
        run_dir=run_dir,
        params={
            "rpm": 1800.0,
            "torque": 80.0,
            "lambda_af": 1.0,
            "endTime": 3.0e-4,
            "deltaT": 1.0e-4,
            "writeInterval": 1,
            "solver_name": "larrakEngineFoam",
        },
    )

    assert (run_dir / "constant" / "runtimeChemistryTable").read_text(encoding="utf-8") == "table\n"
    assert (run_dir / "chemistry" / "runtime_chemistry_table_manifest.json").is_file()
    assert (run_dir / "chemistry" / "runtime_chemistry_jacobian.json").is_file()
    engine_geometry = (run_dir / "constant" / "engineGeometry").read_text(encoding="utf-8")
    assert "runtimeChemistryMode    lookupTableStrict;" in engine_geometry
    assert "runtimeChemistryStrict  true;" in engine_geometry
    assert "runtimeChemistryAbortOnAuthorityMiss true;" in engine_geometry
    assert "runtimeChemistryFallbackPolicy fullReducedKinetics;" in engine_geometry
    assert result["runtime_chemistry_table_id"] == "chem323_engine_ignition_v1"
    assert result["runtime_chemistry_table_hash"] == "table-hash"


def test_default_engine_execute_emits_engine_results_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "chem323_reduced"
    package_dir.mkdir(parents=True)
    (package_dir / "reactions").write_text("reactions\n", encoding="utf-8")
    (package_dir / "thermo.compressibleGas").write_text("thermo\n", encoding="utf-8")
    (package_dir / "transportProperties").write_text("transport\n", encoding="utf-8")
    (package_dir / "package_manifest.json").write_text(
        json.dumps({"package_id": "chem323_reduced_v2512", "package_hash": "package-hash"}),
        encoding="utf-8",
    )

    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
        chemistry_package_dir=package_dir,
        custom_solver_source_dir=tmp_path / "solver-src",
        custom_solver_cache_root=tmp_path / "solver-cache",
    )
    monkeypatch.setattr(pipeline, "_ensure_custom_solver", lambda log_file=None: {})
    monkeypatch.setattr(pipeline, "run_meshing", lambda run_dir: (True, ""))

    def _fake_run_solver_with_custom_dirs(run_dir: Path, *, custom_solver_dirs):
        traces = [
            ("0", "0", -180.0, 100000.0, 600.0, 20.0, 1.20e-3),
            ("0.1", "0.1", -90.0, 180000.0, 650.0, 25.0, 9.0e-4),
            ("0.2", "0.2", 0.0, 500000.0, 1100.0, 40.0, 6.0e-4),
            ("0.3", "0.3", 90.0, 230000.0, 850.0, 30.0, 9.5e-4),
            ("0.4", "0.4", 180.0, 110000.0, 650.0, 18.0, 1.25e-3),
        ]
        for time_name, stem, angle, pressure, temperature, velocity, volume in traces:
            time_dir = run_dir / time_name
            time_dir.mkdir(parents=True, exist_ok=True)
            _write_uniform_scalar_field(time_dir / "V", name="V", value=volume, location=time_name)
            (run_dir / f"logSummary.{stem}.dat").write_text(
                "\n".join(
                    [
                        "# crankAngleDeg meanPressurePa meanTemperatureK meanVelocityMagnitude",
                        f"{angle} {pressure} {temperature} {velocity}",
                    ]
                ),
                encoding="utf-8",
            )
        return True, ""

    monkeypatch.setattr(pipeline, "run_solver_with_custom_dirs", _fake_run_solver_with_custom_dirs)
    monkeypatch.setattr(
        pipeline,
        "_ensure_case_metrics",
        lambda run_dir, params: {
            "trapped_mass": 1.0e-3,
            "scavenging_efficiency": 0.91,
            "residual_fraction": 0.09,
            "trapped_o2_mass": 2.2e-4,
        },
    )

    run_dir = tmp_path / "engine_case"
    result = pipeline.execute(
        run_dir=run_dir,
        params={
            "rpm": 1800.0,
            "torque": 80.0,
            "lambda_af": 1.0,
            "bore_mm": 80.0,
            "stroke_mm": 90.0,
            "solver_name": "larrakEngineFoam",
        },
    )

    artifact_path = run_dir / "engine_results.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact_path.exists()
    assert result["engine_results_path"] == str(artifact_path)
    assert result["peak_pressure_Pa"] == pytest.approx(500000.0)
    assert payload["metrics"]["trapped_mass"] == pytest.approx(1.0e-3)
    assert payload["metrics"]["imep_Pa"] is not None
    assert payload["metrics"]["ca10_deg"] is not None


def test_staged_engine_profile_accepts_near_target_stage_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "chem323_reduced"
    package_dir.mkdir(parents=True)
    (package_dir / "reactions").write_text("reactions\n", encoding="utf-8")
    (package_dir / "thermo.compressibleGas").write_text("thermo\n", encoding="utf-8")
    (package_dir / "transportProperties").write_text("transport\n", encoding="utf-8")
    (package_dir / "package_manifest.json").write_text(
        json.dumps({"package_id": "chem323_reduced_v2512", "package_hash": "package-hash"}),
        encoding="utf-8",
    )

    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
        chemistry_package_dir=package_dir,
        custom_solver_source_dir=tmp_path / "solver-src",
        custom_solver_cache_root=tmp_path / "solver-cache",
    )
    monkeypatch.setattr(pipeline, "_ensure_custom_solver", lambda log_file=None: {})
    monkeypatch.setattr(pipeline, "run_meshing", lambda run_dir: (True, ""))
    monkeypatch.setattr(
        pipeline,
        "_ensure_case_metrics",
        lambda run_dir, params: {
            "trapped_mass": 1.0e-3,
            "scavenging_efficiency": 0.9,
            "residual_fraction": 0.1,
            "trapped_o2_mass": 1.5e-4,
        },
    )

    stage_calls: list[str] = []

    def _fake_stage_run(run_dir: Path, *, custom_solver_dirs, log_name: str):
        stage_calls.append(log_name)
        if "stage_03_chemistry_spinup" in log_name:
            _write_log_summary(
                run_dir,
                time_s=0.00027777360028019994,
                crank_angle_deg=-7.00005,
                mean_pressure_Pa=9.25153e5,
                mean_temperature_K=1010.86,
                mean_velocity_magnitude_m_s=1106.27,
            )
            return False, "solver"
        return True, ""

    monkeypatch.setattr(pipeline, "_run_solver_with_custom_dirs_log", _fake_stage_run)

    run_dir = tmp_path / "engine_case"
    result = pipeline.execute(
        run_dir=run_dir,
        params={
            "rpm": 1800.0,
            "torque": 80.0,
            "lambda_af": 1.0,
            "engine_proof_mode": "reacting_staged_ignition",
            "engine_start_angle_deg": -10.0,
            "engine_end_angle_deg": 0.0,
            "solver_name": "larrakEngineFoam",
        },
    )

    manifest = json.loads((run_dir / "engine_stage_manifest.json").read_text(encoding="utf-8"))
    chemistry_spinup = manifest["stages"][2]

    assert result["ok"] is True
    assert len(stage_calls) == 5
    assert chemistry_spinup["name"] == "chemistry_spinup"
    assert chemistry_spinup["ok"] is True
    assert chemistry_spinup["stage_result"] == ""
    assert chemistry_spinup["completion_mode"] == "near_target_tolerance"
    assert chemistry_spinup["completion_status"]["within_tolerance"] is True
    assert chemistry_spinup["completion_status"]["time_gap_s"] == pytest.approx(
        4.177497577842384e-09
    )


def test_default_engine_execute_proof_mode_derives_full_cycle_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "chem323_reduced"
    package_dir.mkdir(parents=True)
    (package_dir / "reactions").write_text("reactions\n", encoding="utf-8")
    (package_dir / "thermo.compressibleGas").write_text("thermo\n", encoding="utf-8")
    (package_dir / "transportProperties").write_text("transport\n", encoding="utf-8")
    (package_dir / "package_manifest.json").write_text(
        json.dumps({"package_id": "chem323_reduced_v2512", "package_hash": "package-hash"}),
        encoding="utf-8",
    )

    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
        chemistry_package_dir=package_dir,
        custom_solver_source_dir=tmp_path / "solver-src",
        custom_solver_cache_root=tmp_path / "solver-cache",
    )
    monkeypatch.setattr(pipeline, "_ensure_custom_solver", lambda log_file=None: {})
    monkeypatch.setattr(pipeline, "run_meshing", lambda run_dir: (True, ""))
    monkeypatch.setattr(
        pipeline, "run_solver_with_custom_dirs", lambda run_dir, *, custom_solver_dirs: (True, "")
    )
    monkeypatch.setattr(
        pipeline,
        "_ensure_case_metrics",
        lambda run_dir, params: {
            "trapped_mass": 1.0e-3,
            "scavenging_efficiency": 0.9,
            "residual_fraction": 0.1,
            "trapped_o2_mass": 1.5e-4,
        },
    )

    run_dir = tmp_path / "engine_case"
    result = pipeline.execute(
        run_dir=run_dir,
        params={
            "rpm": 1800.0,
            "torque": 80.0,
            "lambda_af": 1.0,
            "engine_proof_mode": "full_cycle_breathing",
            "engine_start_angle_deg": -180.0,
            "engine_end_angle_deg": 180.0,
            "deltaT": 1.0e-4,
            "solver_name": "larrakEngineFoam",
        },
    )

    control_dict = (run_dir / "system" / "controlDict").read_text(encoding="utf-8")
    chemistry_props = (run_dir / "constant" / "chemistryProperties").read_text(encoding="utf-8")
    combustion_props = (run_dir / "constant" / "combustionProperties").read_text(encoding="utf-8")
    engine_geometry = (run_dir / "constant" / "engineGeometry").read_text(encoding="utf-8")
    block_mesh = (run_dir / "system" / "blockMeshDict").read_text(encoding="utf-8")

    assert "endTime         0.03333333333333333;" in control_dict
    assert "maxCo           2.0;" in control_dict
    assert "maxDeltaT       0.0001;" in control_dict
    assert "writeInterval   50;" in control_dict
    assert "chemistry       off;" in chemistry_props
    assert "active               no;" in combustion_props
    assert "minTemperatureK         300.0;" in engine_geometry
    assert "maxTemperatureK         1350.0;" in engine_geometry
    assert "(10 10 14)" in block_mesh
    assert result["ok"] is True


def test_default_engine_execute_reacting_calibration_profile_enables_chemistry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "chem323_reduced"
    package_dir.mkdir(parents=True)
    (package_dir / "reactions").write_text("reactions\n", encoding="utf-8")
    (package_dir / "thermo.compressibleGas").write_text("thermo\n", encoding="utf-8")
    (package_dir / "transportProperties").write_text("transport\n", encoding="utf-8")
    (package_dir / "package_manifest.json").write_text(
        json.dumps({"package_id": "chem323_reduced_v2512", "package_hash": "package-hash"}),
        encoding="utf-8",
    )

    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
        chemistry_package_dir=package_dir,
        custom_solver_source_dir=tmp_path / "solver-src",
        custom_solver_cache_root=tmp_path / "solver-cache",
    )
    monkeypatch.setattr(pipeline, "_ensure_custom_solver", lambda log_file=None: {})
    monkeypatch.setattr(pipeline, "run_meshing", lambda run_dir: (True, ""))
    monkeypatch.setattr(
        pipeline, "run_solver_with_custom_dirs", lambda run_dir, *, custom_solver_dirs: (True, "")
    )
    monkeypatch.setattr(
        pipeline,
        "_ensure_case_metrics",
        lambda run_dir, params: {
            "trapped_mass": 1.0e-3,
            "scavenging_efficiency": 0.9,
            "residual_fraction": 0.1,
            "trapped_o2_mass": 1.5e-4,
        },
    )

    run_dir = tmp_path / "engine_case"
    result = pipeline.execute(
        run_dir=run_dir,
        params={
            "rpm": 1800.0,
            "torque": 80.0,
            "lambda_af": 1.0,
            "engine_proof_mode": "reacting_calibration_window",
            "solver_name": "larrakEngineFoam",
        },
    )

    control_dict = (run_dir / "system" / "controlDict").read_text(encoding="utf-8")
    chemistry_props = (run_dir / "constant" / "chemistryProperties").read_text(encoding="utf-8")
    combustion_props = (run_dir / "constant" / "combustionProperties").read_text(encoding="utf-8")
    engine_geometry = (run_dir / "constant" / "engineGeometry").read_text(encoding="utf-8")

    assert "endTime         0.005555555555555556;" in control_dict
    assert "maxCo           0.5;" in control_dict
    assert "maxDeltaT       2e-05;" in control_dict
    assert "writeInterval   10;" in control_dict
    assert "chemistry       on;" in chemistry_props
    assert "active               yes;" in combustion_props
    assert "minTemperatureK         300.0;" in engine_geometry
    assert "maxTemperatureK         1350.0;" in engine_geometry
    assert result["ok"] is True


def test_default_engine_execute_staged_reacting_profile_runs_multiple_segments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "chem323_reduced"
    package_dir.mkdir(parents=True)
    (package_dir / "reactions").write_text("reactions\n", encoding="utf-8")
    (package_dir / "thermo.compressibleGas").write_text("thermo\n", encoding="utf-8")
    (package_dir / "transportProperties").write_text("transport\n", encoding="utf-8")
    (package_dir / "package_manifest.json").write_text(
        json.dumps({"package_id": "chem323_reduced_v2512", "package_hash": "package-hash"}),
        encoding="utf-8",
    )

    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
        chemistry_package_dir=package_dir,
        custom_solver_source_dir=tmp_path / "solver-src",
        custom_solver_cache_root=tmp_path / "solver-cache",
    )
    monkeypatch.setattr(pipeline, "_ensure_custom_solver", lambda log_file=None: {})
    monkeypatch.setattr(pipeline, "run_meshing", lambda run_dir: (True, ""))
    snapshots: list[dict[str, str]] = []

    def _fake_stage_run(run_dir: Path, *, custom_solver_dirs, log_name: str):
        snapshots.append(
            {
                "log_name": log_name,
                "control": (run_dir / "system" / "controlDict").read_text(encoding="utf-8"),
                "chemistry": (run_dir / "constant" / "chemistryProperties").read_text(
                    encoding="utf-8"
                ),
                "combustion": (run_dir / "constant" / "combustionProperties").read_text(
                    encoding="utf-8"
                ),
                "engine": (run_dir / "constant" / "engineGeometry").read_text(encoding="utf-8"),
            }
        )
        return True, ""

    monkeypatch.setattr(pipeline, "_run_solver_with_custom_dirs_log", _fake_stage_run)
    monkeypatch.setattr(
        pipeline,
        "_ensure_case_metrics",
        lambda run_dir, params: {
            "trapped_mass": 1.0e-3,
            "scavenging_efficiency": 0.9,
            "residual_fraction": 0.1,
            "trapped_o2_mass": 1.5e-4,
        },
    )

    run_dir = tmp_path / "engine_case"
    result = pipeline.execute(
        run_dir=run_dir,
        params={
            "rpm": 1800.0,
            "torque": 80.0,
            "lambda_af": 1.0,
            "engine_proof_mode": "reacting_staged_ignition",
            "engine_start_angle_deg": -10.0,
            "engine_end_angle_deg": 0.0,
            "solver_name": "larrakEngineFoam",
        },
    )

    manifest = json.loads((run_dir / "engine_stage_manifest.json").read_text(encoding="utf-8"))
    assert len(snapshots) == 5
    assert len(manifest["stages"]) == 5
    assert manifest["profile"] == "closed_valve_ignition_v1"
    assert "chemistry       off;" in snapshots[0]["chemistry"]
    assert "active               no;" in snapshots[0]["combustion"]
    assert "chemistry       on;" in snapshots[1]["chemistry"]
    assert "active               no;" in snapshots[1]["combustion"]
    assert "chemistry       on;" in snapshots[2]["chemistry"]
    assert "active               no;" in snapshots[2]["combustion"]
    assert "active               yes;" in snapshots[3]["combustion"]
    assert "tabulation" in snapshots[1]["chemistry"]
    assert "active          off;" in snapshots[1]["chemistry"]
    assert "maxThermoDeltaTK        6;" in snapshots[1]["engine"]
    assert "minPressurePa           25000;" in snapshots[1]["engine"]
    assert "minDensityKgM3          0.08;" in snapshots[1]["engine"]
    assert "maxThermoDeltaTK        4;" in snapshots[2]["engine"]
    assert "minPressurePa           30000;" in snapshots[2]["engine"]
    assert "minDensityKgM3          0.1;" in snapshots[2]["engine"]
    assert "maxCo           0.08;" in snapshots[0]["control"]
    assert "maxCo           0.025;" in snapshots[-1]["control"]
    assert "cycleEndAngleDeg        0;" in snapshots[-1]["engine"]
    assert result["ok"] is True


def test_engine_stage_resume_summary_syncs_from_manifest(tmp_path: Path) -> None:
    pipeline = OpenFoamPipeline(
        template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
        solver_cmd="larrakEngineFoam",
    )
    run_dir = tmp_path / "engine_case"
    run_dir.mkdir(parents=True)
    (run_dir / "engine_stage_manifest.json").write_text(
        json.dumps(
            {
                "profile": "closed_valve_ignition_v1",
                "stages": [
                    {"name": "settle_flow", "ok": True, "stage_result": ""},
                    {"name": "chemistry_seed", "ok": True, "stage_result": ""},
                    {"name": "chemistry_spinup", "ok": True, "stage_result": ""},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = pipeline.write_engine_stage_resume_summary(
        run_dir,
        base_params={
            "engine_stage_profile": "closed_valve_ignition_v1",
            "engine_proof_mode": "reacting_staged_ignition",
            "engine_start_angle_deg": -10.0,
            "engine_end_angle_deg": 0.0,
            "rpm": 1800.0,
            "solver_name": "larrakEngineFoam",
        },
        results=[{"name": "chemistry_spinup", "ok": True, "stage_result": ""}],
        docker_timeout_s=86400,
    )
    assert summary["remaining_stages"] == ["ignition_release", "early_burn"]
    assert summary["current_stage"] == "ignition_release"

    persisted = json.loads(
        (run_dir / "engine_stage_resume_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["remaining_stages"] == ["ignition_release", "early_burn"]
    assert persisted["current_stage"] == "ignition_release"
