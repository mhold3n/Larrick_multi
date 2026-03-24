from pathlib import Path

import pytest

from larrak2.adapters.openfoam import OpenFoamRunner


def _write_scalar_field(path: Path, name: str, values: list[float], location: str) -> None:
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
                "internalField   nonuniform List<scalar> ",
                str(len(values)),
                "(",
                *[str(v) for v in values],
                ")",
                "",
            ]
        )
    )


def _write_uniform_scalar_field(
    path: Path,
    *,
    name: str,
    value: float,
    location: str,
    field_class: str = "volScalarField",
) -> None:
    path.write_text(
        "\n".join(
            [
                "FoamFile",
                "{",
                "    format      ascii;",
                f"    class       {field_class};",
                f'    location    "{location}";',
                f"    object      {name};",
                "}",
                "",
                "dimensions      [0 0 0 0 0 0 0];",
                "",
                f"internalField   uniform {value};",
                "",
            ]
        )
    )


def _write_uniform_vector_field(
    path: Path, *, name: str, value: tuple[float, float, float], location: str
) -> None:
    path.write_text(
        "\n".join(
            [
                "FoamFile",
                "{",
                "    format      ascii;",
                "    class       volVectorField;",
                f'    location    "{location}";',
                f"    object      {name};",
                "}",
                "",
                "dimensions      [0 1 -1 0 0 0 0];",
                "",
                f"internalField   uniform ({value[0]} {value[1]} {value[2]});",
                "",
            ]
        )
    )


def test_openfoam_field_metrics_and_sidecar_parse(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    zero_dir = run_dir / "0"
    latest_dir = run_dir / "0.0003"
    zero_dir.mkdir(parents=True)
    latest_dir.mkdir(parents=True)

    (run_dir / "solver.log").write_text("End\n")
    (zero_dir / "T").write_text(
        "\n".join(
            [
                "FoamFile",
                "{",
                "    format      ascii;",
                "    class       volScalarField;",
                '    location    "0";',
                "    object      T;",
                "}",
                "",
                "dimensions      [0 0 0 1 0 0 0];",
                "",
                "internalField   uniform 300;",
                "",
            ]
        )
    )

    _write_scalar_field(latest_dir / "rho", "rho", [1.0, 2.0, 3.0], "0.0003")
    _write_scalar_field(latest_dir / "T", "T", [300.0, 400.0, 500.0], "0.0003")
    _write_scalar_field(latest_dir / "Vc", "Vc", [0.1, 0.2, 0.3], "0.0003")

    runner = OpenFoamRunner(Path("dummy"))
    fresh_mass_reference = 2.0
    p_manifold = fresh_mass_reference * 287.05 * 300.0 / 0.6
    metrics = runner.compute_field_metrics(run_dir, p_manifold_Pa=p_manifold)

    assert metrics["trapped_mass"] == pytest.approx(1.4)
    assert metrics["domain_volume_m3"] == pytest.approx(0.6)
    assert metrics["mass_weighted_temperature_K"] == pytest.approx((30 + 160 + 450) / 1.4)
    assert metrics["fresh_charge_fraction"] == pytest.approx(0.7)
    assert metrics["residual_fraction"] == pytest.approx(0.3)
    assert metrics["scavenging_efficiency"] == pytest.approx(0.7)
    assert metrics["trapped_o2_mass"] == pytest.approx(1.4 * 0.233 * 0.7)
    assert metrics["metric_source"] == "field_postprocess_mass_reference_v1"

    runner.emit_metrics(run_dir, metrics, log_name="solver.log")
    parsed = runner.parse_results(run_dir, log_name="solver.log")

    assert parsed["trapped_mass"] == pytest.approx(1.4)
    assert parsed["scavenging_efficiency"] == pytest.approx(0.7)
    assert parsed["residual_fraction"] == pytest.approx(0.3)
    assert parsed["trapped_o2_mass"] == pytest.approx(1.4 * 0.233 * 0.7)
    assert parsed["metric_source"] == "field_postprocess_mass_reference_v1"


def test_openfoam_field_metrics_accepts_v2512_volume_field_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_v2512"
    zero_dir = run_dir / "0"
    latest_dir = run_dir / "0.0001"
    zero_dir.mkdir(parents=True)
    latest_dir.mkdir(parents=True)

    (zero_dir / "T").write_text(
        "\n".join(
            [
                "FoamFile",
                "{",
                "    format      ascii;",
                "    class       volScalarField;",
                '    location    "0";',
                "    object      T;",
                "}",
                "",
                "dimensions      [0 0 0 1 0 0 0];",
                "",
                "internalField   uniform 320;",
                "",
            ]
        )
    )

    _write_scalar_field(latest_dir / "rho", "rho", [1.0, 1.0], "0.0001")
    _write_scalar_field(latest_dir / "T", "T", [320.0, 480.0], "0.0001")
    _write_scalar_field(latest_dir / "V", "V", [0.2, 0.3], "0.0001")

    runner = OpenFoamRunner(Path("dummy"))
    metrics = runner.compute_field_metrics(run_dir, p_manifold_Pa=90000.0)

    assert metrics["trapped_mass"] == pytest.approx(0.5)
    assert metrics["domain_volume_m3"] == pytest.approx(0.5)
    assert metrics["metric_time_dir"] == "0.0001"


def test_extract_spray_validation_metrics_from_sampled_artifacts(tmp_path: Path) -> None:
    sample_dir = tmp_path / "postProcessing" / "sprayValidation" / "0.001"
    sample_dir.mkdir(parents=True)
    (sample_dir / "liquidPenetration_mm.dat").write_text("0.001 30.0\n", encoding="utf-8")
    (sample_dir / "vaporSpreadingAngle_deg.dat").write_text("0.001 90.0\n", encoding="utf-8")
    (sample_dir / "dropletSMD_z15mm_um.dat").write_text("0.001 11.0\n", encoding="utf-8")
    (sample_dir / "gasAxialVelocity_z15mm_t1ms.dat").write_text("0.001 1.3531\n", encoding="utf-8")

    metrics = OpenFoamRunner.extract_validation_metrics(
        tmp_path,
        regime_name="spray",
        extractor_cfg={"name": "spray_g_v1"},
    )

    assert metrics["liquid_penetration_max_mm_sprayG"] == pytest.approx(30.0)
    assert metrics["vapor_spreading_angle_deg_sprayG"] == pytest.approx(90.0)
    assert metrics["droplet_smd_um_sprayG_z15mm"] == pytest.approx(11.0)
    assert metrics["gas_axial_velocity_m_s_sprayG_z15mm_t1ms"] == pytest.approx(1.3531)
    assert metrics["metric_authority"] == "live_case_fields"
    assert metrics["sampled_time_dir"] == "0.001"


def test_extract_reacting_validation_metrics_from_sampled_artifacts(tmp_path: Path) -> None:
    sample_dir = tmp_path / "postProcessing" / "reactingValidation" / "0.0003"
    sample_dir.mkdir(parents=True)
    (sample_dir / "temperature_K.dat").write_text("0.0003 1810.0\n", encoding="utf-8")
    (sample_dir / "CO2_molefrac.dat").write_text("0.0003 0.064\n", encoding="utf-8")
    (sample_dir / "OH_molefrac.dat").write_text("0.0003 0.0038\n", encoding="utf-8")
    (sample_dir / "bulkVelocity_m_s.dat").write_text("0.0003 44.0\n", encoding="utf-8")

    metrics = OpenFoamRunner.extract_validation_metrics(
        tmp_path,
        regime_name="reacting_flow",
        extractor_cfg={"name": "reacting_iso_octane_v1"},
    )

    assert metrics["gas_temperature_K_iso_octane_reacting"] == pytest.approx(1810.0)
    assert metrics["CO2_molefrac_iso_octane_reacting"] == pytest.approx(0.064)
    assert metrics["OH_molefrac_iso_octane_reacting"] == pytest.approx(0.0038)
    assert metrics["bulk_velocity_m_s_iso_octane_reacting"] == pytest.approx(44.0)
    assert metrics["metric_authority"] == "live_case_fields"
    assert metrics["sampled_time_dir"] == "0.0003"


def test_generate_live_spray_validation_samples_from_fields(tmp_path: Path) -> None:
    latest_dir = tmp_path / "0.0004"
    system_dir = tmp_path / "system"
    latest_dir.mkdir(parents=True)
    system_dir.mkdir()
    _write_uniform_scalar_field(
        latest_dir / "phi",
        name="phi",
        value=0.003,
        location="0.0004",
        field_class="surfaceScalarField",
    )
    _write_uniform_scalar_field(latest_dir / "T", name="T", value=300.0, location="0.0004")
    _write_uniform_scalar_field(latest_dir / "rho", name="rho", value=1.1, location="0.0004")
    _write_uniform_vector_field(
        latest_dir / "U",
        name="U",
        value=(21.0, 0.0, 0.0),
        location="0.0004",
    )
    (system_dir / "liveValidationSamples.json").write_text(
        """
{
  "sample_root": "postProcessing/sprayValidation",
  "source_time_dir": "latest_time",
  "output_time_dir": "latest_time",
  "metrics": {
    "liquidPenetration_mm.dat": {"field": "phi", "field_kind": "surface_scalar", "statistic": "mean", "scale": 10000.0},
    "vaporSpreadingAngle_deg.dat": {"field": "T", "field_kind": "scalar", "statistic": "mean", "scale": 0.3},
    "dropletSMD_z15mm_um.dat": {"field": "rho", "field_kind": "scalar", "statistic": "mean", "scale": 10.0},
    "gasAxialVelocity_z15mm_t1ms.dat": {"field": "U", "field_kind": "vector", "statistic": "mean_component", "component": "x", "scale": 0.0644333333333}
  }
}
""".strip(),
        encoding="utf-8",
    )

    generated = OpenFoamRunner.generate_live_validation_samples(
        tmp_path,
        regime_name="spray",
        extractor_cfg={"generator_config_path": "system/liveValidationSamples.json"},
    )
    metrics = OpenFoamRunner.extract_validation_metrics(
        tmp_path,
        regime_name="spray",
        extractor_cfg={"name": "spray_g_v1"},
    )

    assert generated["source_time_dir"] == "0.0004"
    assert generated["output_time_dir"] == "0.0004"
    assert metrics["liquid_penetration_max_mm_sprayG"] == pytest.approx(30.0)
    assert metrics["vapor_spreading_angle_deg_sprayG"] == pytest.approx(90.0)
    assert metrics["droplet_smd_um_sprayG_z15mm"] == pytest.approx(11.0)
    assert metrics["gas_axial_velocity_m_s_sprayG_z15mm_t1ms"] == pytest.approx(1.3531)
    assert metrics["sampled_time_dir"] == "0.0004"


def test_generate_live_reacting_validation_samples_from_fields(tmp_path: Path) -> None:
    latest_dir = tmp_path / "0.0004"
    zero_dir = tmp_path / "0"
    system_dir = tmp_path / "system"
    latest_dir.mkdir(parents=True)
    zero_dir.mkdir()
    system_dir.mkdir()
    _write_uniform_scalar_field(latest_dir / "T", name="T", value=465.0, location="0.0004")
    _write_uniform_vector_field(
        latest_dir / "U",
        name="U",
        value=(44.0, 0.0, 0.0),
        location="0.0004",
    )
    _write_uniform_scalar_field(zero_dir / "CO2", name="CO2", value=0.064, location="0")
    _write_uniform_scalar_field(zero_dir / "OH", name="OH", value=0.0038, location="0")
    (system_dir / "liveValidationSamples.json").write_text(
        """
{
  "sample_root": "postProcessing/reactingValidation",
  "source_time_dir": "latest_time",
  "output_time_dir": "latest_time",
  "metrics": {
    "temperature_K.dat": {"field": "T", "field_kind": "scalar", "statistic": "mean", "scale": 3.89247311827957},
    "CO2_molefrac.dat": {"field": "CO2", "field_kind": "scalar", "statistic": "mean", "fallback_time_dir": "0"},
    "OH_molefrac.dat": {"field": "OH", "field_kind": "scalar", "statistic": "mean", "fallback_time_dir": "0"},
    "bulkVelocity_m_s.dat": {"field": "U", "field_kind": "vector", "statistic": "mean_magnitude"}
  }
}
""".strip(),
        encoding="utf-8",
    )

    generated = OpenFoamRunner.generate_live_validation_samples(
        tmp_path,
        regime_name="reacting_flow",
        extractor_cfg={"generator_config_path": "system/liveValidationSamples.json"},
    )
    metrics = OpenFoamRunner.extract_validation_metrics(
        tmp_path,
        regime_name="reacting_flow",
        extractor_cfg={"name": "reacting_iso_octane_v1"},
    )

    assert generated["field_sources"]["CO2_molefrac.dat"] == "0"
    assert generated["field_sources"]["OH_molefrac.dat"] == "0"
    assert metrics["gas_temperature_K_iso_octane_reacting"] == pytest.approx(1810.0)
    assert metrics["CO2_molefrac_iso_octane_reacting"] == pytest.approx(0.064)
    assert metrics["OH_molefrac_iso_octane_reacting"] == pytest.approx(0.0038)
    assert metrics["bulk_velocity_m_s_iso_octane_reacting"] == pytest.approx(44.0)
    assert metrics["sampled_time_dir"] == "0.0004"
