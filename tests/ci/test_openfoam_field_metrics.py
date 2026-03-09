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
