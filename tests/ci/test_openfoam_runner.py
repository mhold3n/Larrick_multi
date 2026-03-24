from __future__ import annotations

from pathlib import Path

from larrak2.adapters.openfoam import OpenFoamRunner


def test_setup_case_with_assets_sanitizes_chemkin_transport_comments(tmp_path: Path) -> None:
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "controlDict").write_text("placeholder\n", encoding="utf-8")

    source = tmp_path / "transport.txt"
    source.write_text(
        "SPEC1 1 2 3 4 5 6 ! inline comment\n! full-line comment\nSPEC2 7 8 9 10 11 12\n",
        encoding="utf-8",
    )

    runner = OpenFoamRunner(template_dir=template_dir)
    run_dir = tmp_path / "run"
    runner.setup_case_with_assets(
        run_dir,
        {},
        staged_inputs=[
            {
                "source": str(source),
                "target": "chemkin/transportProperties",
                "sanitizer": "strip_chemkin_comments",
            }
        ],
    )

    staged = (run_dir / "chemkin" / "transportProperties").read_text(encoding="utf-8")
    assert staged == "SPEC1 1 2 3 4 5 6\nSPEC2 7 8 9 10 11 12\n"


def test_setup_case_with_assets_uses_llnl_gasoline_sanitizer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "controlDict").write_text("placeholder\n", encoding="utf-8")

    source = tmp_path / "chem.inp"
    source.write_text("raw\n", encoding="utf-8")

    calls: list[tuple[str, str]] = []

    def _fake_sanitize(*, source_file: Path, file_kind: str, profile: str = "") -> str:
        calls.append((source_file.name, file_kind))
        assert profile == "llnl_detailed_gasoline_surrogate"
        return "sanitized\n"

    monkeypatch.setattr("larrak2.adapters.openfoam.sanitize_chemkin_file_text", _fake_sanitize)

    runner = OpenFoamRunner(template_dir=template_dir)
    run_dir = tmp_path / "run"
    runner.setup_case_with_assets(
        run_dir,
        {},
        staged_inputs=[
            {
                "source": str(source),
                "target": "chemkin/chem.inp",
                "sanitizer": "llnl_gasoline_input",
            }
        ],
    )

    assert calls == [("chem.inp", "input")]
    assert (run_dir / "chemkin" / "chem.inp").read_text(encoding="utf-8") == "sanitized\n"


def test_clear_validation_outputs_can_purge_numeric_time_dirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "0").mkdir(parents=True)
    (run_dir / "0.0001").mkdir()
    (run_dir / "0.0002").mkdir()
    (run_dir / "postProcessing" / "sprayValidation").mkdir(parents=True)
    (run_dir / "openfoam_metrics.json").write_text("{}", encoding="utf-8")

    OpenFoamRunner.clear_validation_outputs(
        run_dir,
        sample_root="postProcessing/sprayValidation",
        purge_numeric_time_dirs=True,
    )

    assert (run_dir / "0").is_dir()
    assert not (run_dir / "0.0001").exists()
    assert not (run_dir / "0.0002").exists()
    assert not (run_dir / "postProcessing" / "sprayValidation").exists()
    assert not (run_dir / "openfoam_metrics.json").exists()


def test_repair_ami_boundary_values_replaces_zeroed_species_patches(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    zero_dir = run_dir / "0"
    zero_dir.mkdir(parents=True)
    field_path = zero_dir / "O2"
    field_path.write_text(
        "\n".join(
            [
                "FoamFile",
                "{",
                "    class       volScalarField;",
                "}",
                "",
                "internalField   uniform 0.216416;",
                "",
                "boundaryField",
                "{",
                "    inlet",
                "    {",
                "        type            fixedValue;",
                "        value           uniform 0.216416;",
                "    }",
                "    AMI_intake_master",
                "    {",
                "        type            cyclicAMI;",
                "        value           uniform 0;",
                "    }",
                "}",
                "",
            ],
        ),
        encoding="utf-8",
    )

    updated = OpenFoamRunner.repair_ami_boundary_values(run_dir, time_dir="0")

    assert updated == ["O2"]
    repaired = field_path.read_text(encoding="utf-8")
    assert "value           uniform 0.216416;" in repaired
