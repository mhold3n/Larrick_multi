"""Solver-adapter coverage for simulation validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.simulation_validation.cantera_mechanisms import (
    LLNL_DETAILED_GASOLINE_SURROGATE,
    convert_chemkin_to_yaml,
)
from larrak2.simulation_validation.handoff import (
    build_handoff_state_chain,
    build_reduced_state_handoff,
    compute_handoff_conservation,
)
from larrak2.simulation_validation.models import (
    ComparisonMode,
    SourceType,
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationMetricSpec,
)
from larrak2.simulation_validation.solver_adapters import (
    _compute_cantera_flame_speed,
    resolve_simulation_inputs,
)


def _chemistry_dataset() -> ValidationDatasetManifest:
    return ValidationDatasetManifest(
        dataset_id="chem_fixture",
        regime="chemistry",
        fuel_family="gasoline",
        source_type=SourceType.MEASURED,
        metrics=[
            ValidationMetricSpec(
                metric_id="ignition_delay_796K_15bar_phi0p66",
                units="ms",
                comparison_mode=ComparisonMode.ABSOLUTE,
                tolerance_band=15.0,
                source_type=SourceType.MEASURED,
            ),
            ValidationMetricSpec(
                metric_id="laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane",
                units="m/s",
                comparison_mode=ComparisonMode.ABSOLUTE,
                tolerance_band=0.05,
                source_type=SourceType.MEASURED,
            ),
        ],
    )


def test_native_cantera_backend_fails_fast_when_runtime_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: None if name == "cantera" else object()
    )

    dataset = _chemistry_dataset()
    case_spec = ValidationCaseSpec(
        case_id="chem_native",
        regime="chemistry",
        solver_config={
            "simulation_adapter": {
                "kind": "chemistry",
                "backend": "native_cantera",
                "mechanism_file": "mechanisms/iso_octane/llnl_2022.yaml",
                "metrics": {
                    "ignition_delay_796K_15bar_phi0p66": {
                        "method": "ignition_delay",
                        "temperature_K": 796.0,
                        "pressure_bar": 15.0,
                        "equivalence_ratio": 0.66,
                    },
                    "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane": {
                        "method": "flame_speed",
                        "unburned_temperature_K": 353.0,
                        "pressure_bar": 3.33,
                        "equivalence_ratio": 1.0,
                    },
                },
            }
        },
    )

    with pytest.raises(RuntimeError, match="Install the optional combustion extra"):
        resolve_simulation_inputs("chemistry", dataset, case_spec, {})


def test_fixture_chemistry_adapter_populates_missing_metric_values() -> None:
    dataset = _chemistry_dataset()
    case_spec = ValidationCaseSpec(
        case_id="chem_fixture",
        regime="chemistry",
        solver_config={
            "simulation_adapter": {
                "kind": "chemistry",
                "backend": "fixture",
                "fixture_results_path": "data/simulation_validation/gas_combustion_chemistry_fixture_results.json",
            }
        },
    )

    resolved = resolve_simulation_inputs(
        "chemistry",
        dataset,
        case_spec,
        {
            "ignition_delay_796K_15bar_phi0p66_measured": 109.0,
            "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane_measured": 0.503414,
        },
    )

    assert resolved.simulation_data["ignition_delay_796K_15bar_phi0p66"] == pytest.approx(109.0)
    assert resolved.simulation_data[
        "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane"
    ] == pytest.approx(0.503414)
    assert "mechanism_provenance" in resolved.simulation_data


def test_auto_chemistry_adapter_falls_back_when_cantera_run_fails(monkeypatch) -> None:
    dataset = _chemistry_dataset()
    case_spec = ValidationCaseSpec(
        case_id="chem_auto_fallback",
        regime="chemistry",
        solver_config={
            "simulation_adapter": {
                "kind": "chemistry",
                "backend": "auto",
                "fixture_results_path": "data/simulation_validation/gas_combustion_chemistry_fixture_results.json",
                "mechanism_file": "mechanisms/iso_octane/llnl_2022.yaml",
                "metrics": {
                    "ignition_delay_796K_15bar_phi0p66": {
                        "method": "ignition_delay",
                        "temperature_K": 796.0,
                        "pressure_bar": 15.0,
                        "equivalence_ratio": 0.66,
                    },
                    "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane": {
                        "method": "flame_speed",
                        "unburned_temperature_K": 353.0,
                        "pressure_bar": 3.33,
                        "equivalence_ratio": 1.0,
                    },
                },
            }
        },
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic cantera failure")

    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._load_cantera",
        lambda: object(),
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._prepare_cantera_mechanism",
        lambda mechanism_file, adapter_cfg: mechanism_file,
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._compute_cantera_ignition_delay",
        _boom,
    )

    resolved = resolve_simulation_inputs(
        "chemistry",
        dataset,
        case_spec,
        {
            "ignition_delay_796K_15bar_phi0p66_measured": 109.0,
            "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane_measured": 0.503414,
        },
    )

    assert resolved.simulation_data["ignition_delay_796K_15bar_phi0p66"] == pytest.approx(109.0)
    assert any("fell back to fixture" in msg for msg in resolved.messages)


def test_chemistry_adapter_prefers_offline_cache(monkeypatch, tmp_path: Path) -> None:
    dataset = _chemistry_dataset()
    cache_path = tmp_path / "chemistry_offline_results.json"
    cache_path.write_text(
        """
{
  "ignition_delay_796K_15bar_phi0p66": 123.4,
  "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane": 0.456,
  "mechanism_provenance": {
    "mechanism_file": "mechanisms/iso_octane/llnl_2022.yaml"
  }
}
        """.strip(),
        encoding="utf-8",
    )
    case_spec = ValidationCaseSpec(
        case_id="chem_offline_cache",
        regime="chemistry",
        solver_config={
            "simulation_adapter": {
                "kind": "chemistry",
                "backend": "native_cantera",
                "offline_results_path": str(cache_path),
                "mechanism_file": "mechanisms/iso_octane/llnl_2022.yaml",
            }
        },
    )

    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._load_cantera",
        lambda: (_ for _ in ()).throw(AssertionError("live Cantera should not be used")),
    )

    resolved = resolve_simulation_inputs(
        "chemistry",
        dataset,
        case_spec,
        {
            "ignition_delay_796K_15bar_phi0p66_measured": 109.0,
            "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane_measured": 0.503414,
        },
    )

    assert resolved.simulation_data["ignition_delay_796K_15bar_phi0p66"] == pytest.approx(123.4)
    assert resolved.simulation_data[
        "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane"
    ] == pytest.approx(0.456)
    assert resolved.solver_artifacts["chemistry_offline_results"] == str(cache_path)
    assert any("offline cache" in msg for msg in resolved.messages)


def test_chemistry_adapter_uses_fixture_when_offline_cache_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = _chemistry_dataset()
    case_spec = ValidationCaseSpec(
        case_id="chem_offline_cache_missing",
        regime="chemistry",
        solver_config={
            "simulation_adapter": {
                "kind": "chemistry",
                "backend": "auto",
                "fixture_results_path": "data/simulation_validation/gas_combustion_chemistry_fixture_results.json",
                "offline_results_path": str(tmp_path / "missing_cache.json"),
                "offline_results_only": True,
                "mechanism_file": "Docs/Detailed gasoline surrogate/ChemDetailed.inp.txt",
            }
        },
    )

    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._load_cantera",
        lambda: (_ for _ in ()).throw(AssertionError("live Cantera should not be used")),
    )

    resolved = resolve_simulation_inputs(
        "chemistry",
        dataset,
        case_spec,
        {
            "ignition_delay_796K_15bar_phi0p66_measured": 109.0,
            "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane_measured": 0.503414,
        },
    )

    assert resolved.simulation_data["ignition_delay_796K_15bar_phi0p66"] == pytest.approx(109.0)
    assert any("Offline chemistry cache missing" in msg for msg in resolved.messages)


def test_chemistry_adapter_writes_offline_cache_after_live_compute(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = _chemistry_dataset()
    cache_path = tmp_path / "chemistry_offline_results.json"
    case_spec = ValidationCaseSpec(
        case_id="chem_write_offline_cache",
        regime="chemistry",
        solver_config={
            "simulation_adapter": {
                "kind": "chemistry",
                "backend": "native_cantera",
                "offline_results_path": str(cache_path),
                "mechanism_file": "Docs/Detailed gasoline surrogate/ChemDetailed.inp.txt",
                "mechanism_format": "chemkin",
                "thermo_file": "Docs/Detailed gasoline surrogate/gasoline_surrogate_therm.dat.txt",
                "transport_file": "Docs/Detailed gasoline surrogate/gasoline_surrogate_transport.txt",
                "generated_yaml_path": str(tmp_path / "llnl_2022.yaml"),
                "metrics": {
                    "ignition_delay_796K_15bar_phi0p66": {
                        "method": "ignition_delay",
                        "temperature_K": 796.0,
                        "pressure_bar": 15.0,
                        "equivalence_ratio": 0.66,
                    },
                    "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane": {
                        "method": "flame_speed",
                        "unburned_temperature_K": 353.0,
                        "pressure_bar": 3.33,
                        "equivalence_ratio": 1.0,
                    },
                },
            }
        },
    )

    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._load_cantera",
        lambda: object(),
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._prepare_cantera_mechanism",
        lambda mechanism_file, adapter_cfg: str(tmp_path / "llnl_2022.yaml"),
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._compute_cantera_ignition_delay",
        lambda *args, **kwargs: 111.0,
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._compute_cantera_flame_speed",
        lambda *args, **kwargs: 0.501,
    )

    resolved = resolve_simulation_inputs(
        "chemistry",
        dataset,
        case_spec,
        {
            "ignition_delay_796K_15bar_phi0p66_measured": 109.0,
            "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane_measured": 0.503414,
        },
    )

    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["ignition_delay_796K_15bar_phi0p66"] == pytest.approx(111.0)
    assert cached["laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane"] == pytest.approx(0.501)
    assert cached["chemistry_cache_metadata"]["case_id"] == "chem_write_offline_cache"
    assert resolved.solver_artifacts["chemistry_offline_results"] == str(cache_path)
    assert any("Wrote chemistry offline cache" in msg for msg in resolved.messages)


def test_chemistry_adapter_allows_metric_specific_reduced_mechanism_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = _chemistry_dataset()
    cache_path = tmp_path / "chemistry_offline_results.json"
    prepare_calls: list[tuple[str, str]] = []
    case_spec = ValidationCaseSpec(
        case_id="chem_metric_override",
        regime="chemistry",
        solver_config={
            "simulation_adapter": {
                "kind": "chemistry",
                "backend": "native_cantera",
                "offline_results_path": str(cache_path),
                "mechanism_file": "Docs/Detailed gasoline surrogate/ChemDetailed.inp.txt",
                "mechanism_format": "chemkin",
                "thermo_file": "Docs/Detailed gasoline surrogate/gasoline_surrogate_therm.dat.txt",
                "transport_file": "Docs/Detailed gasoline surrogate/gasoline_surrogate_transport.txt",
                "generated_yaml_path": str(tmp_path / "llnl_detailed.yaml"),
                "metrics": {
                    "ignition_delay_796K_15bar_phi0p66": {
                        "method": "ignition_delay",
                        "temperature_K": 796.0,
                        "pressure_bar": 15.0,
                        "equivalence_ratio": 0.66,
                    },
                    "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane": {
                        "method": "flame_speed",
                        "mechanism_file": "Docs/Detailed gasoline surrogate/Reduced/Chem323.inp.txt",
                        "mechanism_format": "chemkin",
                        "thermo_file": "Docs/Detailed gasoline surrogate/gasoline_surrogate_therm.dat.txt",
                        "transport_file": "Docs/Detailed gasoline surrogate/gasoline_surrogate_transport.txt",
                        "generated_yaml_path": str(tmp_path / "chem323_reduced.yaml"),
                        "unburned_temperature_K": 353.0,
                        "pressure_bar": 3.33,
                        "equivalence_ratio": 1.0,
                        "grid_points": 5,
                        "staged_energy": [False, True],
                    },
                },
            }
        },
    )

    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._load_cantera",
        lambda: object(),
    )

    def _fake_prepare(mechanism_file: str, adapter_cfg: dict[str, object]) -> str:
        prepare_calls.append(
            (
                mechanism_file,
                str(adapter_cfg.get("generated_yaml_path", "")),
            )
        )
        return str(tmp_path / f"{Path(mechanism_file).stem}.yaml")

    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._prepare_cantera_mechanism",
        _fake_prepare,
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._compute_cantera_ignition_delay",
        lambda *args, **kwargs: 111.0,
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters._compute_cantera_flame_speed",
        lambda *args, **kwargs: 0.489,
    )

    resolved = resolve_simulation_inputs(
        "chemistry",
        dataset,
        case_spec,
        {
            "ignition_delay_796K_15bar_phi0p66_measured": 109.0,
            "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane_measured": 0.503414,
        },
    )

    assert any(call[0].endswith("ChemDetailed.inp.txt") for call in prepare_calls)
    assert any(call[0].endswith("Chem323.inp.txt") for call in prepare_calls)
    metric_provenance = resolved.simulation_data["metric_mechanism_provenance"]
    assert metric_provenance["ignition_delay_796K_15bar_phi0p66"]["mechanism_file"].endswith(
        "ChemDetailed.inp.txt"
    )
    assert metric_provenance["laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane"][
        "mechanism_file"
    ].endswith("Chem323.inp.txt")

    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["metric_mechanism_provenance"][
        "laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane"
    ]["mechanism_file"].endswith("Chem323.inp.txt")


def test_convert_chemkin_to_yaml_sanitizes_llnl_source_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mechanism_file = tmp_path / "chem.inp.txt"
    thermo_file = tmp_path / "therm.dat.txt"
    transport_file = tmp_path / "tran.dat.txt"
    output_file = tmp_path / "llnl_2022.yaml"

    mechanism_file.write_text(
        "\n".join(
            [
                "SPECIES",
                "IC8H18 C5H81OOH4-5O2 O2",
                "END",
                "A + B => C   1.23+10 0.0 10.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    thermo_file.write_text(
        "\n".join(
            [
                "THERMO ALL",
                "300.0 1000.0 5000.0",
                "C5H81OOH4-5O2                                        1",
                "  1.0 2.0 3.0",
                "  4.0 5.0 6.0",
                "  7.0 8.0 9.0",
                "IC8H18                                              1",
                "  1.0 2.0 3.0",
                "  4.0 5.0 6.0",
                "  7.0 8.0 9.0",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    transport_file.write_text(
        "IC8H18 2.0 1.0 100.0 0.0 0.0\nC5H81OOH4-5O2 2.0 1.0 100.0 0.0 0.0\n",
        encoding="utf-8",
    )

    captured: dict[str, str | bool | None] = {}

    class _FakeCk2Yaml:
        @staticmethod
        def convert(
            *,
            input_file: str,
            thermo_file: str,
            transport_file: str | None,
            out_name: str,
            phase_name: str,
            permissive: bool,
            quiet: bool,
        ) -> None:
            captured["input"] = Path(input_file).read_text(encoding="utf-8")
            captured["thermo"] = Path(thermo_file).read_text(encoding="utf-8")
            captured["transport"] = (
                Path(transport_file).read_text(encoding="utf-8")
                if transport_file is not None
                else None
            )
            captured["phase_name"] = phase_name
            captured["permissive"] = permissive
            captured["quiet"] = quiet
            Path(out_name).write_text("phases: []\n", encoding="utf-8")

    monkeypatch.setattr(
        "larrak2.simulation_validation.cantera_mechanisms.importlib.import_module",
        lambda name: _FakeCk2Yaml,
    )

    converted = convert_chemkin_to_yaml(
        input_file=mechanism_file,
        thermo_file=thermo_file,
        transport_file=transport_file,
        output_file=output_file,
        permissive=True,
        quiet=False,
        sanitizer_profile=LLNL_DETAILED_GASOLINE_SURROGATE,
    )

    assert converted == output_file
    assert output_file.exists()
    assert "C5H81OOH4-5O2" not in str(captured["input"])
    assert "1.23E+10" in str(captured["input"])
    assert "C5H81OOH4-5O2" not in str(captured["thermo"])
    assert str(captured["thermo"]).splitlines()[-2].endswith("4")
    assert "C5H81OOH4-5O2" not in str(captured["transport"])
    assert captured["phase_name"] == "gas"
    assert captured["permissive"] is True
    assert captured["quiet"] is False


def test_compute_cantera_flame_speed_supports_fixed_grid_and_solution_profile(
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    class _FakeGas:
        def __init__(self) -> None:
            self.tp = None
            self.eq = None

        @property
        def TP(self):
            return self.tp

        @TP.setter
        def TP(self, value):
            self.tp = value

        def set_equivalence_ratio(self, phi, fuel, oxidizer) -> None:
            self.eq = (phi, fuel, oxidizer)

    class _FakeFlame:
        def __init__(self, gas, *, grid=None, width=None) -> None:
            calls["grid"] = list(grid) if grid is not None else None
            calls["width"] = width
            self.transport_model = None
            self.flame = object()
            self.energy_enabled = True
            self.grid = [0.0, 0.005, 0.01]
            self.T = [353.0, 1200.0, 1800.0]
            self.velocity = [0.321]

        def set_max_grid_points(self, domain, value) -> None:
            calls["max_grid_points"] = value

        def set_refine_criteria(self, **kwargs) -> None:
            calls["refine_criteria"] = kwargs

        def restore(self, filename, name="solution", loglevel=0) -> None:
            calls["restore"] = (filename, name, loglevel)

        def solve(self, loglevel=0, refine_grid=True, auto=False, stage=None) -> None:
            solve_calls = list(calls.get("solve_calls", []) or [])
            solve_calls.append(
                {
                    "loglevel": loglevel,
                    "refine_grid": refine_grid,
                    "auto": auto,
                    "energy_enabled": self.energy_enabled,
                }
            )
            calls["solve_calls"] = solve_calls

        def save(
            self, filename, name="solution", description=None, overwrite=False, **kwargs
        ) -> None:
            Path(filename).write_text("solution\n", encoding="utf-8")
            calls["save"] = (filename, name, description, overwrite)

    class _FakeCt:
        def Solution(self, mechanism_file, transport_model=None):
            calls["solution"] = (mechanism_file, transport_model)
            return _FakeGas()

        def FreeFlame(self, gas, grid=None, width=None):
            return _FakeFlame(gas, grid=grid, width=width)

    restore_path = tmp_path / "restore.yaml"
    restore_path.write_text("existing\n", encoding="utf-8")
    save_path = tmp_path / "saved.yaml"

    value = _compute_cantera_flame_speed(
        _FakeCt(),
        mechanism_file="mechanism.yaml",
        fuel="IC8H18",
        oxidizer={"O2": 0.2033, "N2": 0.7859},
        metric_cfg={
            "unburned_temperature_K": 353.0,
            "pressure_bar": 3.33,
            "equivalence_ratio": 1.0,
            "transport_model": "mixture-averaged",
            "width_m": 0.01,
            "grid_points": 5,
            "refine_grid": False,
            "max_grid_points": 7,
            "staged_energy": [False, True],
            "solution_profile_path": str(save_path),
            "restore_solution_path": str(restore_path),
            "solution_name": "fixed_grid",
        },
    )

    assert value == pytest.approx(0.321)
    assert calls["solution"] == ("mechanism.yaml", "mixture-averaged")
    assert calls["max_grid_points"] == 7
    assert calls["restore"] == (str(restore_path), "fixed_grid", 0)
    assert Path(save_path).exists()
    assert calls["save"] == (
        str(save_path),
        "fixed_grid",
        "Cached Cantera flame-speed solution",
        True,
    )
    assert [item["refine_grid"] for item in calls["solve_calls"]] == [False, False]
    assert [item["energy_enabled"] for item in calls["solve_calls"]] == [False, True]


def test_handoff_bundle_validation_and_conservation() -> None:
    bundle = build_reduced_state_handoff(
        {
            "bundle_id": "gc_bundle",
            "mechanism_id": "llnl_iso_octane_2022",
            "fuel_name": "iso-octane",
            "pressure_Pa": 1.5e6,
            "temperature_K": 980.0,
            "species_mole_fractions": {"IC8H18": 0.02, "O2": 0.21, "N2": 0.77},
            "vapor_fraction": 0.8,
            "mixture_homogeneity_index": 0.9,
            "velocity_m_s": 40.0,
            "turbulence_intensity": 0.12,
            "stage_marker": "chemistry_exit",
            "cycle_coordinate_deg": 350.0,
            "total_mass_kg": 4.1e-4,
            "total_energy_J": 366.54,
        }
    )
    assert bundle.validate() == []

    state_chain = build_handoff_state_chain(
        base_state=bundle.to_dict(),
        state_overrides=[
            {"stage_marker": "chemistry_exit"},
            {"stage_marker": "closed_cylinder_exit"},
            {"stage_marker": "reacting_flow_entry"},
        ],
    )
    conservation = compute_handoff_conservation(
        state_chain,
        mass_tolerance=1.0e-6,
        energy_tolerance=1.0e-3,
    )

    assert conservation["state_conservation_mass"] == pytest.approx(0.0)
    assert conservation["state_conservation_energy"] == pytest.approx(0.0)
    assert len(conservation["handoff_states"]) == 2
