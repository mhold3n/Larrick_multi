from __future__ import annotations

import json
from pathlib import Path

from larrak2.simulation_validation.openfoam_chemistry_package import (
    build_openfoam_chemistry_package_from_spec,
)


class _FakeSpecies:
    def __init__(self, name: str, composition: dict[str, float], thermo: dict, transport: dict):
        self.name = name
        self.composition = composition
        self.input_data = {
            "name": name,
            "composition": composition,
            "thermo": thermo,
            "transport": transport,
        }


class _FakeReaction:
    def __init__(
        self,
        *,
        reaction_type: str,
        reversible: bool,
        reactants: dict[str, float],
        products: dict[str, float],
        input_data: dict,
    ):
        self.reaction_type = reaction_type
        self.reversible = reversible
        self.reactants = reactants
        self.products = products
        self.input_data = input_data
        self.equation = input_data.get("equation", "")
        self.third_body = type(
            "_ThirdBody",
            (),
            {"default_efficiency": float(input_data.get("default-efficiency", 1.0))},
        )()


class _FakeGas:
    def __init__(self):
        nasa = {
            "model": "NASA7",
            "temperature-ranges": [300.0, 1000.0, 5000.0],
            "data": [
                [3.0, 0.1, 0.01, 0.001, 0.0001, -1000.0, 1.0],
                [4.0, 0.2, 0.02, 0.002, 0.0002, -1200.0, 2.0],
            ],
        }
        transport = {
            "model": "gas",
            "geometry": "linear",
            "diameter": 3.5,
            "well-depth": 100.0,
            "rotational-relaxation": 1.0,
        }
        self._species = [
            _FakeSpecies("O2", {"O": 2}, nasa, transport),
            _FakeSpecies("N2", {"N": 2}, nasa, transport),
            _FakeSpecies("IC8H18", {"C": 8, "H": 18}, nasa, transport),
            _FakeSpecies("C3H51-2,3OOH", {"C": 3, "H": 5, "O": 3}, nasa, transport),
        ]
        self.species_names = [species.name for species in self._species]
        self.element_names = ["O", "N", "C", "H"]
        self.molecular_weights = [31.998, 28.0134, 114.232, 89.0]
        self._species_index = {name: index for index, name in enumerate(self.species_names)}
        self._reactions = [
            _FakeReaction(
                reaction_type="Arrhenius",
                reversible=False,
                reactants={"IC8H18": 1, "O2": 1},
                products={"C3H51-2,3OOH": 1},
                input_data={
                    "equation": "IC8H18 + O2 => C3H51-2,3OOH",
                    "rate-constant": {"A": 1.2e12, "b": 0.5, "Ea": 5.0e7},
                },
            ),
            _FakeReaction(
                reaction_type="three-body-Arrhenius",
                reversible=False,
                reactants={"O2": 1},
                products={"O2": 1},
                input_data={
                    "equation": "O2 + M => O2 + M",
                    "rate-constant": {"A": 2.5e9, "b": 0.0, "Ea": 1.0e7},
                    "efficiencies": {"O2": 2.0, "N2": 0.5},
                    "default-efficiency": 1.0,
                },
            ),
            _FakeReaction(
                reaction_type="falloff-Troe",
                reversible=True,
                reactants={"IC8H18": 1},
                products={"IC8H18": 1},
                input_data={
                    "equation": "IC8H18 (+M) <=> IC8H18 (+M)",
                    "low-P-rate-constant": {"A": 3.0e10, "b": -1.0, "Ea": 2.0e7},
                    "high-P-rate-constant": {"A": 6.0e11, "b": 0.1, "Ea": 1.0e7},
                    "efficiencies": {"IC8H18": 3.0},
                    "default-efficiency": 1.0,
                    "Troe": {"A": 0.7, "T3": 1.0, "T1": 1000.0, "T2": 5000.0},
                },
            ),
        ]
        self.n_species = len(self._species)
        self.n_reactions = len(self._reactions)
        self._current_temperature = 300.0
        self._current_species = "O2"

    @property
    def TPX(self):  # pragma: no cover - convenience only
        return self._current_temperature, 101325.0, {self._current_species: 1.0}

    @TPX.setter
    def TPX(self, value):
        self._current_temperature = float(value[0])
        composition = dict(value[2])
        self._current_species = next(iter(composition.keys()))

    @property
    def viscosity(self) -> float:
        species_factor = 1.0 + self._species_index[self._current_species] * 0.1
        return 1.0e-5 * species_factor * (self._current_temperature / 300.0) ** 0.7

    def species_index(self, name: str) -> int:
        return self._species_index[name]

    def species(self, name: str) -> _FakeSpecies:
        return self._species[self._species_index[name]]

    def reaction(self, index: int) -> _FakeReaction:
        return self._reactions[index]


def test_build_openfoam_chemistry_package_from_spec_writes_manifest_and_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mechanism_file = tmp_path / "Chem323.inp"
    thermo_file = tmp_path / "therm.dat"
    transport_file = tmp_path / "tran.dat"
    mechanism_file.write_text("mechanism\n", encoding="utf-8")
    thermo_file.write_text("thermo\n", encoding="utf-8")
    transport_file.write_text("transport\n", encoding="utf-8")

    fake_gas = _FakeGas()

    def _fake_convert(**kwargs):
        output_file = Path(kwargs["output_file"])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("phases:\n- name: gas\n", encoding="utf-8")
        return output_file

    class _FakeCanteraModule:
        def Solution(self, path: str):
            assert path.endswith(".yaml")
            return fake_gas

    monkeypatch.setattr(
        "larrak2.simulation_validation.openfoam_chemistry_package.convert_chemkin_to_yaml",
        _fake_convert,
    )
    monkeypatch.setattr(
        "larrak2.simulation_validation.openfoam_chemistry_package._load_cantera",
        lambda: _FakeCanteraModule(),
    )

    output_dir = tmp_path / "package"
    manifest = build_openfoam_chemistry_package_from_spec(
        {
            "package_id": "chem323_reduced_v2512",
            "mechanism_file": str(mechanism_file),
            "thermo_file": str(thermo_file),
            "transport_file": str(transport_file),
            "generated_yaml_path": str(tmp_path / "chem323_reduced.yaml"),
            "output_dir": str(output_dir),
            "openfoam_version": "2512",
            "fuel_species": "IC8H18",
        },
        refresh=True,
    )

    assert manifest["package_id"] == "chem323_reduced_v2512"
    assert manifest["openfoam_version"] == "2512"
    assert manifest["fuel_species"] == "IC8H18"
    assert manifest["species_count"] == 4
    assert manifest["reaction_count"] == 3
    assert Path(manifest["files"]["reactions"]).exists()
    assert Path(manifest["files"]["thermo.compressibleGas"]).exists()
    assert Path(manifest["files"]["transportProperties"]).exists()

    reactions_text = (output_dir / "reactions").read_text(encoding="utf-8")
    thermo_text = (output_dir / "thermo.compressibleGas").read_text(encoding="utf-8")
    assert 'reaction        "IC8H18 + O2 = C3H51-2,3OOH";' in reactions_text
    assert "irreversiblethirdBodyArrheniusReaction" in reactions_text
    assert "reversibleArrheniusTroeFallOffReaction" in reactions_text
    assert '"C3H51-2,3OOH"' in reactions_text
    assert "foamChemistryThermoFile" not in thermo_text
    assert '"C3H51-2,3OOH"' in thermo_text

    manifest_on_disk = json.loads(
        (output_dir / "package_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_on_disk["package_hash"] == manifest["package_hash"]
