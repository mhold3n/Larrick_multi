"""Build tracked OpenFOAM-native chemistry packages from CHEMKIN/Cantera inputs."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

from .cantera_mechanisms import convert_chemkin_to_yaml

PACKAGE_SCHEMA_VERSION = 1
OPENFOAM_VERSION_DEFAULT = "2512"
DEFAULT_TRANSPORT_AS = 1.67212e-06
DEFAULT_TRANSPORT_TS = 170.672
R_UNIVERSAL_J_PER_KMOL_K = 8314.46261815324
OPENFOAM_PLAIN_WORD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_()+-]*$")


def _load_cantera():
    if importlib.util.find_spec("cantera") is None:
        raise RuntimeError(
            "Cantera runtime is required for OpenFOAM chemistry packaging. "
            "Install the optional combustion extra with `pip install .[combustion]`."
        )
    return importlib.import_module("cantera")


def _load_config(config_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at '{config_path}'")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _format_number(value: float) -> str:
    if math.isfinite(value):
        return f"{value:.12g}"
    raise ValueError(f"Cannot serialize non-finite value '{value}'")


def _format_word(value: str) -> str:
    stripped = str(value).strip()
    if not stripped:
        raise ValueError("OpenFOAM package cannot serialize an empty word")
    if OPENFOAM_PLAIN_WORD_RE.match(stripped):
        return stripped
    escaped = stripped.replace('"', '\\"')
    return f'"{escaped}"'


def _format_coeff_list(values: list[float]) -> str:
    return "( " + " ".join(_format_number(float(value)) for value in values) + " )"


def _format_stoich_term(species_name: str, coefficient: float) -> str:
    coeff = float(coefficient)
    if abs(coeff - round(coeff)) < 1.0e-12:
        rounded = int(round(coeff))
        prefix = "" if rounded == 1 else str(rounded)
    else:
        prefix = _format_number(coeff)
    return f"{prefix}{species_name}"


def _format_reaction_side(side: dict[str, float]) -> str:
    pieces = [
        _format_stoich_term(species_name, coefficient)
        for species_name, coefficient in side.items()
        if float(coefficient) != 0.0
    ]
    if not pieces:
        return "0"
    return " + ".join(pieces)


def _reaction_equation(reaction: Any) -> str:
    return f"{_format_reaction_side(dict(reaction.reactants))} = {_format_reaction_side(dict(reaction.products))}"


def _arrhenius_block_lines(
    *,
    rate_constant: dict[str, Any],
    indent: str = "        ",
) -> list[str]:
    activation_energy = float(rate_constant.get("Ea", 0.0)) / R_UNIVERSAL_J_PER_KMOL_K
    return [
        f"{indent}A               {_format_number(float(rate_constant.get('A', 0.0)))};",
        f"{indent}beta            {_format_number(float(rate_constant.get('b', 0.0)))};",
        f"{indent}Ta              {_format_number(activation_energy)};",
    ]


def _transport_coefficients(gas: Any, species_name: str) -> tuple[float, float]:
    gas.TPX = 300.0, 101325.0, {species_name: 1.0}
    mu_300 = float(gas.viscosity)
    gas.TPX = 1000.0, 101325.0, {species_name: 1.0}
    mu_1000 = float(gas.viscosity)

    if mu_300 <= 0.0 or mu_1000 <= 0.0:
        return DEFAULT_TRANSPORT_AS, DEFAULT_TRANSPORT_TS

    m1 = mu_300 / math.sqrt(300.0)
    m2 = mu_1000 / math.sqrt(1000.0)
    denominator = (m1 / 300.0) - (m2 / 1000.0)
    if abs(denominator) < 1.0e-20:
        return DEFAULT_TRANSPORT_AS, DEFAULT_TRANSPORT_TS
    ts = (m2 - m1) / denominator
    if not math.isfinite(ts) or ts <= -299.0:
        return DEFAULT_TRANSPORT_AS, DEFAULT_TRANSPORT_TS
    a_s = m1 * (1.0 + ts / 300.0)
    if not math.isfinite(a_s) or a_s <= 0.0:
        return DEFAULT_TRANSPORT_AS, DEFAULT_TRANSPORT_TS
    return a_s, ts


def _transport_coefficients_for_species(gas: Any) -> dict[str, tuple[float, float]]:
    coefficients: dict[str, tuple[float, float]] = {}
    for species_name in gas.species_names:
        coefficients[species_name] = _transport_coefficients(gas, species_name)
    return coefficients


def _thermo_lines(gas: Any) -> list[str]:
    transport_coeffs = _transport_coefficients_for_species(gas)
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      thermo.compressibleGas;",
        "}",
        "",
    ]
    for species_name in gas.species_names:
        species = gas.species(species_name)
        thermo = dict(species.input_data.get("thermo", {}) or {})
        temperature_ranges = list(thermo.get("temperature-ranges", []) or [])
        coefficient_sets = list(thermo.get("data", []) or [])
        if len(temperature_ranges) != 3 or len(coefficient_sets) != 2:
            raise ValueError(
                f"Species '{species_name}' does not provide NASA7 thermo data required "
                "for OpenFOAM packaging"
            )
        low_coeffs = [float(value) for value in coefficient_sets[0]]
        high_coeffs = [float(value) for value in coefficient_sets[1]]
        if len(low_coeffs) != 7 or len(high_coeffs) != 7:
            raise ValueError(
                f"Species '{species_name}' NASA7 block must contain 7 coefficients per range"
            )
        composition = dict(species.composition or {})
        a_s, ts = transport_coeffs[species_name]
        mol_weight = float(gas.molecular_weights[gas.species_index(species_name)])
        lines.extend(
            [
                f"{_format_word(species_name)}",
                "{",
                "    specie",
                "    {",
                f"        molWeight       {_format_number(mol_weight)};",
                "    }",
                "    elements",
                "    {",
            ]
        )
        for element_name, count in composition.items():
            numeric_count = float(count)
            if abs(numeric_count - round(numeric_count)) < 1.0e-12:
                formatted_count = str(int(round(numeric_count)))
            else:
                formatted_count = _format_number(numeric_count)
            lines.append(f"        {_format_word(str(element_name))}       {formatted_count};")
        lines.extend(
            [
                "    }",
                "    thermodynamics",
                "    {",
                f"        Tlow            {_format_number(float(temperature_ranges[0]))};",
                f"        Thigh           {_format_number(float(temperature_ranges[2]))};",
                f"        Tcommon         {_format_number(float(temperature_ranges[1]))};",
                f"        highCpCoeffs    {_format_coeff_list(high_coeffs)};",
                f"        lowCpCoeffs     {_format_coeff_list(low_coeffs)};",
                "    }",
                "    transport",
                "    {",
                f"        As              {_format_number(a_s)};",
                f"        Ts              {_format_number(ts)};",
                "    }",
                "}",
                "",
            ]
        )
    return lines


def _third_body_coeff_lines(
    *,
    gas: Any,
    efficiencies: dict[str, float],
    default_efficiency: float,
    indent: str,
) -> list[str]:
    lines = [
        f"{indent}defaultEfficiency {_format_number(default_efficiency)};",
        f"{indent}coeffs",
        f"{indent}{len(gas.species_names)}",
        f"{indent}(",
    ]
    for species_name in gas.species_names:
        efficiency = float(efficiencies.get(species_name, default_efficiency))
        lines.append(f"{indent}({_format_word(species_name)} {_format_number(efficiency)})")
    lines.extend(
        [
            f"{indent})",
            f"{indent};",
        ]
    )
    return lines


def _reaction_entry_lines(gas: Any, reaction_index: int, reaction: Any) -> list[str]:
    input_data = dict(reaction.input_data or {})
    rate_type = str(reaction.reaction_type)
    base_type = "reversible" if bool(reaction.reversible) else "irreversible"
    lines = [
        f"    reaction-{reaction_index:05d}",
        "    {",
    ]

    if rate_type == "Arrhenius":
        reaction_type = f"{base_type}ArrheniusReaction"
        lines.extend(
            [
                f"        type            {reaction_type};",
                f'        reaction        "{_reaction_equation(reaction)}";',
                *_arrhenius_block_lines(
                    rate_constant=dict(input_data.get("rate-constant", {}) or {})
                ),
            ]
        )
    elif rate_type == "three-body-Arrhenius":
        reaction_type = f"{base_type}thirdBodyArrheniusReaction"
        third_body = getattr(reaction, "third_body", None)
        lines.extend(
            [
                f"        type            {reaction_type};",
                f'        reaction        "{_reaction_equation(reaction)}";',
                *_arrhenius_block_lines(
                    rate_constant=dict(input_data.get("rate-constant", {}) or {})
                ),
                *_third_body_coeff_lines(
                    gas=gas,
                    efficiencies=dict(input_data.get("efficiencies", {}) or {}),
                    default_efficiency=float(getattr(third_body, "default_efficiency", 1.0)),
                    indent="        ",
                ),
            ]
        )
    elif rate_type in {"falloff-Troe", "falloff-Lindemann"}:
        falloff_name = "Troe" if rate_type.endswith("Troe") else "Lindemann"
        reaction_type = f"{base_type}Arrhenius{falloff_name}FallOffReaction"
        third_body = getattr(reaction, "third_body", None)
        lines.extend(
            [
                f"        type            {reaction_type};",
                f'        reaction        "{_reaction_equation(reaction)}";',
                "        k0",
                "        {",
                *_arrhenius_block_lines(
                    rate_constant=dict(input_data.get("low-P-rate-constant", {}) or {}),
                    indent="            ",
                ),
                "        }",
                "        kInf",
                "        {",
                *_arrhenius_block_lines(
                    rate_constant=dict(input_data.get("high-P-rate-constant", {}) or {}),
                    indent="            ",
                ),
                "        }",
                "        F",
                "        {",
            ]
        )
        if falloff_name == "Troe":
            troe = dict(input_data.get("Troe", {}) or {})
            if troe:
                lines.extend(
                    [
                        f"            alpha           {_format_number(float(troe.get('A', 0.0)))};",
                        f"            Tsss            {_format_number(float(troe.get('T3', 0.0)))};",
                        f"            Ts              {_format_number(float(troe.get('T1', 0.0)))};",
                    ]
                )
                if "T2" in troe:
                    lines.append(
                        f"            Tss             {_format_number(float(troe.get('T2', 0.0)))};"
                    )
        lines.extend(
            [
                "        }",
                "        thirdBodyEfficiencies",
                "        {",
                *_third_body_coeff_lines(
                    gas=gas,
                    efficiencies=dict(input_data.get("efficiencies", {}) or {}),
                    default_efficiency=float(getattr(third_body, "default_efficiency", 1.0)),
                    indent="            ",
                ),
                "        }",
            ]
        )
    else:
        raise ValueError(
            f"Unsupported OpenFOAM chemistry package reaction type '{rate_type}' "
            f"for reaction '{reaction.equation}'"
        )

    lines.extend(
        [
            "    }",
            "",
        ]
    )
    return lines


def _reactions_lines(gas: Any) -> list[str]:
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      reactions;",
        "}",
        "",
        "elements",
        "(",
    ]
    for element_name in getattr(gas, "element_names", []):
        lines.append(_format_word(str(element_name)))
    lines.extend(
        [
            ");",
            "",
            "",
            "species",
            "(",
        ]
    )
    for species_name in gas.species_names:
        lines.append(f"    {_format_word(species_name)}")
    lines.extend(
        [
            ");",
            "",
            "reactions",
            "{",
        ]
    )
    for reaction_index in range(gas.n_reactions):
        lines.extend(_reaction_entry_lines(gas, reaction_index, gas.reaction(reaction_index)))
    lines.append("}")
    return lines


def _transport_properties_lines(gas: Any) -> list[str]:
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      transportProperties;",
        "}",
        "",
        "species",
        "{",
    ]
    for species_name in gas.species_names:
        lines.extend(
            [
                f"    {_format_word(species_name)}",
                "    {",
                f"        As          {_format_number(DEFAULT_TRANSPORT_AS)};",
                f"        Ts          {_format_number(DEFAULT_TRANSPORT_TS)};",
                "    }",
            ]
        )
    lines.extend(
        [
            "}",
            "",
        ]
    )
    return lines


def _resolve_openfoam_package_config(config: dict[str, Any]) -> dict[str, Any]:
    def _from_adapter(adapter: dict[str, Any]) -> dict[str, Any]:
        package_cfg = dict(adapter.get("openfoam_chemistry_package", {}) or {})
        if package_cfg:
            return package_cfg
        return {}

    if "regimes" in config:
        regimes = dict(config.get("regimes", {}) or {})
        for regime_name in ("spray", "reacting_flow", "chemistry"):
            regime_cfg = dict(regimes.get(regime_name, {}) or {})
            solver_cfg = dict(regime_cfg.get("case_spec", {}).get("solver_config", {}) or {})
            adapter_cfg = dict(solver_cfg.get("simulation_adapter", {}) or {})
            package_cfg = _from_adapter(adapter_cfg)
            if package_cfg:
                return package_cfg
        raise ValueError(
            "Config does not define simulation_adapter.openfoam_chemistry_package in a "
            "spray, reacting_flow, or chemistry regime"
        )

    solver_cfg = dict(config.get("case_spec", {}).get("solver_config", {}) or {})
    adapter_cfg = dict(solver_cfg.get("simulation_adapter", {}) or {})
    package_cfg = _from_adapter(adapter_cfg)
    if package_cfg:
        return package_cfg
    raise ValueError(
        "Config does not define case_spec.solver_config.simulation_adapter."
        "openfoam_chemistry_package"
    )


def build_openfoam_chemistry_package(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Build the tracked OpenFOAM chemistry package referenced by a config JSON."""
    config = _load_config(config_path)
    package_cfg = _resolve_openfoam_package_config(config)
    return build_openfoam_chemistry_package_from_spec(
        package_cfg,
        output_dir=output_dir,
        refresh=refresh,
    )


def build_openfoam_chemistry_package_from_spec(
    package_cfg: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Build a tracked OpenFOAM chemistry package from a package spec."""
    mechanism_file = Path(str(package_cfg.get("mechanism_file", "")).strip())
    thermo_file = Path(str(package_cfg.get("thermo_file", "")).strip())
    transport_file_raw = str(package_cfg.get("transport_file", "")).strip()
    transport_file = Path(transport_file_raw) if transport_file_raw else None
    if not mechanism_file:
        raise ValueError("OpenFOAM chemistry package requires mechanism_file")
    if not thermo_file:
        raise ValueError("OpenFOAM chemistry package requires thermo_file")
    if not mechanism_file.exists():
        raise FileNotFoundError(f"Mechanism file not found: {mechanism_file}")
    if not thermo_file.exists():
        raise FileNotFoundError(f"Thermo file not found: {thermo_file}")
    if transport_file is not None and not transport_file.exists():
        raise FileNotFoundError(f"Transport file not found: {transport_file}")

    package_dir = Path(
        output_dir
        or str(package_cfg.get("output_dir", "")).strip()
        or Path("mechanisms") / "openfoam" / f"v{OPENFOAM_VERSION_DEFAULT}" / "chem323_reduced"
    )
    package_dir.mkdir(parents=True, exist_ok=True)

    package_id = str(package_cfg.get("package_id", "chem323_reduced_v2512")).strip()
    openfoam_version = (
        str(package_cfg.get("openfoam_version", OPENFOAM_VERSION_DEFAULT)).strip()
        or OPENFOAM_VERSION_DEFAULT
    )
    fuel_species = str(package_cfg.get("fuel_species", "IC8H18")).strip() or "IC8H18"
    generated_yaml_path = Path(
        str(package_cfg.get("generated_yaml_path", "")).strip()
        or Path("outputs") / "validation_runtime" / "mechanisms" / f"{mechanism_file.stem}.yaml"
    )

    reactions_path = package_dir / "reactions"
    thermo_path = package_dir / "thermo.compressibleGas"
    transport_path = package_dir / "transportProperties"
    manifest_path = package_dir / "package_manifest.json"

    source_files = [mechanism_file, thermo_file]
    if transport_file is not None:
        source_files.append(transport_file)
    output_files = [reactions_path, thermo_path, transport_path, manifest_path]
    if (
        not refresh
        and all(path.exists() for path in output_files)
        and all(
            min(path.stat().st_mtime for path in output_files) >= source_path.stat().st_mtime
            for source_path in source_files
        )
    ):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    yaml_path = convert_chemkin_to_yaml(
        input_file=mechanism_file,
        thermo_file=thermo_file,
        transport_file=transport_file,
        output_file=generated_yaml_path,
        phase_name=str(package_cfg.get("phase_name", "gas")),
        permissive=bool(package_cfg.get("permissive", True)),
        quiet=True,
        sanitizer_profile=str(
            package_cfg.get("sanitizer_profile", "llnl_detailed_gasoline_surrogate")
        ),
    )

    ct = _load_cantera()
    gas = ct.Solution(str(yaml_path))

    reactions_path.write_text("\n".join(_reactions_lines(gas)) + "\n", encoding="utf-8")
    thermo_path.write_text("\n".join(_thermo_lines(gas)) + "\n", encoding="utf-8")
    transport_path.write_text(
        "\n".join(_transport_properties_lines(gas)) + "\n",
        encoding="utf-8",
    )

    source_hashes = {str(path): _sha256_file(path) for path in source_files}
    generated_hashes = {
        str(reactions_path): _sha256_file(reactions_path),
        str(thermo_path): _sha256_file(thermo_path),
        str(transport_path): _sha256_file(transport_path),
        str(yaml_path): _sha256_file(Path(yaml_path)),
    }
    package_hash = hashlib.sha256(
        "".join(
            generated_hashes[str(path)] for path in (reactions_path, thermo_path, transport_path)
        ).encode("utf-8")
    ).hexdigest()

    manifest = {
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "package_id": package_id,
        "openfoam_version": openfoam_version,
        "fuel_species": fuel_species,
        "source_raw_files": [str(path) for path in source_files],
        "sanitizer_profile": str(
            package_cfg.get("sanitizer_profile", "llnl_detailed_gasoline_surrogate")
        ),
        "generated_yaml_path": str(yaml_path),
        "source_file_hashes": source_hashes,
        "generated_file_hashes": generated_hashes,
        "species_count": int(gas.n_species),
        "reaction_count": int(gas.n_reactions),
        "package_hash": package_hash,
        "files": {
            "reactions": str(reactions_path),
            "thermo.compressibleGas": str(thermo_path),
            "transportProperties": str(transport_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
