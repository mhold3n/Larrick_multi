"""CHEMKIN sanitization and Cantera conversion helpers."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

LLNL_DETAILED_GASOLINE_SURROGATE = "llnl_detailed_gasoline_surrogate"

NUMERIC_TOKEN_RE = re.compile(r"(?<![Ee])(?P<coeff>(?:\d+\.\d*|\.\d+))(?P<exp>[+-]\d{2,3})(?!\d)")
EXCLUDED_SPECIES = {"C5H81OOH4-5O2", "C5H81OOH5-4O2"}


def _fix_numeric_tokens(text: str) -> str:
    return NUMERIC_TOKEN_RE.sub(r"\g<coeff>E\g<exp>", text)


def _sanitize_llnl_mechanism_lines(lines: list[str]) -> list[str]:
    sanitized: list[str] = []
    in_species = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        upper = stripped.upper()
        if upper == "SPECIES":
            in_species = True
            sanitized.append("SPECIES\n")
            continue
        if in_species and upper == "END":
            in_species = False
            sanitized.append("END\n")
            continue

        if in_species:
            tokens = [tok for tok in stripped.split() if tok not in EXCLUDED_SPECIES]
            if tokens:
                sanitized.append(" ".join(tokens) + "\n")
            continue

        if any(species in line for species in EXCLUDED_SPECIES):
            continue

        sanitized.append(_fix_numeric_tokens(line) + "\n")

    return sanitized


def _sanitize_llnl_thermo_lines(lines: list[str]) -> list[str]:
    sanitized: list[str] = []
    in_block = False
    block_line = 0
    skip_block = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if not in_block:
            if stripped and not stripped.startswith("!") and stripped[-1:] == "1":
                in_block = True
                block_line = 1
                species_name = stripped.split()[0]
                skip_block = species_name in EXCLUDED_SPECIES
                if skip_block:
                    continue
            sanitized.append(line + "\n")
            continue

        block_line += 1
        if skip_block:
            if block_line >= 4:
                in_block = False
                block_line = 0
                skip_block = False
            continue
        if block_line == 4 and stripped and not stripped.startswith("!") and stripped[-1:] != "4":
            line = line.rstrip() + "                   4"
        sanitized.append(line + "\n")

        if block_line >= 4:
            in_block = False
            block_line = 0
            skip_block = False

    return sanitized


def sanitize_chemkin_file_text(
    *,
    source_file: Path,
    file_kind: str,
    profile: str = "",
) -> str:
    """Return sanitized CHEMKIN text for a single file.

    This is the text-level helper used by staged OpenFOAM inputs. It mirrors the
    bundle sanitizer so callers can sanitize one file at a time without first
    materializing the full CHEMKIN bundle in a scratch directory.
    """

    normalized_profile = profile.strip().lower()
    if not normalized_profile:
        return source_file.read_text(encoding="utf-8")
    if normalized_profile != LLNL_DETAILED_GASOLINE_SURROGATE:
        raise ValueError(f"Unsupported CHEMKIN sanitizer profile '{profile}'")

    normalized_kind = file_kind.strip().lower()
    lines = source_file.read_text(encoding="utf-8").splitlines(True)
    if normalized_kind == "input":
        return "".join(_sanitize_llnl_mechanism_lines(lines))
    if normalized_kind == "thermo":
        return "".join(_sanitize_llnl_thermo_lines(lines))
    if normalized_kind == "transport":
        return "".join(
            line for line in lines if not any(species in line for species in EXCLUDED_SPECIES)
        )
    raise ValueError(f"Unsupported CHEMKIN sanitizer file_kind '{file_kind}'")


def sanitize_chemkin_bundle(
    *,
    input_file: Path,
    thermo_file: Path,
    transport_file: Path | None = None,
    scratch_dir: Path,
    profile: str = "",
) -> tuple[Path, Path, Path | None]:
    """Write a sanitized CHEMKIN bundle for known malformed mechanism profiles."""
    normalized = profile.strip().lower()
    if not normalized:
        return input_file, thermo_file, transport_file
    if normalized != LLNL_DETAILED_GASOLINE_SURROGATE:
        raise ValueError(f"Unsupported CHEMKIN sanitizer profile '{profile}'")

    scratch_dir.mkdir(parents=True, exist_ok=True)

    sanitized_input = scratch_dir / input_file.name
    sanitized_thermo = scratch_dir / thermo_file.name
    sanitized_transport = scratch_dir / transport_file.name if transport_file else None

    sanitized_input.write_text(
        "".join(
            _sanitize_llnl_mechanism_lines(input_file.read_text(encoding="utf-8").splitlines(True))
        ),
        encoding="utf-8",
    )
    sanitized_thermo.write_text(
        "".join(
            _sanitize_llnl_thermo_lines(thermo_file.read_text(encoding="utf-8").splitlines(True))
        ),
        encoding="utf-8",
    )
    if sanitized_transport and transport_file:
        sanitized_transport.write_text(
            "".join(
                line
                for line in transport_file.read_text(encoding="utf-8").splitlines(True)
                if not any(species in line for species in EXCLUDED_SPECIES)
            ),
            encoding="utf-8",
        )

    return sanitized_input, sanitized_thermo, sanitized_transport


def _is_current(output_file: Path, inputs: list[Path]) -> bool:
    if not output_file.exists():
        return False
    output_mtime = output_file.stat().st_mtime
    return all(output_mtime >= path.stat().st_mtime for path in inputs)


def convert_chemkin_to_yaml(
    *,
    input_file: Path,
    thermo_file: Path,
    output_file: Path,
    transport_file: Path | None = None,
    phase_name: str = "gas",
    permissive: bool = False,
    quiet: bool = True,
    sanitizer_profile: str = "",
) -> Path:
    """Convert a CHEMKIN bundle to a Cantera YAML file, with optional sanitization."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    source_files = [input_file, thermo_file]
    if transport_file is not None:
        source_files.append(transport_file)
    if _is_current(output_file, source_files):
        return output_file

    prepared_input, prepared_thermo, prepared_transport = sanitize_chemkin_bundle(
        input_file=input_file,
        thermo_file=thermo_file,
        transport_file=transport_file,
        scratch_dir=output_file.parent / "_ck_sanitized",
        profile=sanitizer_profile,
    )

    ck2yaml = importlib.import_module("cantera.ck2yaml")
    ck2yaml.convert(
        input_file=str(prepared_input),
        thermo_file=str(prepared_thermo),
        transport_file=str(prepared_transport) if prepared_transport else None,
        out_name=str(output_file),
        phase_name=phase_name,
        permissive=permissive,
        quiet=quiet,
    )
    return output_file
