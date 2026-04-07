"""Sanitize LLNL CHEMKIN files and convert them to Cantera YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from larrak_simulation.simulation_validation.cantera_mechanisms import (
    LLNL_DETAILED_GASOLINE_SURROGATE,
    convert_chemkin_to_yaml,
)


def build_yaml(
    *,
    input_file: Path,
    thermo_file: Path,
    transport_file: Path,
    output_file: Path,
) -> None:
    convert_chemkin_to_yaml(
        input_file=input_file,
        thermo_file=thermo_file,
        transport_file=transport_file,
        output_file=output_file,
        permissive=True,
        quiet=False,
        sanitizer_profile=LLNL_DETAILED_GASOLINE_SURROGATE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LLNL gasoline Cantera YAML")
    parser.add_argument("--input", required=True)
    parser.add_argument("--thermo", required=True)
    parser.add_argument("--transport", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_yaml(
        input_file=Path(args.input),
        thermo_file=Path(args.thermo),
        transport_file=Path(args.transport),
        output_file=Path(args.output),
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
