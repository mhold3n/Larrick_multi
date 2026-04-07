"""Run standalone diagnostics for the LLNL detailed flame-speed leg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from larrak_simulation.simulation_validation.flame_speed_diagnostics import (
    run_flame_speed_diagnostics,
)


def _parse_oxidizer(raw: str) -> dict[str, float]:
    pairs = [item.strip() for item in raw.split(",") if item.strip()]
    oxidizer: dict[str, float] = {}
    for pair in pairs:
        name, value = pair.split("=", 1)
        oxidizer[name.strip()] = float(value.strip())
    return oxidizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose LLNL flame-speed tractability")
    parser.add_argument(
        "--mechanism-file",
        default="mechanisms/iso_octane/llnl_2022.yaml",
        help="YAML mechanism path or CHEMKIN input file",
    )
    parser.add_argument(
        "--mechanism-format",
        default="",
        help="Set to 'chemkin' when using a CHEMKIN input file",
    )
    parser.add_argument(
        "--thermo-file",
        default="",
        help="Thermo file for CHEMKIN conversion",
    )
    parser.add_argument(
        "--transport-file",
        default="",
        help="Transport file for CHEMKIN conversion",
    )
    parser.add_argument(
        "--generated-yaml-path",
        default="mechanisms/iso_octane/llnl_2022.yaml",
        help="Target YAML path when converting CHEMKIN input",
    )
    parser.add_argument(
        "--sanitizer-profile",
        default="",
        help="Optional known sanitizer profile for malformed CHEMKIN bundles",
    )
    parser.add_argument(
        "--case-set",
        default="quick",
        choices=["transport", "quick", "benchmark", "matrix"],
        help="Preset diagnostic case set",
    )
    parser.add_argument("--temperature-k", type=float, default=353.0)
    parser.add_argument("--pressure-bar", type=float, default=3.33)
    parser.add_argument("--equivalence-ratio", type=float, default=1.0)
    parser.add_argument("--fuel", default="IC8H18")
    parser.add_argument("--oxidizer", default="O2=0.2033,N2=0.7859")
    parser.add_argument(
        "--outdir",
        default="outputs/diagnostics/llnl_flame_speed",
        help="Directory for JSON and Markdown summaries",
    )
    args = parser.parse_args()

    summary = run_flame_speed_diagnostics(
        mechanism_file=str(args.mechanism_file),
        mechanism_format=str(args.mechanism_format),
        thermo_file=str(args.thermo_file),
        transport_file=str(args.transport_file),
        generated_yaml_path=str(args.generated_yaml_path),
        sanitizer_profile=str(args.sanitizer_profile),
        case_set=str(args.case_set),
        temperature_K=float(args.temperature_k),
        pressure_bar=float(args.pressure_bar),
        equivalence_ratio=float(args.equivalence_ratio),
        fuel=str(args.fuel),
        oxidizer=_parse_oxidizer(str(args.oxidizer)),
        outdir=Path(args.outdir),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
