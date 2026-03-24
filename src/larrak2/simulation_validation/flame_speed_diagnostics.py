"""Standalone diagnostics for detailed-mechanism laminar flame-speed runs."""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .cantera_mechanisms import convert_chemkin_to_yaml


@dataclass
class FlameDiagnosticCase:
    """One Cantera flame-speed diagnostic attempt."""

    case_id: str
    mode: str
    transport_model: str | None
    timeout_s: int
    width_m: float = 0.02
    grid_points: int = 6
    loglevel: int = 0
    auto: bool = False
    refine_ratio: float = 20.0
    refine_slope: float = 0.5
    refine_curve: float = 0.5
    refine_prune: float = 0.1
    staged_energy: list[bool] = field(default_factory=list)
    refine_grid: bool = True
    max_grid_points: int = 0
    description: str = ""


@dataclass
class FlameDiagnosticResult:
    """Outcome of a single flame-speed diagnostic attempt."""

    case_id: str
    mode: str
    transport_model: str | None
    timeout_s: int
    success: bool
    timed_out: bool
    load_time_s: float = 0.0
    solve_time_s: float = 0.0
    total_time_s: float = 0.0
    n_species: int = 0
    n_reactions: int = 0
    n_grid_points: int = 0
    flame_speed_m_s: float | None = None
    max_temperature_K: float | None = None
    warnings: list[str] = field(default_factory=list)
    error_type: str = ""
    error_message: str = ""
    traceback_text: str = ""
    stage_timings_s: list[float] = field(default_factory=list)


def default_case_set(case_set: str = "quick") -> list[FlameDiagnosticCase]:
    """Return predefined diagnostic cases."""
    normalized = case_set.strip().lower()
    if normalized == "transport":
        return [
            FlameDiagnosticCase(
                case_id="load_mixture_averaged",
                mode="load_only",
                transport_model="mixture-averaged",
                timeout_s=900,
                description="Load the detailed LLNL mechanism with mixture-averaged transport.",
            )
        ]
    if normalized == "quick":
        return [
            FlameDiagnosticCase(
                case_id="load_transport_none",
                mode="load_only",
                transport_model=None,
                timeout_s=900,
                description="Load the detailed LLNL mechanism without transport to isolate parse cost.",
            ),
            FlameDiagnosticCase(
                case_id="free_flame_staged_mixture_averaged",
                mode="free_flame",
                transport_model="mixture-averaged",
                timeout_s=900,
                width_m=0.015,
                grid_points=6,
                auto=False,
                refine_ratio=20.0,
                refine_slope=0.5,
                refine_curve=0.5,
                refine_prune=0.1,
                staged_energy=[False, True],
                description=(
                    "Try a staged adiabatic FreeFlame solve with a coarse grid and "
                    "mixture-averaged transport."
                ),
            ),
        ]
    if normalized == "benchmark":
        return [
            FlameDiagnosticCase(
                case_id="load_transport_none",
                mode="load_only",
                transport_model=None,
                timeout_s=300,
                description="Load without transport for a reduced-mechanism benchmark path.",
            ),
            FlameDiagnosticCase(
                case_id="free_flame_benchmark_mixture_averaged",
                mode="free_flame",
                transport_model="mixture-averaged",
                timeout_s=300,
                width_m=0.01,
                grid_points=5,
                auto=False,
                refine_ratio=50.0,
                refine_slope=0.9,
                refine_curve=0.9,
                refine_prune=0.6,
                staged_energy=[False, True],
                refine_grid=False,
                description=(
                    "Aggressive reduced-mechanism benchmark using a fixed coarse grid "
                    "without adaptive refinement."
                ),
            ),
        ]
    if normalized == "matrix":
        return [
            FlameDiagnosticCase(
                case_id="load_transport_none",
                mode="load_only",
                transport_model=None,
                timeout_s=900,
                description="Load without transport.",
            ),
            FlameDiagnosticCase(
                case_id="load_mixture_averaged",
                mode="load_only",
                transport_model="mixture-averaged",
                timeout_s=900,
                description="Load with mixture-averaged transport.",
            ),
            FlameDiagnosticCase(
                case_id="free_flame_manual_mixture_averaged",
                mode="free_flame",
                transport_model="mixture-averaged",
                timeout_s=900,
                width_m=0.02,
                grid_points=6,
                auto=False,
                refine_ratio=20.0,
                refine_slope=0.5,
                refine_curve=0.5,
                refine_prune=0.1,
                staged_energy=[],
                description="Single-stage manual solve with mixture-averaged transport.",
            ),
            FlameDiagnosticCase(
                case_id="free_flame_staged_mixture_averaged",
                mode="free_flame",
                transport_model="mixture-averaged",
                timeout_s=900,
                width_m=0.015,
                grid_points=6,
                auto=False,
                refine_ratio=20.0,
                refine_slope=0.5,
                refine_curve=0.5,
                refine_prune=0.1,
                staged_energy=[False, True],
                description="Staged energy solve with mixture-averaged transport.",
            ),
            FlameDiagnosticCase(
                case_id="free_flame_staged_unity_lewis",
                mode="free_flame",
                transport_model="unity-Lewis-number",
                timeout_s=900,
                width_m=0.015,
                grid_points=6,
                auto=False,
                refine_ratio=20.0,
                refine_slope=0.5,
                refine_curve=0.5,
                refine_prune=0.1,
                staged_energy=[False, True],
                description="Staged energy solve with unity-Lewis-number transport.",
            ),
        ]
    raise ValueError(f"Unknown flame diagnostic case set '{case_set}'")


def classify_diagnostic_results(
    results: list[FlameDiagnosticResult],
    *,
    tractable_limit_s: float = 600.0,
) -> tuple[str, str]:
    """Classify tractability based on diagnostic outcomes."""
    attempted_flame_cases = [result for result in results if result.mode == "free_flame"]
    successful_flame_cases = [
        result
        for result in results
        if result.mode == "free_flame" and result.success and result.flame_speed_m_s is not None
    ]
    if successful_flame_cases:
        fastest = min(successful_flame_cases, key=lambda item: item.total_time_s)
        if fastest.total_time_s <= tractable_limit_s:
            return (
                "tractable",
                "A detailed-mechanism flame-speed solve completed within the tractable threshold.",
            )
        return (
            "marginal",
            "A detailed-mechanism flame-speed solve completed, but runtime is too high for "
            "routine validation use.",
        )

    if not attempted_flame_cases:
        successful_load_cases = [
            result for result in results if result.mode == "load_only" and result.success
        ]
        if successful_load_cases:
            return (
                "load_only_success",
                "Only load-only diagnostics were run; the detailed mechanism loads successfully.",
            )
        return (
            "load_failure",
            "Load-only diagnostics were run, but the detailed mechanism did not complete them.",
        )

    successful_load_cases = [
        result for result in results if result.mode == "load_only" and result.success
    ]
    transport_data_failures = [
        result
        for result in attempted_flame_cases
        if "Missing gas-phase transport data" in result.error_message
    ]
    if transport_data_failures and successful_load_cases:
        return (
            "transport_data_missing",
            "The mechanism loads successfully, but the configured flame-speed transport "
            "model is unavailable for at least one species.",
        )
    if successful_load_cases:
        return (
            "reduced_mechanism_recommended",
            "The mechanism loads successfully, but no flame-speed solve completed "
            "within the configured time limits.",
        )
    return (
        "load_failure",
        "The detailed mechanism did not complete even the load-only diagnostic case.",
    )


def _prepare_mechanism(
    *,
    mechanism_file: str,
    mechanism_format: str,
    thermo_file: str,
    transport_file: str,
    generated_yaml_path: str,
    sanitizer_profile: str,
) -> str:
    mechanism_path = Path(mechanism_file)
    normalized_format = mechanism_format.strip().lower()
    if mechanism_path.suffix.lower() in {".yaml", ".yml"} and normalized_format != "chemkin":
        return str(mechanism_path)
    if normalized_format not in {"", "chemkin"} and mechanism_path.suffix.lower() not in {
        ".inp",
        ".txt",
    }:
        return str(mechanism_path)
    if not thermo_file:
        raise ValueError("CHEMKIN flame diagnostics require thermo_file")
    output_path = Path(generated_yaml_path)
    convert_chemkin_to_yaml(
        input_file=mechanism_path,
        thermo_file=Path(thermo_file),
        transport_file=Path(transport_file) if transport_file else None,
        output_file=output_path,
        permissive=True,
        quiet=True,
        sanitizer_profile=sanitizer_profile,
    )
    return str(output_path)


def _worker_run_case(
    queue: mp.Queue,
    payload: dict[str, Any],
) -> None:
    """Run one diagnostic case in a child process."""
    started = time.perf_counter()
    result = FlameDiagnosticResult(
        case_id=str(payload["case_id"]),
        mode=str(payload["mode"]),
        transport_model=payload.get("transport_model"),
        timeout_s=int(payload["timeout_s"]),
        success=False,
        timed_out=False,
    )
    try:
        ct = importlib.import_module("cantera")
        warnings.simplefilter("always")
        with warnings.catch_warnings(record=True) as captured:
            load_started = time.perf_counter()
            if payload.get("transport_model") is None:
                gas = ct.Solution(str(payload["cantera_mechanism_file"]), transport_model=None)
            else:
                gas = ct.Solution(
                    str(payload["cantera_mechanism_file"]),
                    transport_model=str(payload["transport_model"]),
                )
            result.load_time_s = time.perf_counter() - load_started
            result.n_species = int(gas.n_species)
            result.n_reactions = int(gas.n_reactions)

            if payload["mode"] == "load_only":
                result.success = True
            else:
                gas.TP = float(payload["temperature_K"]), float(payload["pressure_bar"]) * 1.0e5
                gas.set_equivalence_ratio(
                    float(payload["equivalence_ratio"]),
                    str(payload["fuel"]),
                    dict(payload["oxidizer"]),
                )
                width_m = float(payload["width_m"])
                grid_points = int(payload["grid_points"])
                grid = [i * width_m / max(grid_points - 1, 1) for i in range(grid_points)]
                flame = ct.FreeFlame(gas, grid=grid)
                if payload.get("transport_model") is not None:
                    flame.transport_model = str(payload["transport_model"])
                max_grid_points = int(payload.get("max_grid_points", 0) or 0)
                if max_grid_points > 0:
                    flame.set_max_grid_points(flame.flame, max_grid_points)
                flame.set_refine_criteria(
                    ratio=float(payload["refine_ratio"]),
                    slope=float(payload["refine_slope"]),
                    curve=float(payload["refine_curve"]),
                    prune=float(payload["refine_prune"]),
                )

                solve_started = time.perf_counter()
                stage_timings: list[float] = []
                staged_energy = list(payload.get("staged_energy", []))
                if staged_energy:
                    for energy_enabled in staged_energy:
                        stage_started = time.perf_counter()
                        flame.energy_enabled = bool(energy_enabled)
                        flame.solve(
                            loglevel=int(payload["loglevel"]),
                            refine_grid=bool(payload.get("refine_grid", True)),
                            auto=bool(payload["auto"]),
                        )
                        stage_timings.append(time.perf_counter() - stage_started)
                else:
                    flame.solve(
                        loglevel=int(payload["loglevel"]),
                        refine_grid=bool(payload.get("refine_grid", True)),
                        auto=bool(payload["auto"]),
                    )
                result.solve_time_s = time.perf_counter() - solve_started
                result.stage_timings_s = stage_timings
                result.success = True
                result.flame_speed_m_s = float(flame.velocity[0])
                result.n_grid_points = int(len(flame.grid))
                result.max_temperature_K = float(max(flame.T))

            result.total_time_s = time.perf_counter() - started
            result.warnings = [str(item.message) for item in captured]
    except Exception as exc:
        result.total_time_s = time.perf_counter() - started
        result.error_type = type(exc).__name__
        result.error_message = str(exc)
        result.traceback_text = traceback.format_exc()

    queue.put(asdict(result))


def run_case_with_timeout(
    case: FlameDiagnosticCase,
    *,
    cantera_mechanism_file: str,
    temperature_K: float,
    pressure_bar: float,
    equivalence_ratio: float,
    fuel: str,
    oxidizer: dict[str, float],
) -> FlameDiagnosticResult:
    """Execute one diagnostic case in a separate process with a hard timeout."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    payload = {
        **asdict(case),
        "cantera_mechanism_file": cantera_mechanism_file,
        "temperature_K": temperature_K,
        "pressure_bar": pressure_bar,
        "equivalence_ratio": equivalence_ratio,
        "fuel": fuel,
        "oxidizer": oxidizer,
    }
    proc = ctx.Process(target=_worker_run_case, args=(queue, payload))
    proc.start()
    proc.join(case.timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return FlameDiagnosticResult(
            case_id=case.case_id,
            mode=case.mode,
            transport_model=case.transport_model,
            timeout_s=case.timeout_s,
            success=False,
            timed_out=True,
            total_time_s=float(case.timeout_s),
            error_type="TimeoutExpired",
            error_message=f"Case exceeded {case.timeout_s}s timeout",
        )

    if not queue.empty():
        return FlameDiagnosticResult(**queue.get())

    return FlameDiagnosticResult(
        case_id=case.case_id,
        mode=case.mode,
        transport_model=case.transport_model,
        timeout_s=case.timeout_s,
        success=False,
        timed_out=False,
        error_type="MissingResult",
        error_message="Worker exited without returning a result payload",
    )


def write_diagnostic_artifacts(
    *,
    outdir: Path,
    metadata: dict[str, Any],
    results: list[FlameDiagnosticResult],
    diagnosis_classification: str,
    diagnosis_summary: str,
) -> tuple[Path, Path]:
    """Persist JSON and Markdown summaries."""
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "metadata": metadata,
        "diagnosis_classification": diagnosis_classification,
        "diagnosis_summary": diagnosis_summary,
        "results": [asdict(result) for result in results],
    }
    json_path = outdir / "llnl_flame_speed_diagnostic.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LLNL Flame-Speed Diagnostic",
        "",
        f"**Classification:** {diagnosis_classification}",
        f"**Summary:** {diagnosis_summary}",
        "",
        "| Case | Mode | Success | Timeout | Load (s) | Solve (s) | Total (s) | Flame Speed (m/s) |",
        "|------|------|---------|---------|----------|-----------|-----------|-------------------|",
    ]
    for result in results:
        flame_speed = "" if result.flame_speed_m_s is None else f"{result.flame_speed_m_s:.6g}"
        lines.append(
            f"| {result.case_id} | {result.mode} | {result.success} | {result.timed_out} | "
            f"{result.load_time_s:.3f} | {result.solve_time_s:.3f} | {result.total_time_s:.3f} | "
            f"{flame_speed} |"
        )
        if result.error_message:
            lines.append("")
            lines.append(f"- `{result.case_id}` error: {result.error_type}: {result.error_message}")
        if result.warnings:
            for warning in result.warnings[:3]:
                lines.append(f"- `{result.case_id}` warning: {warning}")
    md_path = outdir / "llnl_flame_speed_diagnostic.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def run_flame_speed_diagnostics(
    *,
    mechanism_file: str,
    mechanism_format: str = "",
    thermo_file: str = "",
    transport_file: str = "",
    generated_yaml_path: str = "mechanisms/iso_octane/llnl_2022.yaml",
    sanitizer_profile: str = "",
    case_set: str = "quick",
    temperature_K: float = 353.0,
    pressure_bar: float = 3.33,
    equivalence_ratio: float = 1.0,
    fuel: str = "IC8H18",
    oxidizer: dict[str, float] | None = None,
    outdir: str | Path = "outputs/diagnostics/llnl_flame_speed",
) -> dict[str, Any]:
    """Run a preset flame-speed diagnostic sweep and persist results."""
    oxidizer_map = dict(oxidizer or {"O2": 0.2033, "N2": 0.7859})
    cantera_mechanism_file = _prepare_mechanism(
        mechanism_file=mechanism_file,
        mechanism_format=mechanism_format,
        thermo_file=thermo_file,
        transport_file=transport_file,
        generated_yaml_path=generated_yaml_path,
        sanitizer_profile=sanitizer_profile,
    )

    results: list[FlameDiagnosticResult] = []
    for case in default_case_set(case_set):
        results.append(
            run_case_with_timeout(
                case,
                cantera_mechanism_file=cantera_mechanism_file,
                temperature_K=temperature_K,
                pressure_bar=pressure_bar,
                equivalence_ratio=equivalence_ratio,
                fuel=fuel,
                oxidizer=oxidizer_map,
            )
        )

    classification, summary = classify_diagnostic_results(results)
    metadata = {
        "mechanism_file": mechanism_file,
        "cantera_mechanism_file": cantera_mechanism_file,
        "mechanism_format": mechanism_format,
        "thermo_file": thermo_file,
        "transport_file": transport_file,
        "generated_yaml_path": generated_yaml_path,
        "sanitizer_profile": sanitizer_profile,
        "case_set": case_set,
        "temperature_K": temperature_K,
        "pressure_bar": pressure_bar,
        "equivalence_ratio": equivalence_ratio,
        "fuel": fuel,
        "oxidizer": oxidizer_map,
        "python_executable": os.sys.executable,
    }
    json_path, md_path = write_diagnostic_artifacts(
        outdir=Path(outdir),
        metadata=metadata,
        results=results,
        diagnosis_classification=classification,
        diagnosis_summary=summary,
    )
    return {
        "diagnosis_classification": classification,
        "diagnosis_summary": summary,
        "results": [asdict(result) for result in results],
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "metadata": metadata,
    }
