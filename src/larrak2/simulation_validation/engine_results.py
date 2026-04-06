"""Reusable engine-run results extraction for cylinder thermo metrics."""

from __future__ import annotations

import json
import math
import re
from bisect import bisect_left
from pathlib import Path
from typing import Any

from ..adapters.openfoam import OpenFoamRunner

DEFAULT_APPARENT_HEAT_RELEASE_GAMMA = 1.32


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _numeric_time_dirs(run_dir: Path) -> list[tuple[float, Path]]:
    time_dirs: list[tuple[float, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            time_value = float(child.name)
        except ValueError:
            continue
        time_dirs.append((time_value, child))
    time_dirs.sort(key=lambda item: item[0])
    return time_dirs


def _load_logsummary_trace(engine_case_dir: str | Path) -> list[dict[str, float]]:
    root = Path(engine_case_dir)
    summaries = sorted(
        root.glob("logSummary.*.dat"),
        key=lambda path: float(path.name[len("logSummary.") : -len(".dat")]),
    )
    trace: list[dict[str, float]] = []
    for summary in summaries:
        rows = [
            line.strip()
            for line in summary.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not rows:
            continue
        crank, pressure, temperature, velocity = rows[-1].split()
        trace.append(
            {
                "time_s": float(summary.name[len("logSummary.") : -len(".dat")]),
                "crank_angle_deg": float(crank),
                "mean_pressure_Pa": float(pressure),
                "mean_temperature_K": float(temperature),
                "mean_velocity_magnitude_m_s": float(velocity),
            }
        )
    return trace


def _closest_time_dir(
    target_time_s: float,
    time_dirs: list[tuple[float, Path]],
) -> Path | None:
    if not time_dirs:
        return None
    times = [item[0] for item in time_dirs]
    idx = bisect_left(times, float(target_time_s))
    candidates: list[tuple[float, Path]] = []
    if idx < len(time_dirs):
        candidates.append(time_dirs[idx])
    if idx > 0:
        candidates.append(time_dirs[idx - 1])
    if not candidates:
        return None
    nearest_time, nearest_path = min(
        candidates, key=lambda item: abs(item[0] - float(target_time_s))
    )
    tolerance = max(1.0e-9, max(abs(float(target_time_s)), 1.0) * 1.0e-3)
    if abs(nearest_time - float(target_time_s)) > tolerance:
        return None
    return nearest_path


def _domain_volume_for_time_dir(time_dir: Path) -> float | None:
    field_path = OpenFoamRunner.cell_volume_field_path(time_dir)
    if field_path is None or not field_path.exists():
        return None
    values = OpenFoamRunner._read_scalar_field(field_path)
    if not values:
        return None
    return float(sum(values))


def _displacement_volume_m3(params: dict[str, Any]) -> float | None:
    if "displacement_volume_m3" in params:
        return float(params["displacement_volume_m3"])
    if "bore_mm" not in params or "stroke_mm" not in params:
        return None
    bore_m = float(params["bore_mm"]) / 1000.0
    stroke_m = float(params["stroke_mm"]) / 1000.0
    if bore_m <= 0.0 or stroke_m <= 0.0:
        return None
    return float(math.pi * 0.25 * bore_m * bore_m * stroke_m)


def _load_stage_boundaries(engine_case_dir: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(engine_case_dir) / "engine_stage_manifest.json"
    if not manifest_path.exists():
        return []
    payload = _load_json(manifest_path)
    stages = list(payload.get("stages", []) or [])
    boundaries: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        boundaries.append(
            {
                "name": str(stage.get("name", "")),
                "end_angle_deg": float(stage.get("end_angle_deg", 0.0)),
                "ok": bool(stage.get("ok", False)),
                "stage_result": str(stage.get("stage_result", "")),
            }
        )
    return boundaries


def _collect_floor_hit_counts(engine_case_dir: str | Path) -> dict[str, int]:
    root = Path(engine_case_dir)
    patterns = {
        "pressure_floor_hits": re.compile(r"Clipped pressure floor:\s+(\d+)\s+values"),
        "density_floor_hits": re.compile(r"Clipped density floor:\s+(\d+)\s+values"),
        "thermo_clip_hits": re.compile(
            r"Clipped thermo energy state before correction:\s+(\d+)\s+values"
        ),
        "thermo_window_hits": re.compile(
            r"Limited thermo correction window with maxThermoDeltaTK=[^\s]+\s+on\s+(\d+)\s+values"
        ),
    }
    counts = {key: 0 for key in patterns}
    for log_path in sorted(root.glob("larrakEngineFoam*.log")):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        for key, pattern in patterns.items():
            counts[key] += sum(int(match) for match in pattern.findall(text))
    return counts


def _find_fraction_angle(
    trace: list[dict[str, float]],
    *,
    fraction: float,
) -> float | None:
    if not trace:
        return None
    target = float(fraction)
    for previous, current in zip(trace, trace[1:]):
        prev_fraction = float(previous["cumulative_positive_heat_release_fraction"])
        current_fraction = float(current["cumulative_positive_heat_release_fraction"])
        if prev_fraction >= target:
            return float(previous["crank_angle_deg"])
        if current_fraction < target or current_fraction <= prev_fraction:
            continue
        span = current_fraction - prev_fraction
        if span <= 0.0:
            return float(current["crank_angle_deg"])
        alpha = (target - prev_fraction) / span
        return float(previous["crank_angle_deg"]) + alpha * (
            float(current["crank_angle_deg"]) - float(previous["crank_angle_deg"])
        )
    if float(trace[-1]["cumulative_positive_heat_release_fraction"]) >= target:
        return float(trace[-1]["crank_angle_deg"])
    return None


def _load_runtime_chemistry_summary(engine_case_dir: str | Path) -> dict[str, Any]:
    root = Path(engine_case_dir)
    summary_files = sorted(
        root.glob("runtimeChemistrySummary.*.dat"),
        key=lambda path: float(path.name[len("runtimeChemistrySummary.") : -len(".dat")]),
    )
    if not summary_files:
        return {
            "runtime_summary_count": 0.0,
            "table_query_cells": 0.0,
            "table_hit_cells": 0.0,
            "table_hit_fraction": None,
            "fallback_timesteps": 0.0,
            "fallback_hit_fraction": None,
            "interpolation_cache_hits": 0.0,
            "interpolation_cache_hit_fraction": None,
            "coverage_reject_cells": 0.0,
            "coverage_reject_cell_fraction": 0.0,
            "trust_region_reject_cells": 0.0,
            "runtime_coverage_corpus_path": "",
            "runtime_coverage_corpus_npz_path": "",
        }

    header: list[str] = []
    latest_row: list[float] | None = None
    for path in summary_files:
        rows = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        header_line = next((line for line in rows if line.startswith("#")), "")
        if header_line:
            header = header_line.lstrip("#").split()
        numeric_rows = [
            [float(token) for token in line.split()]
            for line in rows
            if not line.startswith("#")
        ]
        if numeric_rows:
            latest_row = numeric_rows[-1]
    if latest_row is None:
        return {
            "runtime_summary_count": float(len(summary_files)),
            "table_query_cells": 0.0,
            "table_hit_cells": 0.0,
            "table_hit_fraction": None,
            "fallback_timesteps": 0.0,
            "fallback_hit_fraction": None,
            "interpolation_cache_hits": 0.0,
            "interpolation_cache_hit_fraction": None,
            "coverage_reject_cells": 0.0,
            "coverage_reject_cell_fraction": 0.0,
            "trust_region_reject_cells": 0.0,
            "runtime_coverage_corpus_path": "",
            "runtime_coverage_corpus_npz_path": "",
        }

    values = dict(zip(header, latest_row, strict=False))
    table_query_cells = float(values.get("tableQueryCells", 0.0))
    table_hit_cells = float(values.get("tableHitCells", 0.0))
    fallback_timesteps = float(values.get("fallbackTimesteps", 0.0))
    interpolation_cache_hits = float(values.get("interpolationCacheHits", 0.0))
    coverage_reject_cells = float(values.get("coverageRejectCells", 0.0))
    trust_region_reject_cells = float(values.get("trustRegionRejectCells", 0.0))
    runtime_coverage_corpus_path = ""
    runtime_coverage_corpus_npz_path = ""
    json_candidate = root / "runtimeChemistryCoverageCorpus.json"
    npz_candidate = root / "runtimeChemistryCoverageCorpus.npz"
    if json_candidate.exists():
        runtime_coverage_corpus_path = str(json_candidate.resolve())
    if npz_candidate.exists():
        runtime_coverage_corpus_npz_path = str(npz_candidate.resolve())

    return {
        "runtime_summary_count": float(len(summary_files)),
        "table_query_cells": table_query_cells,
        "table_hit_cells": table_hit_cells,
        "table_hit_fraction": (
            None if table_query_cells <= 0.0 else float(table_hit_cells / table_query_cells)
        ),
        "fallback_timesteps": fallback_timesteps,
        "fallback_hit_fraction": (
            None if table_query_cells <= 0.0 else float(fallback_timesteps / table_query_cells)
        ),
        "interpolation_cache_hits": interpolation_cache_hits,
        "interpolation_cache_hit_fraction": (
            None
            if table_query_cells <= 0.0
            else float(interpolation_cache_hits / table_query_cells)
        ),
        "coverage_reject_cells": coverage_reject_cells,
        "coverage_reject_cell_fraction": (
            0.0 if table_query_cells <= 0.0 else float(coverage_reject_cells / table_query_cells)
        ),
        "trust_region_reject_cells": trust_region_reject_cells,
        "runtime_coverage_corpus_path": runtime_coverage_corpus_path,
        "runtime_coverage_corpus_npz_path": runtime_coverage_corpus_npz_path,
    }


def _current_engine_stage_name(engine_case_dir: str | Path) -> str:
    run_dir = Path(engine_case_dir)
    resume_summary_path = run_dir / "engine_stage_resume_summary.json"
    if resume_summary_path.exists():
        payload = _load_json(resume_summary_path)
        current_stage = str(payload.get("current_stage", "")).strip()
        if current_stage:
            return current_stage
        remaining = list(payload.get("remaining_stages", []) or [])
        if remaining:
            return str(remaining[0])
    for stage in _load_stage_boundaries(run_dir):
        if not bool(stage.get("ok", False)):
            return str(stage.get("name", ""))
    return ""


def build_engine_progress_summary(
    *,
    engine_case_dir: str | Path,
    params: dict[str, Any] | None = None,
    engine_metrics: dict[str, Any] | None = None,
    solver_name: str = "",
    handoff_bundle_id: str = "",
    mechanism_id: str = "",
    openfoam_chemistry_package_id: str = "",
    openfoam_chemistry_package_hash: str = "",
    custom_solver_source_hash: str = "",
    run_ok: bool | None = None,
    stage: str = "",
    gamma: float = DEFAULT_APPARENT_HEAT_RELEASE_GAMMA,
) -> dict[str, Any]:
    results = build_engine_results(
        engine_case_dir=engine_case_dir,
        params=params,
        engine_metrics=engine_metrics,
        solver_name=solver_name,
        handoff_bundle_id=handoff_bundle_id,
        mechanism_id=mechanism_id,
        openfoam_chemistry_package_id=openfoam_chemistry_package_id,
        openfoam_chemistry_package_hash=openfoam_chemistry_package_hash,
        custom_solver_source_hash=custom_solver_source_hash,
        run_ok=run_ok,
        stage=stage,
        gamma=gamma,
    )
    trace = list(results.get("trace", []) or [])
    stage_boundaries = list(results.get("stage_boundaries", []) or [])
    current_stage = _current_engine_stage_name(engine_case_dir)
    next_stage = next(
        (
            float(boundary["end_angle_deg"])
            for boundary in stage_boundaries
            if str(boundary.get("name", "")) == current_stage
        ),
        None,
    )
    progress = {
        "artifact_kind": "progress_summary",
        "engine_case_dir": str(Path(engine_case_dir).resolve()),
        "current_stage": current_stage,
        "next_stage_end_angle_deg": next_stage,
        "latest_checkpoint": {} if not trace else dict(trace[-1]),
        "metrics_so_far": dict(results.get("metrics", {}) or {}),
        "numeric_stability": dict(results.get("numeric_stability", {}) or {}),
        "runtime_chemistry": _load_runtime_chemistry_summary(engine_case_dir),
        "stage_boundaries": stage_boundaries,
        "trace_point_count": int(results.get("trace_point_count", 0) or 0),
        "trace_angle_span_deg": results.get("trace_angle_span_deg"),
        "sources": dict(results.get("sources", {}) or {}),
    }
    return progress


def emit_engine_progress_artifacts(
    *,
    engine_case_dir: str | Path,
    params: dict[str, Any] | None = None,
    engine_metrics: dict[str, Any] | None = None,
    solver_name: str = "",
    handoff_bundle_id: str = "",
    mechanism_id: str = "",
    openfoam_chemistry_package_id: str = "",
    openfoam_chemistry_package_hash: str = "",
    custom_solver_source_hash: str = "",
    run_ok: bool | None = None,
    stage: str = "",
    gamma: float = DEFAULT_APPARENT_HEAT_RELEASE_GAMMA,
) -> dict[str, Any]:
    partial_results = emit_engine_results_artifact(
        engine_case_dir=engine_case_dir,
        params=params,
        engine_metrics=engine_metrics,
        solver_name=solver_name,
        handoff_bundle_id=handoff_bundle_id,
        mechanism_id=mechanism_id,
        openfoam_chemistry_package_id=openfoam_chemistry_package_id,
        openfoam_chemistry_package_hash=openfoam_chemistry_package_hash,
        custom_solver_source_hash=custom_solver_source_hash,
        run_ok=run_ok,
        stage=stage,
        artifact_name="engine_results_partial.json",
        gamma=gamma,
    )
    progress = build_engine_progress_summary(
        engine_case_dir=engine_case_dir,
        params=params,
        engine_metrics=engine_metrics,
        solver_name=solver_name,
        handoff_bundle_id=handoff_bundle_id,
        mechanism_id=mechanism_id,
        openfoam_chemistry_package_id=openfoam_chemistry_package_id,
        openfoam_chemistry_package_hash=openfoam_chemistry_package_hash,
        custom_solver_source_hash=custom_solver_source_hash,
        run_ok=run_ok,
        stage=stage,
        gamma=gamma,
    )
    progress_path = Path(engine_case_dir) / "engine_progress_summary.json"
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "engine_results_partial_path": str((Path(engine_case_dir) / "engine_results_partial.json").resolve()),
        "engine_progress_summary_path": str(progress_path.resolve()),
        "partial_results": partial_results,
        "progress_summary": progress,
    }


def build_engine_results(
    *,
    engine_case_dir: str | Path,
    params: dict[str, Any] | None = None,
    engine_metrics: dict[str, Any] | None = None,
    solver_name: str = "",
    handoff_bundle_id: str = "",
    mechanism_id: str = "",
    openfoam_chemistry_package_id: str = "",
    openfoam_chemistry_package_hash: str = "",
    custom_solver_source_hash: str = "",
    run_ok: bool | None = None,
    stage: str = "",
    gamma: float = DEFAULT_APPARENT_HEAT_RELEASE_GAMMA,
) -> dict[str, Any]:
    run_dir = Path(engine_case_dir)
    resolved_params = dict(params or {})
    trace = _load_logsummary_trace(run_dir)
    time_dirs = _numeric_time_dirs(run_dir)

    enriched_trace: list[dict[str, Any]] = []
    for index, row in enumerate(trace):
        entry = dict(row)
        matched_dir = _closest_time_dir(float(row["time_s"]), time_dirs)
        volume = _domain_volume_for_time_dir(matched_dir) if matched_dir is not None else None
        entry["matched_time_dir"] = matched_dir.name if matched_dir is not None else None
        entry["domain_volume_m3"] = volume
        entry["delta_t_s"] = (
            None if index == 0 else float(row["time_s"]) - float(trace[index - 1]["time_s"])
        )
        enriched_trace.append(entry)

    resolved_metrics = dict(engine_metrics or {})
    if not resolved_metrics:
        metrics_path = run_dir / "openfoam_metrics.json"
        if metrics_path.exists():
            resolved_metrics = _load_json(metrics_path)

    peak_pressure = None
    peak_pressure_angle = None
    if enriched_trace:
        peak_entry = max(enriched_trace, key=lambda item: float(item["mean_pressure_Pa"]))
        peak_pressure = float(peak_entry["mean_pressure_Pa"])
        peak_pressure_angle = float(peak_entry["crank_angle_deg"])

    displacement_volume = _displacement_volume_m3(resolved_params)
    angle_span = None
    if len(enriched_trace) >= 2:
        angle_span = float(enriched_trace[-1]["crank_angle_deg"]) - float(
            enriched_trace[0]["crank_angle_deg"]
        )

    heat_release_trace: list[dict[str, float]] = []
    net_work = 0.0
    cumulative_positive_heat_release = 0.0
    total_positive_heat_release = 0.0
    heat_release_ready = all(entry.get("domain_volume_m3") is not None for entry in enriched_trace)
    if heat_release_ready and len(enriched_trace) >= 2 and float(gamma) > 1.0:
        for previous, current in zip(enriched_trace, enriched_trace[1:]):
            p0 = float(previous["mean_pressure_Pa"])
            p1 = float(current["mean_pressure_Pa"])
            v0 = float(previous["domain_volume_m3"])
            v1 = float(current["domain_volume_m3"])
            dV = v1 - v0
            dp = p1 - p0
            p_mid = 0.5 * (p0 + p1)
            v_mid = 0.5 * (v0 + v1)
            dQ = (
                float(gamma) / (float(gamma) - 1.0) * p_mid * dV
                + 1.0 / (float(gamma) - 1.0) * v_mid * dp
            )
            total_positive_heat_release += max(dQ, 0.0)
            net_work += p_mid * dV
            heat_release_trace.append(
                {
                    "time_s": float(current["time_s"]),
                    "crank_angle_deg": float(current["crank_angle_deg"]),
                    "apparent_heat_release_step_J": float(dQ),
                }
            )

        for point in heat_release_trace:
            cumulative_positive_heat_release += max(
                float(point["apparent_heat_release_step_J"]),
                0.0,
            )
            point["cumulative_positive_heat_release_J"] = float(cumulative_positive_heat_release)
            point["cumulative_positive_heat_release_fraction"] = (
                float(cumulative_positive_heat_release / total_positive_heat_release)
                if total_positive_heat_release > 0.0
                else 0.0
            )

        by_angle = {float(point["crank_angle_deg"]): point for point in heat_release_trace}
        for entry in enriched_trace:
            point = by_angle.get(float(entry["crank_angle_deg"]))
            entry["apparent_heat_release_step_J"] = (
                None if point is None else float(point["apparent_heat_release_step_J"])
            )
            entry["cumulative_positive_heat_release_fraction"] = (
                None if point is None else float(point["cumulative_positive_heat_release_fraction"])
            )
    else:
        for entry in enriched_trace:
            entry["apparent_heat_release_step_J"] = None
            entry["cumulative_positive_heat_release_fraction"] = None

    ca10 = ca50 = ca90 = None
    if heat_release_trace and total_positive_heat_release > 0.0:
        ca10 = _find_fraction_angle(heat_release_trace, fraction=0.10)
        ca50 = _find_fraction_angle(heat_release_trace, fraction=0.50)
        ca90 = _find_fraction_angle(heat_release_trace, fraction=0.90)

    imep = None
    if (
        displacement_volume
        and displacement_volume > 0.0
        and angle_span is not None
        and abs(angle_span) >= 300.0
    ):
        imep = float(net_work / displacement_volume)

    results = {
        "engine_case_dir": str(run_dir),
        "run_ok": bool(run_ok) if run_ok is not None else None,
        "stage": str(stage),
        "solver_name": str(solver_name),
        "handoff_bundle_id": str(handoff_bundle_id),
        "mechanism_id": str(mechanism_id),
        "openfoam_chemistry_package_id": str(openfoam_chemistry_package_id),
        "openfoam_chemistry_package_hash": str(openfoam_chemistry_package_hash),
        "custom_solver_source_hash": str(custom_solver_source_hash),
        "trace_point_count": len(enriched_trace),
        "trace_angle_span_deg": angle_span,
        "displacement_volume_m3": displacement_volume,
        "stage_boundaries": _load_stage_boundaries(run_dir),
        "numeric_stability": _collect_floor_hit_counts(run_dir),
        "metrics": {
            "peak_pressure_Pa": peak_pressure,
            "peak_pressure_crank_angle_deg": peak_pressure_angle,
            "ca10_deg": ca10,
            "ca50_deg": ca50,
            "ca90_deg": ca90,
            "imep_Pa": imep,
            "net_indicated_work_J": float(net_work) if heat_release_trace else None,
            "total_positive_apparent_heat_release_J": (
                float(total_positive_heat_release) if heat_release_trace else None
            ),
            "trapped_mass": (
                None
                if "trapped_mass" not in resolved_metrics
                else float(resolved_metrics["trapped_mass"])
            ),
            "residual_fraction": (
                None
                if "residual_fraction" not in resolved_metrics
                else float(resolved_metrics["residual_fraction"])
            ),
            "trapped_o2_mass": (
                None
                if "trapped_o2_mass" not in resolved_metrics
                else float(resolved_metrics["trapped_o2_mass"])
            ),
            "scavenging_efficiency": (
                None
                if "scavenging_efficiency" not in resolved_metrics
                else float(resolved_metrics["scavenging_efficiency"])
            ),
        },
        "trace": enriched_trace,
        "sources": {
            "openfoam_metrics_path": str(run_dir / "openfoam_metrics.json"),
            "engine_stage_manifest_path": str(run_dir / "engine_stage_manifest.json"),
            "time_dir_count": len(time_dirs),
            "log_summary_count": len(trace),
        },
    }
    return results


def emit_engine_results_artifact(
    *,
    engine_case_dir: str | Path,
    params: dict[str, Any] | None = None,
    engine_metrics: dict[str, Any] | None = None,
    solver_name: str = "",
    handoff_bundle_id: str = "",
    mechanism_id: str = "",
    openfoam_chemistry_package_id: str = "",
    openfoam_chemistry_package_hash: str = "",
    custom_solver_source_hash: str = "",
    run_ok: bool | None = None,
    stage: str = "",
    artifact_name: str = "engine_results.json",
    gamma: float = DEFAULT_APPARENT_HEAT_RELEASE_GAMMA,
) -> dict[str, Any]:
    payload = build_engine_results(
        engine_case_dir=engine_case_dir,
        params=params,
        engine_metrics=engine_metrics,
        solver_name=solver_name,
        handoff_bundle_id=handoff_bundle_id,
        mechanism_id=mechanism_id,
        openfoam_chemistry_package_id=openfoam_chemistry_package_id,
        openfoam_chemistry_package_hash=openfoam_chemistry_package_hash,
        custom_solver_source_hash=custom_solver_source_hash,
        run_ok=run_ok,
        stage=stage,
        gamma=gamma,
    )
    artifact_path = Path(engine_case_dir) / artifact_name
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


__all__ = [
    "build_engine_progress_summary",
    "build_engine_results",
    "emit_engine_progress_artifacts",
    "emit_engine_results_artifact",
]
