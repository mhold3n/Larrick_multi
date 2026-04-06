"""Numeric scoring for artifact-only restart regression analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SLOT_FAMILY_CONTEXT = "context"
SLOT_FAMILY_OPERATIONAL = "operational"
SLOT_FAMILY_FAILURE = "failure"
SLOT_FAMILY_DENSE = "dense_derived"

_FAMILY_PRIORITY_WEIGHTS = {
    SLOT_FAMILY_CONTEXT: 0.15,
    SLOT_FAMILY_OPERATIONAL: 1.15,
    SLOT_FAMILY_FAILURE: 1.45,
    SLOT_FAMILY_DENSE: 1.0,
}

_FAMILY_SCALE_MULTIPLIERS = {
    SLOT_FAMILY_CONTEXT: 5.0e-2,
    SLOT_FAMILY_OPERATIONAL: 1.0e-2,
    SLOT_FAMILY_FAILURE: 1.0e-2,
    SLOT_FAMILY_DENSE: 1.0e-2,
}

_FAMILY_ABSOLUTE_FLOORS = {
    SLOT_FAMILY_CONTEXT: 1.0,
    SLOT_FAMILY_OPERATIONAL: 1.0e-3,
    SLOT_FAMILY_FAILURE: 1.0e-4,
    SLOT_FAMILY_DENSE: 1.0e-3,
}

_CONTEXT_LABEL_TOKENS = (
    "baseline_start.",
    "runtime_table_id",
    "runtime_table_hash",
    "runtime_table_dir",
    "runtime_package_id",
    "runtime_package_hash",
    "benchmark_run_dir",
    "base_run_dir",
    "checkpoint_angle_deg",
    "checkpoint_time_s",
    "resolved_profile",
    "profile_name",
    "target_end_angle_deg",
    "writeinterval",
    "end_angle_deg",
    "engine_min_",
    "engine_max_",
    "mean_pressure_pa",
    "mean_temperature_k",
)

_OPERATIONAL_LABEL_TOKENS = (
    "angle_advance_deg",
    "latest_checkpoint",
    "table_hit_fraction",
    "coverage_reject_cell",
    "coverage_reject_cells",
    "fallback_timesteps",
    "trust_region_reject_cells",
    "total_numeric_hits",
    "solver_ok",
    "stage_result",
    "wall_seconds_per_0p01deg",
    "wall_elapsed_s",
    "speed_score",
    "sim_time_advance_s",
    "executed_stage_count",
    "executed_stage_names",
    "executed_stage_runtime_tables",
    "latest_angle_deg",
    "runtime_summary_count",
    "table_hit_cells",
    "table_query_cells",
    "interpolation_cache_hit",
    "chem323_maturity_gate_passed",
    "chem323_runtime_replacement_gate_passed",
)

_FAILURE_LABEL_TOKENS = (
    "first_miss",
    "first_offending",
    "failure_class",
    "failure_branch",
    "reject_",
    "qdot",
    "miss_counts_by_variable",
    "max_out_of_bound_by_variable",
    "trust_reject_cell_count",
    "uncovered_cell_count",
    "crossed_local_stencil_sign",
    "sign_flip",
    "offending",
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _median_abs_deviation(values: list[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def _robust_scale(values: list[float]) -> float:
    mad = _median_abs_deviation(values)
    if mad > 1.0e-9:
        return mad
    if len(values) >= 2:
        diffs = np.diff(np.asarray(values, dtype=float))
        diff_mad = float(np.median(np.abs(diffs - np.median(diffs))))
        if diff_mad > 1.0e-9:
            return diff_mad
    max_abs = max((abs(value) for value in values), default=0.0)
    return max(max_abs * 1.0e-6, 1.0e-9)


def _rolling_slope(values: list[float | None], *, history_window: int) -> float | None:
    indexed = [(index, value) for index, value in enumerate(values) if value is not None]
    if len(indexed) < 2:
        return None
    window = indexed[-max(int(history_window), 2) :]
    x = np.asarray([item[0] for item in window], dtype=float)
    y = np.asarray([float(item[1]) for item in window], dtype=float)
    if np.allclose(x, x[0]):
        return None
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _series_from_slot_map(
    *,
    run_items: list[dict[str, Any]],
    item_key: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    per_run_maps: list[dict[tuple[str, int], dict[str, Any]]] = []
    max_position_by_artifact: dict[str, int] = {}
    observed_metadata: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for run in run_items:
        current: dict[tuple[str, int], dict[str, Any]] = {}
        for entry in list(run.get(item_key, []) or []):
            key = (str(entry["artifact_slot_id"]), int(entry["position_index"]))
            current[key] = entry
            max_position_by_artifact[key[0]] = max(max_position_by_artifact.get(key[0], -1), key[1])
            observed_metadata.setdefault(key, {})[str(run["run_id"])] = dict(
                entry.get("metadata", {}) or {}
            )
        per_run_maps.append(current)

    manifest: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    run_ids = [str(run["run_id"]) for run in run_items]
    for artifact_slot_id in sorted(max_position_by_artifact):
        for position_index in range(max_position_by_artifact[artifact_slot_id] + 1):
            key = (artifact_slot_id, position_index)
            entries = [run_map.get(key) for run_map in per_run_maps]
            latest_metadata = next(
                (
                    dict(entry.get("metadata", {}) or {})
                    for entry in reversed(entries)
                    if entry is not None
                ),
                {},
            )
            latest_label = str(
                latest_metadata.get("semantic_label")
                or latest_metadata.get("json_path")
                or latest_metadata.get("column_name")
                or ""
            )
            observed = observed_metadata.get(key, {})
            is_dense = item_key == "dense_series"
            slot_family = (
                classify_dense_series_family(
                    artifact_slot_id=artifact_slot_id,
                    semantic_label=latest_label,
                    metadata=latest_metadata,
                )
                if is_dense
                else classify_scalar_slot_family(
                    artifact_slot_id=artifact_slot_id,
                    semantic_label=latest_label,
                    metadata=latest_metadata,
                )
            )
            manifest.append(
                {
                    "slot_or_series_id": f"{artifact_slot_id}:{position_index}",
                    "artifact_slot_id": artifact_slot_id,
                    "position_index": position_index,
                    "presence_mask": [entry is not None for entry in entries],
                    "latest_semantic_label": latest_label,
                    "slot_family": slot_family,
                    "observed_metadata_by_run": observed,
                    "run_ids": run_ids,
                }
            )
            grouped[key] = entries
    return manifest, grouped


def _classify_from_label(artifact_slot_id: str, semantic_label: str) -> str:
    artifact = str(artifact_slot_id).lower()
    label = str(semantic_label).lower()
    if artifact == "runtimechemistryauthoritymiss":
        return SLOT_FAMILY_FAILURE
    if any(token in label for token in _FAILURE_LABEL_TOKENS):
        return SLOT_FAMILY_FAILURE
    if any(token in label for token in _OPERATIONAL_LABEL_TOKENS):
        return SLOT_FAMILY_OPERATIONAL
    if artifact == "engine_stage_manifest":
        if any(
            token in label
            for token in ("latest_angle_deg", "ok", "stage_result", "runtime_chemistry_")
        ):
            return SLOT_FAMILY_OPERATIONAL
        return SLOT_FAMILY_CONTEXT
    if any(token in label for token in _CONTEXT_LABEL_TOKENS):
        return SLOT_FAMILY_CONTEXT
    if artifact == "engine_restart_benchmark_summary":
        return SLOT_FAMILY_OPERATIONAL
    if artifact == "engine_results":
        return SLOT_FAMILY_CONTEXT
    return SLOT_FAMILY_CONTEXT


def classify_scalar_slot_family(
    *,
    artifact_slot_id: str,
    semantic_label: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    del metadata
    return _classify_from_label(artifact_slot_id, semantic_label)


def classify_dense_series_family(
    *,
    artifact_slot_id: str,
    semantic_label: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    del artifact_slot_id, semantic_label, metadata
    return SLOT_FAMILY_DENSE


def _scale_reference(values: list[float | None]) -> float:
    finite = [abs(float(value)) for value in values if value is not None and math.isfinite(value)]
    return max(finite, default=1.0)


def _scale_floor(*, slot_family: str, values: list[float | None]) -> float:
    reference = _scale_reference(values)
    multiplier = float(_FAMILY_SCALE_MULTIPLIERS.get(slot_family, 1.0e-2))
    absolute_floor = float(_FAMILY_ABSOLUTE_FLOORS.get(slot_family, 1.0e-3))
    return max(reference * multiplier, absolute_floor)


def _compress_score(value: float) -> float:
    return float(math.log1p(max(value, 0.0)))


def _focus_eligibility(slot_family: str, *, normalized_drift: float | None) -> tuple[bool, str]:
    if slot_family == SLOT_FAMILY_CONTEXT:
        return False, "context_slot"
    if normalized_drift is None:
        return False, "missing_comparable_baseline"
    return True, "eligible"


def _scalar_priority(
    *,
    slot_family: str,
    raw_scale: float,
    scale_floor: float,
    delta: float | None,
    current_deviation: float | None,
    previous_deviation: float | None,
    appearance_score: float,
    disappearance_score: float,
) -> tuple[float, dict[str, float], float, bool]:
    scale_used = max(raw_scale, scale_floor)
    scale_floor_applied = scale_used > raw_scale
    relative_delta = (
        0.0
        if delta is None
        else float(abs(delta) / max(scale_used, _FAMILY_ABSOLUTE_FLOORS[SLOT_FAMILY_OPERATIONAL]))
    )
    relative_shift = (
        0.0
        if current_deviation is None or previous_deviation is None
        else float(max(abs(current_deviation - previous_deviation), 0.0) / max(scale_used, 1.0e-12))
    )
    compressed_delta = _compress_score(relative_delta)
    compressed_shift = _compress_score(relative_shift)
    appearance_component = float(appearance_score) * 1.25
    disappearance_component = float(disappearance_score) * 1.25
    family_weight = float(_FAMILY_PRIORITY_WEIGHTS.get(slot_family, 1.0))
    priority_score = family_weight * (
        compressed_delta
        + 0.75 * compressed_shift
        + appearance_component
        + disappearance_component
    )
    components = {
        "compressed_delta": compressed_delta,
        "compressed_shift": compressed_shift,
        "appearance": appearance_component,
        "disappearance": disappearance_component,
        "family_weight": family_weight,
    }
    return float(priority_score), components, float(scale_used), bool(scale_floor_applied)


def analyze_scalar_slots(
    *,
    run_items: list[dict[str, Any]],
    history_window: int = 5,
) -> dict[str, Any]:
    if len(run_items) < 2:
        return {
            "slot_manifest": [],
            "records": [],
            "top_leverage": [],
        }

    manifest, grouped = _series_from_slot_map(run_items=run_items, item_key="scalar_slots")
    records: list[dict[str, Any]] = []
    latest_run_id = str(run_items[-1]["run_id"])
    baseline_run_id = str(run_items[-2]["run_id"])

    for entry in manifest:
        key = (str(entry["artifact_slot_id"]), int(entry["position_index"]))
        series = grouped[key]
        values = [_safe_float(item["value"]) if item is not None else None for item in series]
        latest_value = values[-1]
        previous_value = values[-2]
        delta = None
        if latest_value is not None and previous_value is not None:
            delta = float(latest_value - previous_value)

        recent_values = [
            value for value in values[-max(history_window + 1, 2) :] if value is not None
        ]
        historical_values = [
            value for value in values[:-1][-max(history_window, 1) :] if value is not None
        ]
        center = None if not historical_values else float(np.median(np.asarray(historical_values)))
        raw_scale = _robust_scale(historical_values or recent_values)
        scale_floor = _scale_floor(
            slot_family=str(entry["slot_family"]),
            values=historical_values + [previous_value, latest_value],
        )
        current_deviation = (
            None if latest_value is None or center is None else abs(float(latest_value) - center)
        )
        previous_deviation = (
            None
            if previous_value is None or center is None
            else abs(float(previous_value) - center)
        )
        normalized_drift = None if delta is None else float(abs(delta) / raw_scale)
        regression_signal = (
            0.0
            if current_deviation is None or previous_deviation is None
            else float(max(current_deviation - previous_deviation, 0.0) / raw_scale)
        )
        improvement_signal = (
            0.0
            if current_deviation is None or previous_deviation is None
            else float(max(previous_deviation - current_deviation, 0.0) / raw_scale)
        )
        appearance_score = 1.0 if previous_value is None and latest_value is not None else 0.0
        disappearance_score = 1.0 if previous_value is not None and latest_value is None else 0.0
        confidence = min(len(recent_values) / max(history_window, 1), 1.0)
        priority_score, priority_components, scale_used, scale_floor_applied = _scalar_priority(
            slot_family=str(entry["slot_family"]),
            raw_scale=raw_scale,
            scale_floor=scale_floor,
            delta=delta,
            current_deviation=current_deviation,
            previous_deviation=previous_deviation,
            appearance_score=appearance_score,
            disappearance_score=disappearance_score,
        )
        focus_eligible, focus_reason = _focus_eligibility(
            str(entry["slot_family"]),
            normalized_drift=normalized_drift,
        )
        records.append(
            {
                "run_id": latest_run_id,
                "baseline_run_id": baseline_run_id,
                "slot_or_series_id": entry["slot_or_series_id"],
                "artifact_slot_id": entry["artifact_slot_id"],
                "position_index": entry["position_index"],
                "semantic_label": entry["latest_semantic_label"],
                "slot_family": entry["slot_family"],
                "latest_value": latest_value,
                "baseline_value": previous_value,
                "delta": delta,
                "normalized_drift_score": normalized_drift,
                "trend_slope": _rolling_slope(values, history_window=history_window),
                "regression_signal": regression_signal,
                "improvement_signal": improvement_signal,
                "appearance_score": appearance_score,
                "disappearance_score": disappearance_score,
                "priority_score": priority_score,
                "priority_components": priority_components,
                "scale_used": scale_used,
                "scale_floor_applied": scale_floor_applied,
                "excluded_from_focus": not focus_eligible,
                "focus_eligibility_reason": focus_reason,
                "breakpoint_detected": bool(priority_score >= 3.0),
                "confidence": float(confidence),
            }
        )

    leverage = sorted(
        records,
        key=lambda item: (
            0 if bool(item["excluded_from_focus"]) else 1,
            float(item["priority_score"] or 0.0),
        ),
        reverse=True,
    )
    return {
        "slot_manifest": manifest,
        "records": records,
        "top_leverage": leverage[:25],
    }


def _resample_series(y_values: list[float], *, count: int = 64) -> np.ndarray:
    if not y_values:
        return np.asarray([], dtype=float)
    source = np.asarray(y_values, dtype=float)
    if source.size == 1:
        return np.repeat(source, count)
    source_x = np.linspace(0.0, 1.0, num=source.size)
    target_x = np.linspace(0.0, 1.0, num=count)
    finite_mask = np.isfinite(source)
    if finite_mask.sum() < 2:
        fill_value = float(np.nanmean(source)) if finite_mask.any() else 0.0
        return np.repeat(fill_value, count)
    return np.interp(target_x, source_x[finite_mask], source[finite_mask])


def _dominant_windows(abs_diff: np.ndarray) -> list[dict[str, Any]]:
    if abs_diff.size == 0:
        return []
    top_indices = np.argsort(abs_diff)[::-1][:3]
    windows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw_index in top_indices:
        start = max(int(raw_index) - 1, 0)
        end = min(int(raw_index) + 1, int(abs_diff.size) - 1)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        windows.append(
            {
                "start_fraction": float(start / max(abs_diff.size - 1, 1)),
                "end_fraction": float(end / max(abs_diff.size - 1, 1)),
                "mean_abs_difference": float(np.mean(abs_diff[start : end + 1])),
            }
        )
    return windows


def _dense_collapse_priority(
    *,
    semantic_label: str,
    latest_final: float | None,
    previous_final: float | None,
    latest_length: int,
    previous_length: int,
    first_divergence_fraction: float | None,
    area_between: float,
    raw_scale: float,
    slot_family: str,
) -> tuple[float, dict[str, float | bool], float, bool, str]:
    scale_floor = _scale_floor(
        slot_family=slot_family,
        values=[latest_final, previous_final, area_between, float(latest_length), float(previous_length)],
    )
    scale_used = max(raw_scale, scale_floor)
    scale_floor_applied = scale_used > raw_scale
    truncation_depth_delta = int(latest_length - previous_length)
    latest_shorter = latest_length < previous_length
    truncation_fraction = (
        float(max(previous_length - latest_length, 0) / max(previous_length, 1))
        if previous_length
        else 0.0
    )
    early_divergence = (
        first_divergence_fraction is not None and float(first_divergence_fraction) <= 0.25
    )
    early_collapse_detected = bool(
        latest_shorter and (early_divergence or first_divergence_fraction is None or truncation_fraction >= 0.25)
    )
    table_hit_collapse_detected = bool(
        semantic_label == "tableHitCells"
        and previous_final not in (None, 0.0)
        and latest_final is not None
        and float(latest_final) / float(previous_final) <= 0.5
    )
    query_collapse_detected = bool(
        semantic_label == "tableQueryCells"
        and previous_final not in (None, 0.0)
        and latest_final is not None
        and float(latest_final) / float(previous_final) <= 0.5
    )
    trust_reject_onset = bool(
        semantic_label == "trustRegionRejectCells"
        and previous_final is not None
        and latest_final is not None
        and float(latest_final) > float(previous_final)
        and (first_divergence_fraction is None or float(first_divergence_fraction) <= 0.25)
    )
    base_distance = _compress_score(float(area_between / scale_used))
    divergence_bonus = (
        0.0
        if first_divergence_fraction is None
        else float(max(1.0 - float(first_divergence_fraction), 0.0) * 1.75)
    )
    priority_score = float(_FAMILY_PRIORITY_WEIGHTS[slot_family] * base_distance)
    priority_score += truncation_fraction * 4.0
    priority_score += divergence_bonus
    if early_collapse_detected:
        priority_score += 2.5
    if table_hit_collapse_detected or query_collapse_detected:
        priority_score += 2.0
    if trust_reject_onset:
        priority_score += 1.75

    if early_collapse_detected:
        collapse_classification = "early_collapse"
    elif latest_shorter:
        collapse_classification = "truncated"
    elif trust_reject_onset:
        collapse_classification = "trust_reject_onset"
    elif table_hit_collapse_detected or query_collapse_detected:
        collapse_classification = "hit_collapse"
    else:
        collapse_classification = "stable_divergence"

    components: dict[str, float | bool] = {
        "base_distance": base_distance,
        "truncation_fraction": truncation_fraction,
        "divergence_bonus": divergence_bonus,
        "early_collapse_detected": early_collapse_detected,
        "table_hit_collapse_detected": table_hit_collapse_detected,
        "query_collapse_detected": query_collapse_detected,
        "trust_reject_onset": trust_reject_onset,
        "family_weight": float(_FAMILY_PRIORITY_WEIGHTS[slot_family]),
        "truncation_depth_delta": float(truncation_depth_delta),
    }
    return priority_score, components, float(scale_used), bool(scale_floor_applied), collapse_classification


def analyze_dense_series(
    *,
    run_items: list[dict[str, Any]],
    history_window: int = 5,
) -> dict[str, Any]:
    if len(run_items) < 2:
        return {
            "series_manifest": [],
            "records": [],
            "top_divergence": [],
        }

    manifest, grouped = _series_from_slot_map(run_items=run_items, item_key="dense_series")
    records: list[dict[str, Any]] = []
    latest_run_id = str(run_items[-1]["run_id"])
    baseline_run_id = str(run_items[-2]["run_id"])

    for entry in manifest:
        key = (str(entry["artifact_slot_id"]), int(entry["position_index"]))
        series_entries = grouped[key]
        latest_entry = series_entries[-1]
        previous_entry = series_entries[-2]
        if latest_entry is None or previous_entry is None:
            records.append(
                {
                    "run_id": latest_run_id,
                    "baseline_run_id": baseline_run_id,
                    "slot_or_series_id": entry["slot_or_series_id"],
                    "artifact_slot_id": entry["artifact_slot_id"],
                    "position_index": entry["position_index"],
                    "semantic_label": entry["latest_semantic_label"],
                    "slot_family": entry["slot_family"],
                    "delta": None,
                    "normalized_drift_score": None,
                    "trend_slope": None,
                    "regression_signal": 0.0,
                    "improvement_signal": 0.0,
                    "confidence": 0.0,
                    "series_length_latest": None if latest_entry is None else len(latest_entry["y_values"]),
                    "series_length_baseline": None if previous_entry is None else len(previous_entry["y_values"]),
                    "truncation_depth_delta": None,
                    "latest_series_shorter_than_baseline": None,
                    "first_divergence_index": None,
                    "first_divergence_fraction": None,
                    "early_collapse_detected": False,
                    "table_hit_collapse_detected": False,
                    "area_between_curves": None,
                    "slope_distance": None,
                    "curvature_distance": None,
                    "raw_distance_metrics": {},
                    "priority_score": 0.0,
                    "priority_components": {},
                    "scale_used": None,
                    "scale_floor_applied": False,
                    "collapse_classification": "missing_dense_series",
                    "excluded_from_focus": True,
                    "focus_eligibility_reason": "missing_comparable_dense_series",
                    "dominant_divergence_windows": [],
                }
            )
            continue

        latest_y = [float(value) for value in latest_entry["y_values"]]
        previous_y = [float(value) for value in previous_entry["y_values"]]
        latest_resampled = _resample_series(latest_y)
        previous_resampled = _resample_series(previous_y)
        abs_diff = np.abs(latest_resampled - previous_resampled)
        area_between = float(np.mean(abs_diff))
        slope_distance = float(
            np.mean(np.abs(np.diff(latest_resampled) - np.diff(previous_resampled)))
        )
        curvature_distance = float(
            np.mean(np.abs(np.diff(latest_resampled, n=2) - np.diff(previous_resampled, n=2)))
        )
        threshold = max(area_between * 1.5, 1.0e-9)
        divergence_indices = np.flatnonzero(abs_diff > threshold)
        first_divergence_index = (
            None if divergence_indices.size == 0 else int(divergence_indices[0])
        )
        first_divergence_fraction = (
            None
            if first_divergence_index is None
            else float(first_divergence_index / max(abs_diff.size - 1, 1))
        )

        prior_resampled: list[np.ndarray] = []
        for historical in series_entries[:-1][-max(history_window, 1) :]:
            if historical is None:
                continue
            prior_resampled.append(
                _resample_series([float(value) for value in historical["y_values"]])
            )
        historical_distances: list[float] = []
        if len(prior_resampled) >= 2:
            baseline = np.median(np.vstack(prior_resampled[:-1]), axis=0)
            latest_dev = float(np.mean(np.abs(latest_resampled - baseline)))
            previous_dev = float(np.mean(np.abs(previous_resampled - baseline)))
            historical_distances = [
                float(np.mean(np.abs(item - baseline))) for item in prior_resampled[:-1]
            ]
        else:
            latest_dev = area_between
            previous_dev = 0.0
            historical_distances = [area_between]
        raw_scale = _robust_scale(historical_distances or [area_between])
        final_values: list[float | None] = []
        for item in series_entries:
            if item is None or not item["y_values"]:
                final_values.append(None)
            else:
                final_values.append(float(item["y_values"][-1]))

        latest_final = float(latest_y[-1]) if latest_y else None
        previous_final = float(previous_y[-1]) if previous_y else None
        delta = (
            None
            if latest_final is None or previous_final is None
            else float(latest_final - previous_final)
        )
        confidence = min(len(prior_resampled) / max(history_window, 1), 1.0)
        priority_score, priority_components, scale_used, scale_floor_applied, collapse_classification = (
            _dense_collapse_priority(
                semantic_label=str(entry["latest_semantic_label"]),
                latest_final=latest_final,
                previous_final=previous_final,
                latest_length=len(latest_y),
                previous_length=len(previous_y),
                first_divergence_fraction=first_divergence_fraction,
                area_between=area_between,
                raw_scale=raw_scale,
                slot_family=str(entry["slot_family"]),
            )
        )
        focus_eligible = collapse_classification != "missing_dense_series"
        focus_reason = "eligible" if focus_eligible else "missing_comparable_dense_series"
        records.append(
            {
                "run_id": latest_run_id,
                "baseline_run_id": baseline_run_id,
                "slot_or_series_id": entry["slot_or_series_id"],
                "artifact_slot_id": entry["artifact_slot_id"],
                "position_index": entry["position_index"],
                "semantic_label": entry["latest_semantic_label"],
                "slot_family": entry["slot_family"],
                "delta": delta,
                "normalized_drift_score": float(area_between / raw_scale),
                "trend_slope": _rolling_slope(final_values, history_window=history_window),
                "regression_signal": float(max(latest_dev - previous_dev, 0.0) / raw_scale),
                "improvement_signal": float(max(previous_dev - latest_dev, 0.0) / raw_scale),
                "confidence": float(confidence),
                "series_length_latest": len(latest_y),
                "series_length_baseline": len(previous_y),
                "truncation_depth_delta": len(latest_y) - len(previous_y),
                "latest_series_shorter_than_baseline": len(latest_y) < len(previous_y),
                "first_divergence_index": first_divergence_index,
                "first_divergence_fraction": first_divergence_fraction,
                "early_collapse_detected": bool(priority_components.get("early_collapse_detected", False)),
                "table_hit_collapse_detected": bool(
                    priority_components.get("table_hit_collapse_detected", False)
                    or priority_components.get("query_collapse_detected", False)
                ),
                "area_between_curves": area_between,
                "slope_distance": slope_distance,
                "curvature_distance": curvature_distance,
                "raw_distance_metrics": {
                    "area_between_curves": area_between,
                    "slope_distance": slope_distance,
                    "curvature_distance": curvature_distance,
                },
                "priority_score": priority_score,
                "priority_components": priority_components,
                "scale_used": scale_used,
                "scale_floor_applied": scale_floor_applied,
                "collapse_classification": collapse_classification,
                "excluded_from_focus": not focus_eligible,
                "focus_eligibility_reason": focus_reason,
                "dominant_divergence_windows": _dominant_windows(abs_diff),
            }
        )

    top_divergence = sorted(
        [record for record in records if not bool(record.get("excluded_from_focus", False))],
        key=lambda item: float(item.get("priority_score") or 0.0),
        reverse=True,
    )
    return {
        "series_manifest": manifest,
        "records": records,
        "top_divergence": top_divergence[:25],
    }


__all__ = [
    "SLOT_FAMILY_CONTEXT",
    "SLOT_FAMILY_DENSE",
    "SLOT_FAMILY_FAILURE",
    "SLOT_FAMILY_OPERATIONAL",
    "analyze_dense_series",
    "analyze_scalar_slots",
    "classify_dense_series_family",
    "classify_scalar_slot_family",
]
