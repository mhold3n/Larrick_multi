"""Artifact-only extraction for restart regression analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_existing_path(
    *,
    candidate: str | Path | None,
    roots: list[Path],
) -> Path | None:
    if candidate is None:
        return None
    raw = str(candidate).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path if path.exists() else None
    for root in roots:
        resolved = (root / path).resolve()
        if resolved.exists():
            return resolved
    return None


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _flatten_numeric_json(
    *,
    payload: Any,
    artifact_slot_id: str,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []

    def _walk(node: Any, path: str) -> None:
        numeric = _coerce_numeric(node)
        if numeric is not None:
            slots.append(
                {
                    "artifact_slot_id": artifact_slot_id,
                    "position_index": len(slots),
                    "value": numeric,
                    "metadata": {
                        "json_path": path,
                        "semantic_label": path,
                    },
                }
            )
            return
        if isinstance(node, dict):
            for key in sorted(node):
                next_path = f"{path}.{key}" if path else str(key)
                _walk(node[key], next_path)
            return
        if isinstance(node, list):
            for index, item in enumerate(node):
                next_path = f"{path}[{index}]"
                _walk(item, next_path)

    _walk(payload, "")
    return slots


def _load_selected_profile(
    summary_path: Path,
    *,
    profile_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    summary_payload = _load_json(summary_path)
    profiles = list(summary_payload.get("profiles", []) or [])
    if not profiles:
        raise ValueError(f"No profiles found in {summary_path}")
    selected_profile: dict[str, Any]
    if profile_name is None:
        if len(profiles) != 1:
            raise ValueError(
                f"{summary_path} defines {len(profiles)} profiles; --profile-name is required"
            )
        selected_profile = dict(profiles[0])
    else:
        matches = [
            dict(item) for item in profiles if str(item.get("profile_name", "")) == profile_name
        ]
        if len(matches) != 1:
            raise ValueError(f"Profile '{profile_name}' not found uniquely in {summary_path}")
        selected_profile = matches[0]
    normalized_summary = dict(summary_payload)
    normalized_summary["selected_profile"] = selected_profile
    normalized_summary.pop("profiles", None)
    normalized_summary["selected_profile_name"] = str(selected_profile.get("profile_name", ""))
    return normalized_summary, selected_profile, str(selected_profile.get("profile_name", ""))


def _parse_runtime_summary_series(run_case_dir: Path) -> list[dict[str, Any]]:
    files = sorted(
        run_case_dir.glob("runtimeChemistrySummary.*.dat"),
        key=lambda path: float(path.name[len("runtimeChemistrySummary.") : -len(".dat")]),
    )
    if not files:
        return []

    x_values: list[float] = []
    header: list[str] = []
    rows: list[list[float]] = []
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        data_lines = [line.strip() for line in lines if line.strip()]
        if not data_lines:
            continue
        header_line = next((line for line in data_lines if line.startswith("#")), "")
        if header_line:
            header = header_line.lstrip("#").split()
        numeric_rows = [
            [float(token) for token in line.split()]
            for line in data_lines
            if not line.startswith("#")
        ]
        if not numeric_rows:
            continue
        x_values.append(float(path.name[len("runtimeChemistrySummary.") : -len(".dat")]))
        rows.append(numeric_rows[-1])

    if not rows:
        return []

    column_count = min(len(row) for row in rows)
    if not header:
        header = [f"column_{index}" for index in range(column_count)]
    if len(header) < column_count:
        header.extend(f"column_{index}" for index in range(len(header), column_count))

    series: list[dict[str, Any]] = []
    matrix = np.asarray([row[:column_count] for row in rows], dtype=float)
    for index in range(column_count):
        label = header[index]
        series.append(
            {
                "artifact_slot_id": "runtimeChemistrySummary.dat",
                "position_index": index,
                "series_id": f"runtimeChemistrySummary.dat:{index}",
                "x_values": [float(value) for value in x_values],
                "y_values": [float(value) for value in matrix[:, index]],
                "metadata": {
                    "column_name": label,
                    "semantic_label": label,
                    "file_count": len(files),
                },
            }
        )
    return series


def _column_labels_for_matrix(
    *,
    key: str,
    width: int,
    state_labels: list[str] | None,
) -> list[str]:
    if state_labels and width == len(state_labels):
        return list(state_labels)
    return [f"{key}[{index}]" for index in range(width)]


def _extract_dense_from_matrix_bundle(
    *,
    bundle: dict[str, np.ndarray],
    bundle_name: str,
    state_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    dense: list[dict[str, Any]] = []
    for key in sorted(bundle):
        array = np.asarray(bundle[key])
        if array.size == 0 or not np.issubdtype(array.dtype, np.number):
            continue
        if array.ndim == 0:
            continue
        if array.ndim == 1:
            dense.append(
                {
                    "artifact_slot_id": f"{bundle_name}.{key}",
                    "position_index": 0,
                    "series_id": f"{bundle_name}.{key}:0",
                    "x_values": [float(index) for index in range(int(array.shape[0]))],
                    "y_values": [float(value) for value in array.astype(float)],
                    "metadata": {
                        "column_name": key,
                        "semantic_label": key,
                    },
                }
            )
            continue
        matrix = array.reshape(array.shape[0], -1).astype(float)
        labels = _column_labels_for_matrix(
            key=key, width=matrix.shape[1], state_labels=state_labels
        )
        for index in range(matrix.shape[1]):
            dense.append(
                {
                    "artifact_slot_id": f"{bundle_name}.{key}",
                    "position_index": index,
                    "series_id": f"{bundle_name}.{key}:{index}",
                    "x_values": [float(row_index) for row_index in range(int(matrix.shape[0]))],
                    "y_values": [float(value) for value in matrix[:, index]],
                    "metadata": {
                        "column_name": labels[index],
                        "semantic_label": labels[index],
                    },
                }
            )
    return dense


def _parse_coverage_corpus_npz(path: Path) -> list[dict[str, Any]]:
    payload = np.load(path, allow_pickle=True)
    state_labels = None
    if "state_variables" in payload.files:
        state_labels = [str(value) for value in payload["state_variables"].tolist()]
    numeric_bundle: dict[str, np.ndarray] = {}
    for key in payload.files:
        array = np.asarray(payload[key])
        if np.issubdtype(array.dtype, np.number):
            numeric_bundle[key] = array
    return _extract_dense_from_matrix_bundle(
        bundle=numeric_bundle,
        bundle_name="runtimeChemistryCoverageCorpus.npz",
        state_labels=state_labels,
    )


def _numeric_list(values: list[Any]) -> np.ndarray:
    return np.asarray(
        [float(_coerce_numeric(value) or float("nan")) for value in values], dtype=float
    )


def _parse_coverage_corpus_json(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows = list(payload.get("rows", []) or [])
    if not rows:
        return []
    state_labels = [str(value) for value in list(payload.get("state_variables", []) or [])]
    if not state_labels:
        first_raw = dict(rows[0].get("raw_state", {}) or {})
        state_labels = sorted(
            key for key in first_raw if _coerce_numeric(first_raw[key]) is not None
        )

    raw_states = np.asarray(
        [
            [
                float(
                    _coerce_numeric(dict(row.get("raw_state", {}) or {}).get(label)) or float("nan")
                )
                for label in state_labels
            ]
            for row in rows
        ],
        dtype=float,
    )
    transformed_states = np.asarray(
        [
            [
                float(
                    _coerce_numeric(dict(row.get("transformed_state", {}) or {}).get(label))
                    or float("nan")
                )
                for label in state_labels
            ]
            for row in rows
        ],
        dtype=float,
    )
    bucket_width = max(len(list(row.get("coverage_bucket_key", []) or [])) for row in rows)
    coverage_bucket_keys = np.asarray(
        [
            [
                float(
                    _coerce_numeric(
                        (
                            list(row.get("coverage_bucket_key", []) or [])
                            + [float("nan")] * bucket_width
                        )[index]
                    )
                    or float("nan")
                )
                for index in range(bucket_width)
            ]
            for row in rows
        ],
        dtype=float,
    )
    numeric_bundle: dict[str, np.ndarray] = {
        "coverage_bucket_keys": coverage_bucket_keys,
        "raw_states": raw_states,
        "transformed_states": transformed_states,
        "query_counts": _numeric_list([row.get("query_count") for row in rows]),
        "table_hit_counts": _numeric_list([row.get("table_hit_count") for row in rows]),
        "coverage_reject_counts": _numeric_list([row.get("coverage_reject_count") for row in rows]),
        "trust_reject_counts": _numeric_list([row.get("trust_reject_count") for row in rows]),
        "worst_reject_excess": _numeric_list([row.get("worst_reject_excess") for row in rows]),
        "nearest_sample_distance_min": _numeric_list(
            [row.get("nearest_sample_distance_min") for row in rows]
        ),
        "nearest_sample_distance_max": _numeric_list(
            [row.get("nearest_sample_distance_max") for row in rows]
        ),
    }
    return _extract_dense_from_matrix_bundle(
        bundle=numeric_bundle,
        bundle_name="runtimeChemistryCoverageCorpus.json",
        state_labels=state_labels,
    )


def _parse_engine_results_dense(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    trace = list(payload.get("trace", []) or [])
    if not trace:
        return []
    numeric_keys = sorted(
        key for key in trace[0] if any(_coerce_numeric(row.get(key)) is not None for row in trace)
    )
    if not numeric_keys:
        return []
    x_key = (
        "crank_angle_deg"
        if all(_coerce_numeric(row.get("crank_angle_deg")) is not None for row in trace)
        else None
    )
    x_values = (
        [float(row["crank_angle_deg"]) for row in trace]
        if x_key is not None
        else [float(index) for index in range(len(trace))]
    )
    dense: list[dict[str, Any]] = []
    for index, key in enumerate(numeric_keys):
        y_values = [float(_coerce_numeric(row.get(key)) or float("nan")) for row in trace]
        dense.append(
            {
                "artifact_slot_id": "engine_results.trace",
                "position_index": index,
                "series_id": f"engine_results.trace:{index}",
                "x_values": x_values,
                "y_values": y_values,
                "metadata": {
                    "column_name": key,
                    "semantic_label": key,
                    "x_axis": x_key or "trace_index",
                },
            }
        )
    return dense


def extract_restart_run_artifacts(
    *,
    run_dir: str | Path,
    profile_name: str | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    summary_path = root / "engine_restart_benchmark_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Restart benchmark summary not found: {summary_path}")

    normalized_summary, selected_profile, resolved_profile_name = _load_selected_profile(
        summary_path,
        profile_name=profile_name,
    )
    roots = [root, summary_path.parent.resolve(), Path.cwd().resolve()]
    benchmark_run_dir = _resolve_existing_path(
        candidate=selected_profile.get("benchmark_run_dir"),
        roots=roots,
    )
    if benchmark_run_dir is not None:
        roots.insert(0, benchmark_run_dir)

    scalar_slots = _flatten_numeric_json(
        payload=normalized_summary,
        artifact_slot_id="engine_restart_benchmark_summary",
    )

    authority_miss_path = _resolve_existing_path(
        candidate=selected_profile.get("runtime_chemistry_authority_miss_path"),
        roots=roots,
    )
    authority_miss_payload: dict[str, Any] | None = None
    if authority_miss_path is not None:
        authority_miss_payload = _load_json(authority_miss_path)
        scalar_slots.extend(
            _flatten_numeric_json(
                payload=authority_miss_payload,
                artifact_slot_id="runtimeChemistryAuthorityMiss",
            )
        )

    stage_manifest_path = (
        None if benchmark_run_dir is None else benchmark_run_dir / "engine_stage_manifest.json"
    )
    if stage_manifest_path is not None and stage_manifest_path.exists():
        scalar_slots.extend(
            _flatten_numeric_json(
                payload=_load_json(stage_manifest_path),
                artifact_slot_id="engine_stage_manifest",
            )
        )

    engine_results_path = (
        None if benchmark_run_dir is None else benchmark_run_dir / "engine_results.json"
    )
    if engine_results_path is not None and engine_results_path.exists():
        engine_results_payload = _load_json(engine_results_path)
        scalar_slots.extend(
            _flatten_numeric_json(
                payload=engine_results_payload,
                artifact_slot_id="engine_results",
            )
        )
    else:
        engine_results_payload = None

    dense_series: list[dict[str, Any]] = []
    if benchmark_run_dir is not None:
        dense_series.extend(_parse_runtime_summary_series(benchmark_run_dir))

    coverage_npz_path = _resolve_existing_path(
        candidate=selected_profile.get("runtime_coverage_corpus_npz_path"),
        roots=roots,
    )
    coverage_json_path = _resolve_existing_path(
        candidate=selected_profile.get("runtime_coverage_corpus_path"),
        roots=roots,
    )
    if coverage_npz_path is not None:
        dense_series.extend(_parse_coverage_corpus_npz(coverage_npz_path))
    elif coverage_json_path is not None:
        dense_series.extend(_parse_coverage_corpus_json(coverage_json_path))

    if engine_results_path is not None and engine_results_path.exists():
        dense_series.extend(_parse_engine_results_dense(engine_results_path))

    return {
        "run_id": root.name,
        "run_dir": str(root),
        "profile_name": resolved_profile_name,
        "benchmark_run_dir": None if benchmark_run_dir is None else str(benchmark_run_dir),
        "scalar_slots": scalar_slots,
        "dense_series": dense_series,
        "summary_payload": normalized_summary,
        "selected_profile": selected_profile,
        "authority_miss_payload": authority_miss_payload,
        "artifact_paths": {
            "summary_path": str(summary_path),
            "authority_miss_path": None
            if authority_miss_path is None
            else str(authority_miss_path),
            "coverage_npz_path": None if coverage_npz_path is None else str(coverage_npz_path),
            "coverage_json_path": None if coverage_json_path is None else str(coverage_json_path),
            "stage_manifest_path": (
                None
                if stage_manifest_path is None or not stage_manifest_path.exists()
                else str(stage_manifest_path)
            ),
            "engine_results_path": (
                None
                if engine_results_path is None or not engine_results_path.exists()
                else str(engine_results_path)
            ),
        },
    }


__all__ = ["extract_restart_run_artifacts"]
