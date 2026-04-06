"""Diagnostics: coverage corpus geometry vs species/Qdot miss targets in transformed state space."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .runtime_chemistry_table import (
    _load_config,
    _load_coverage_corpus_rows,
    _load_current_window_qdot_targets,
    _load_json,
    _load_species_miss_targets,
    _point_key,
    _resolve_repo_relative_path,
    _resolve_transformed_state_variables,
    _state_transform_floor_map,
    _transform_state_vector,
)


def _min_transformed_distance_to_corpus(
    point_state: dict[str, float],
    coverage_rows: list[dict[str, Any]],
    *,
    axis_order: list[str],
    transformed_state_variables: list[str],
    state_transform_floors: dict[str, float],
) -> float | None:
    if not coverage_rows:
        return None
    t_target = _transform_state_vector(
        point_state,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
    )
    best: float | None = None
    for row in coverage_rows:
        t_row = _transform_state_vector(
            row["point_state"],
            axis_order=axis_order,
            transformed_state_variables=transformed_state_variables,
            state_transform_floors=state_transform_floors,
        )
        dist = float(np.linalg.norm(t_target - t_row))
        if best is None or dist < best:
            best = dist
    return best


def analyze_coverage_corpus_vs_targets(
    *,
    table_config_path: str | Path,
    repo_root: Path | None = None,
    extra_corpus_paths: list[str] | None = None,
    authority_miss_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare ignition-entry corpus support to species and Qdot miss targets.

    Loads corpus rows and targets using the same paths and transforms as
    :func:`runtime_chemistry_table.build_runtime_chemistry_table_from_spec`.
    Reports per-target minimum transformed-space distance to the nearest corpus row
    and corpus row / ``_point_key`` deduplication stats.
    """
    root = Path.cwd() if repo_root is None else Path(repo_root)
    table_cfg = _load_config(table_config_path)
    if extra_corpus_paths:
        merged = dict(table_cfg)
        merged["coverage_corpora"] = list(table_cfg.get("coverage_corpora", []) or []) + list(
            extra_corpus_paths
        )
        table_cfg = merged

    state_species = [
        str(item).strip()
        for item in list(table_cfg.get("state_species", []) or [])
        if str(item).strip()
    ]
    axis_order = ["Temperature", "Pressure", *state_species]
    transformed_state_variables = _resolve_transformed_state_variables(
        table_cfg,
        axis_order=axis_order,
    )
    state_transform_floors = _state_transform_floor_map(
        table_cfg,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
    )

    species_targets = _load_species_miss_targets(
        table_cfg,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
        repo_root=root,
    )
    qdot_targets = _load_current_window_qdot_targets(
        table_cfg,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
        repo_root=root,
    )
    coverage_rows, coverage_meta = _load_coverage_corpus_rows(
        table_cfg,
        axis_order=axis_order,
        repo_root=root,
    )
    keys = [_point_key(row["point_state"], axis_order) for row in coverage_rows]
    unique_keys = len(set(keys))

    species_rows: list[dict[str, Any]] = []
    for target in species_targets:
        dist = _min_transformed_distance_to_corpus(
            dict(target["point_state"]),
            coverage_rows,
            axis_order=axis_order,
            transformed_state_variables=transformed_state_variables,
            state_transform_floors=state_transform_floors,
        )
        species_rows.append(
            {
                "reject_variable": str(target.get("reject_variable", "")),
                "stage_name": str(target.get("stage_name", "")),
                "min_transformed_distance_to_corpus": dist,
                "miss_artifact_path": str(target.get("source_path", "")),
            }
        )

    qdot_rows: list[dict[str, Any]] = []
    for target in qdot_targets:
        dist = _min_transformed_distance_to_corpus(
            dict(target["point_state"]),
            coverage_rows,
            axis_order=axis_order,
            transformed_state_variables=transformed_state_variables,
            state_transform_floors=state_transform_floors,
        )
        qdot_rows.append(
            {
                "stage_name": str(target.get("stage_name", "")),
                "min_transformed_distance_to_corpus": dist,
                "miss_artifact_path": str(target.get("source_path", "")),
            }
        )

    out: dict[str, Any] = {
        "table_config_path": str(Path(table_config_path).resolve()),
        "axis_order": list(axis_order),
        "transformed_state_variables": list(transformed_state_variables),
        "corpus": {
            **coverage_meta,
            "loaded_row_count": int(len(coverage_rows)),
            "unique_point_key_count": int(unique_keys),
            "point_key_dedupe_collapsed_rows": max(0, len(coverage_rows) - unique_keys),
        },
        "species_miss_targets": species_rows,
        "qdot_miss_targets": qdot_rows,
    }

    miss_path = str(authority_miss_path or "").strip()
    if miss_path:
        path = _resolve_repo_relative_path(miss_path, repo_root=root)
        if path.exists():
            payload = _load_json(path)
            reject = str(payload.get("reject_variable", "")).strip()
            out["authority_miss_sample"] = {
                "path": str(path),
                "reject_variable": reject,
                "failure_class": str(payload.get("failure_class", "")).strip(),
                "stage_name": str(payload.get("stage_name", "")).strip(),
            }

    return out


def summarize_authority_miss_cluster(paths: list[str | Path], *, repo_root: Path) -> dict[str, Any]:
    """Aggregate reject_variable / failure_class counts from a list of miss JSON paths."""
    reject_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    existing = 0
    for raw in paths:
        path = _resolve_repo_relative_path(str(raw), repo_root=repo_root)
        if not path.exists():
            continue
        existing += 1
        payload = _load_json(path)
        reject_counts[str(payload.get("reject_variable", "")).strip() or "(empty)"] += 1
        failure_counts[str(payload.get("failure_class", "")).strip() or "(empty)"] += 1
    return {
        "artifact_paths_resolved": int(existing),
        "reject_variable_counts": dict(
            sorted(reject_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "failure_class_counts": dict(
            sorted(failure_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
    }
