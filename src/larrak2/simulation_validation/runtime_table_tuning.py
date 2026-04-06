"""Artifact-driven helpers for stage-local runtime-table tuning."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .restart_regression_artifacts import extract_restart_run_artifacts
from .restart_regression_suite import analyze_restart_regression_runs
from .runtime_chemistry_table import (
    _resolve_transformed_state_variables,
    _state_transform_floor_map,
    _transform_state_vector,
)

MISS_FAMILY_QDOT = "qdot"
MISS_FAMILY_SPECIES_DIAG = "species_diag"


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _first_cluster(general: dict[str, Any], *, kind: str) -> dict[str, Any]:
    for cluster in list(general.get("focus_clusters", []) or []):
        if str(cluster.get("kind", "")) == kind:
            return dict(cluster)
    return {}


def _resolve_strategy_entry_path(
    *,
    strategy_config_path: str | Path,
    candidate: str,
) -> Path:
    raw = Path(str(candidate).strip())
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (Path(strategy_config_path).resolve().parent / raw).resolve()


def _resolve_config_artifact_path(
    *,
    config_path: str | Path,
    candidate: str,
) -> Path:
    raw = Path(str(candidate).strip())
    if raw.is_absolute():
        return raw.resolve()
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (Path(config_path).resolve().parent / raw).resolve()


def _infer_failure_family(
    *,
    cluster: dict[str, Any],
    authority_miss_payload: dict[str, Any],
    tracked_state_species: set[str] | None = None,
) -> str:
    tracked_species = set(tracked_state_species or set())
    failure_class = str(
        cluster.get("failure_class") or authority_miss_payload.get("failure_class") or ""
    ).strip()
    top_variable = str(
        cluster.get("top_variable")
        or authority_miss_payload.get("reject_variable")
        or authority_miss_payload.get("qdot_reject_variable")
        or ""
    ).strip()
    if top_variable == "Qdot" or failure_class == "qdot":
        return MISS_FAMILY_QDOT
    if (
        top_variable.endswith("_diag")
        or top_variable in tracked_species
        or failure_class == "same_sign_overshoot"
    ):
        return MISS_FAMILY_SPECIES_DIAG
    max_out_of_bound = dict(authority_miss_payload.get("max_out_of_bound_by_variable", {}) or {})
    if any(str(name).endswith("_diag") or str(name) in tracked_species for name in max_out_of_bound):
        return MISS_FAMILY_SPECIES_DIAG
    first_variables = [
        str(value) for value in list(authority_miss_payload.get("first_offending_variables", []) or [])
    ]
    if any(value.endswith("_diag") or value in tracked_species for value in first_variables):
        return MISS_FAMILY_SPECIES_DIAG
    raise ValueError(
        "Unable to infer latest miss family from analyzer output and authority miss payload"
    )


def _storage_path(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _latest_coverage_corpus_path(latest_artifacts: dict[str, Any]) -> Path | None:
    artifact_paths = dict(latest_artifacts.get("artifact_paths", {}) or {})
    candidates = [
        artifact_paths.get("coverage_json_path"),
        artifact_paths.get("coverage_npz_path"),
    ]
    benchmark_run_dir = str(latest_artifacts.get("benchmark_run_dir", "") or "").strip()
    if benchmark_run_dir:
        benchmark_path = Path(benchmark_run_dir)
        candidates.extend(
            [
                str(benchmark_path / "runtimeChemistryCoverageCorpus.json"),
                str(benchmark_path / "runtimeChemistryCoverageCorpus.npz"),
            ]
        )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).resolve()
        if path.exists():
            return path
    return None


def _seedability_status(
    *,
    miss_family: str,
    authority_miss_payload: dict[str, Any],
    tracked_state_species: set[str] | None = None,
) -> tuple[bool, str]:
    tracked_species = set(tracked_state_species or set())
    reject_state = dict(authority_miss_payload.get("reject_state", {}) or {})
    if not reject_state:
        return False, "missing_reject_state"
    reject_variable = str(authority_miss_payload.get("reject_variable", "")).strip()
    failure_class = str(authority_miss_payload.get("failure_class", "")).strip()
    if miss_family == MISS_FAMILY_QDOT:
        if reject_variable != "Qdot":
            return False, "reject_variable_not_qdot"
        return True, "seedable_qdot_reject_state"
    if reject_variable.endswith("_diag"):
        return True, "seedable_species_diag_reject_state"
    if reject_variable in tracked_species:
        return True, "seedable_species_state_reject_state"
    if failure_class == "same_sign_overshoot" and reject_variable and reject_variable != "Qdot":
        return True, "seedable_species_output_reject_state"
    if tracked_species:
        return False, "reject_variable_not_tracked_species"
    return False, "reject_variable_not_species_diag"


def _rounded_transformed_signature(vector: Any) -> tuple[float, ...]:
    return tuple(round(float(value), 3) for value in list(vector))


def _candidate_run_authority_miss_paths(
    run_dirs: list[str | Path],
    *,
    profile_name: str | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for run_dir in run_dirs:
        artifacts = extract_restart_run_artifacts(run_dir=run_dir, profile_name=profile_name)
        authority_miss_path = str(artifacts.get("artifact_paths", {}).get("authority_miss_path", "")).strip()
        if authority_miss_path:
            paths.append(Path(authority_miss_path).resolve())
    return paths


def _load_stage_runtime_context(
    *,
    strategy_config_path: str | Path,
    stage_name: str,
) -> tuple[Path, dict[str, Any], set[str], list[str], list[str], dict[str, float]]:
    strategy = _load_json(strategy_config_path)
    runtime_entry = dict(strategy.get("runtime_package", {}) or {})
    stage_entries = dict(runtime_entry.get("stage_runtime_tables", {}) or {})
    stage_entry = dict(stage_entries.get(stage_name, {}) or {})
    if not stage_entry:
        raise ValueError(f"Stage '{stage_name}' is not defined in {strategy_config_path}")
    stage_config_path = _resolve_strategy_entry_path(
        strategy_config_path=strategy_config_path,
        candidate=str(stage_entry.get("runtime_table_config_path", "")).strip(),
    )
    stage_config = _load_json(stage_config_path)
    stage_runtime_cfg = dict(stage_config.get("runtime_chemistry_table", {}) or {})
    tracked_state_species = {
        str(item).strip()
        for item in list(stage_runtime_cfg.get("state_species", []) or [])
        if str(item).strip()
    }
    axis_order = ["Temperature", "Pressure", *list(stage_runtime_cfg.get("state_species", []) or [])]
    transformed_state_variables = _resolve_transformed_state_variables(
        stage_runtime_cfg,
        axis_order=axis_order,
    )
    state_transform_floors = _state_transform_floor_map(
        stage_runtime_cfg,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
    )
    return (
        stage_config_path,
        stage_runtime_cfg,
        tracked_state_species,
        axis_order,
        transformed_state_variables,
        state_transform_floors,
    )


def _load_frontier_candidate(
    *,
    path: Path,
    recency_rank: int,
    required_stage_name: str,
    tracked_state_species: set[str],
    axis_order: list[str],
    transformed_state_variables: list[str],
    state_transform_floors: dict[str, float],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    stage_name = str(payload.get("stage_name", "")).strip()
    if stage_name and stage_name != required_stage_name:
        return None
    miss_family = _infer_failure_family(
        cluster={},
        authority_miss_payload=payload,
        tracked_state_species=tracked_state_species,
    )
    seedable, seedability_reason = _seedability_status(
        miss_family=miss_family,
        authority_miss_payload=payload,
        tracked_state_species=tracked_state_species,
    )
    reject_state = dict(payload.get("reject_state", {}) or {})
    transformed_state = None
    transformed_signature = None
    if reject_state:
        transformed_state = _transform_state_vector(
            reject_state,
            axis_order=axis_order,
            transformed_state_variables=transformed_state_variables,
            state_transform_floors=state_transform_floors,
        )
        transformed_signature = _rounded_transformed_signature(transformed_state)
    return {
        "path": str(path.resolve()),
        "recency_rank": int(recency_rank),
        "stage_name": stage_name or required_stage_name,
        "reject_variable": str(payload.get("reject_variable", "")).strip(),
        "miss_family": miss_family,
        "seedable": bool(seedable),
        "seedability_reason": seedability_reason,
        "reject_state": reject_state,
        "transformed_state": transformed_state,
        "transformed_signature": transformed_signature,
    }


def _collect_frontier_candidates(
    *,
    stage_config_path: str | Path,
    existing_paths: list[str],
    run_dirs: list[str | Path],
    profile_name: str | None,
    required_stage_name: str,
    tracked_state_species: set[str],
    axis_order: list[str],
    transformed_state_variables: list[str],
    state_transform_floors: dict[str, float],
) -> list[dict[str, Any]]:
    deduped_paths: OrderedDict[str, Path] = OrderedDict()
    for item in existing_paths:
        if not str(item).strip():
            continue
        resolved = _resolve_config_artifact_path(config_path=stage_config_path, candidate=str(item))
        path_key = str(resolved)
        if path_key in deduped_paths:
            deduped_paths.pop(path_key)
        deduped_paths[path_key] = resolved
    for resolved in _candidate_run_authority_miss_paths(run_dirs, profile_name=profile_name):
        path_key = str(resolved)
        if path_key in deduped_paths:
            deduped_paths.pop(path_key)
        deduped_paths[path_key] = resolved
    raw_paths = list(deduped_paths.values())
    candidates: list[dict[str, Any]] = []
    for recency_rank, path in enumerate(raw_paths):
        candidate = _load_frontier_candidate(
            path=path,
            recency_rank=recency_rank,
            required_stage_name=required_stage_name,
            tracked_state_species=tracked_state_species,
            axis_order=axis_order,
            transformed_state_variables=transformed_state_variables,
            state_transform_floors=state_transform_floors,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _select_species_frontier(
    candidates: list[dict[str, Any]],
    *,
    target_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduped: OrderedDict[tuple[str, tuple[float, ...]], dict[str, Any]] = OrderedDict()
    pruned: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["miss_family"] != MISS_FAMILY_SPECIES_DIAG:
            continue
        if not candidate["seedable"]:
            pruned.append(
                {
                    "path": candidate["path"],
                    "reason": str(candidate["seedability_reason"]),
                }
            )
            continue
        signature = candidate["transformed_signature"]
        if signature is None:
            pruned.append({"path": candidate["path"], "reason": "missing_transformed_signature"})
            continue
        key = (str(candidate["reject_variable"]), signature)
        if key in deduped:
            older = deduped.pop(key)
            pruned.append({"path": older["path"], "reason": "superseded_duplicate_signature"})
        deduped[key] = candidate
    active = list(deduped.values())[-max(int(target_limit), 1) :]
    active_paths = {str(item["path"]) for item in active}
    for candidate in deduped.values():
        if str(candidate["path"]) not in active_paths:
            pruned.append({"path": candidate["path"], "reason": "older_than_active_species_frontier"})
    return active, pruned


def _select_qdot_frontier(
    candidates: list[dict[str, Any]],
    *,
    target_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduped: OrderedDict[tuple[float, ...], dict[str, Any]] = OrderedDict()
    pruned: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["miss_family"] != MISS_FAMILY_QDOT:
            continue
        if not candidate["seedable"]:
            pruned.append({"path": candidate["path"], "reason": str(candidate["seedability_reason"])})
            continue
        signature = candidate["transformed_signature"]
        if signature is None:
            pruned.append({"path": candidate["path"], "reason": "missing_transformed_signature"})
            continue
        if signature in deduped:
            older = deduped.pop(signature)
            pruned.append({"path": older["path"], "reason": "superseded_duplicate_signature"})
        deduped[signature] = candidate
    seedable = list(deduped.values())
    if not seedable:
        return [], pruned
    active: list[dict[str, Any]] = [seedable[-1]]
    remaining = list(seedable[:-1])
    frontier_limit = max(int(target_limit), 1)
    while remaining and len(active) < frontier_limit:
        best_index: int | None = None
        best_key: tuple[float, int] | None = None
        for index, candidate in enumerate(remaining):
            candidate_state = candidate.get("transformed_state")
            if candidate_state is None:
                continue
            min_distance = min(
                float((candidate_state - selected_state).dot(candidate_state - selected_state)) ** 0.5
                for selected_state in [
                    item.get("transformed_state") for item in active if item.get("transformed_state") is not None
                ]
            )
            candidate_key = (float(min_distance), int(candidate["recency_rank"]))
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_index = index
        if best_index is None:
            break
        active.append(remaining.pop(best_index))
    active.sort(key=lambda item: int(item["recency_rank"]))
    active_paths = {str(item["path"]) for item in active}
    for candidate in seedable:
        if str(candidate["path"]) not in active_paths:
            pruned.append({"path": candidate["path"], "reason": "not_selected_for_active_qdot_frontier"})
    return active, pruned


def plan_stage_local_runtime_table_refresh(
    *,
    run_dirs: list[str | Path],
    strategy_config_path: str | Path,
    profile_name: str | None = None,
    history_window: int = 5,
    analysis_payload: dict[str, Any] | None = None,
    analysis_general_path: str | Path | None = None,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("run_dirs must contain at least one analyzed run")
    if analysis_payload is None:
        if analysis_general_path is not None:
            analysis_payload = {"general": _load_json(analysis_general_path)}
        else:
            analysis_payload = analyze_restart_regression_runs(
                run_dirs=[str(Path(path)) for path in run_dirs],
                profile_name=profile_name,
                history_window=history_window,
            )
    general = dict(analysis_payload.get("general", {}) or {})
    failure_cluster = _first_cluster(general, kind="failure_cluster")
    operational_cluster = _first_cluster(general, kind="operational_cluster")
    latest_run_dir = Path(run_dirs[-1]).resolve()
    latest_artifacts = extract_restart_run_artifacts(run_dir=latest_run_dir, profile_name=profile_name)
    authority_miss_path = latest_artifacts["artifact_paths"].get("authority_miss_path")
    if not authority_miss_path:
        raise FileNotFoundError(
            f"No runtimeChemistryAuthorityMiss artifact found in latest run {latest_run_dir}"
        )
    authority_miss_payload = dict(latest_artifacts.get("authority_miss_payload", {}) or {})
    stage_name = str(
        failure_cluster.get("stage_name")
        or authority_miss_payload.get("stage_name")
        or ""
    ).strip()
    if not stage_name:
        raise ValueError("Unable to infer target stage from latest analyzer output")
    (
        stage_config_path,
        stage_runtime_cfg,
        tracked_state_species,
        _axis_order,
        _transformed_state_variables,
        _state_transform_floors,
    ) = _load_stage_runtime_context(
        strategy_config_path=strategy_config_path,
        stage_name=stage_name,
    )
    miss_family = _infer_failure_family(
        cluster=failure_cluster,
        authority_miss_payload=authority_miss_payload,
        tracked_state_species=tracked_state_species,
    )
    seedable, seedability_reason = _seedability_status(
        miss_family=miss_family,
        authority_miss_payload=authority_miss_payload,
        tracked_state_species=tracked_state_species,
    )
    coverage_corpus_path = _latest_coverage_corpus_path(latest_artifacts)
    return {
        "latest_run_dir": str(latest_run_dir),
        "latest_run_id": str(latest_artifacts.get("run_id", latest_run_dir.name)),
        "profile_name": str(latest_artifacts.get("profile_name", "") or ""),
        "stage_name": stage_name,
        "stage_config_path": str(stage_config_path),
        "failure_cluster": failure_cluster,
        "operational_cluster": operational_cluster,
        "miss_family": miss_family,
        "seedable": bool(seedable),
        "seedability_reason": seedability_reason,
        "authority_miss_path": str(Path(authority_miss_path).resolve()),
        "coverage_corpus_path": None if coverage_corpus_path is None else str(coverage_corpus_path),
        "coverage_corpus_exists": coverage_corpus_path is not None,
        "target_list_name": (
            "seed_qdot_miss_artifacts"
            if miss_family == MISS_FAMILY_QDOT
            else "seed_species_miss_artifacts"
        ),
    }


def plan_stage_local_runtime_table_frontier_rebalance(
    *,
    run_dirs: list[str | Path],
    strategy_config_path: str | Path,
    profile_name: str | None = None,
    history_window: int = 5,
    analysis_payload: dict[str, Any] | None = None,
    analysis_general_path: str | Path | None = None,
) -> dict[str, Any]:
    base_plan = plan_stage_local_runtime_table_refresh(
        run_dirs=run_dirs,
        strategy_config_path=strategy_config_path,
        profile_name=profile_name,
        history_window=history_window,
        analysis_payload=analysis_payload,
        analysis_general_path=analysis_general_path,
    )
    (
        stage_config_path,
        stage_runtime_cfg,
        tracked_state_species,
        axis_order,
        transformed_state_variables,
        state_transform_floors,
    ) = _load_stage_runtime_context(
        strategy_config_path=strategy_config_path,
        stage_name=str(base_plan["stage_name"]),
    )
    candidates = _collect_frontier_candidates(
        stage_config_path=stage_config_path,
        existing_paths=[
            *list(stage_runtime_cfg.get("seed_species_miss_artifacts", []) or []),
            *list(stage_runtime_cfg.get("seed_qdot_miss_artifacts", []) or []),
        ],
        run_dirs=run_dirs,
        profile_name=profile_name,
        required_stage_name=str(base_plan["stage_name"]),
        tracked_state_species=tracked_state_species,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
    )
    species_active, species_pruned = _select_species_frontier(
        candidates,
        target_limit=max(int(stage_runtime_cfg.get("current_window_diag_target_limit", 4)), 1),
    )
    qdot_active, qdot_pruned = _select_qdot_frontier(
        candidates,
        target_limit=max(int(stage_runtime_cfg.get("current_window_qdot_target_limit", 2)), 1),
    )
    return {
        **base_plan,
        "selection_reason": "bounded_frontier_rebalance",
        "active_species_frontier": [str(item["path"]) for item in species_active],
        "active_qdot_frontier": [str(item["path"]) for item in qdot_active],
        "pruned_species_artifacts": species_pruned,
        "pruned_qdot_artifacts": qdot_pruned,
    }


def apply_stage_local_runtime_table_refresh(
    *,
    refresh_plan: dict[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root_path = Path(repo_root or Path.cwd()).resolve()
    config_path = Path(str(refresh_plan["stage_config_path"])).resolve()
    if not bool(refresh_plan.get("seedable", False)):
        return {
            "stage_name": str(refresh_plan["stage_name"]),
            "stage_config_path": str(config_path),
            "target_list_name": str(refresh_plan["target_list_name"]),
            "seedable": False,
            "seedability_reason": str(refresh_plan.get("seedability_reason", "not_seedable")),
            "skipped": True,
            "appended_miss_artifact": None,
            "appended_coverage_corpus": None,
        }
    payload = _load_json(config_path)
    runtime_cfg = dict(payload.get("runtime_chemistry_table", {}) or {})
    target_list_name = str(refresh_plan["target_list_name"])
    miss_storage_path = _storage_path(Path(str(refresh_plan["authority_miss_path"])), repo_root=repo_root_path)
    miss_items = [str(item) for item in list(runtime_cfg.get(target_list_name, []) or [])]
    if miss_storage_path not in miss_items:
        miss_items.append(miss_storage_path)
    runtime_cfg[target_list_name] = miss_items

    appended_coverage = None
    coverage_path_raw = refresh_plan.get("coverage_corpus_path")
    if coverage_path_raw:
        coverage_storage_path = _storage_path(Path(str(coverage_path_raw)), repo_root=repo_root_path)
        coverage_items = [str(item) for item in list(runtime_cfg.get("coverage_corpora", []) or [])]
        if coverage_storage_path not in coverage_items:
            coverage_items.append(coverage_storage_path)
            appended_coverage = coverage_storage_path
        runtime_cfg["coverage_corpora"] = coverage_items

    payload["runtime_chemistry_table"] = runtime_cfg
    _write_json(config_path, payload)
    return {
        "stage_name": str(refresh_plan["stage_name"]),
        "stage_config_path": str(config_path),
        "target_list_name": target_list_name,
        "seedable": True,
        "seedability_reason": str(refresh_plan.get("seedability_reason", "")),
        "skipped": False,
        "appended_miss_artifact": miss_storage_path,
        "appended_coverage_corpus": appended_coverage,
    }


def apply_stage_local_runtime_table_frontier_rebalance(
    *,
    rebalance_plan: dict[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root_path = Path(repo_root or Path.cwd()).resolve()
    config_path = Path(str(rebalance_plan["stage_config_path"])).resolve()
    payload = _load_json(config_path)
    runtime_cfg = dict(payload.get("runtime_chemistry_table", {}) or {})
    active_species = [
        _storage_path(Path(str(path)), repo_root=repo_root_path)
        for path in list(rebalance_plan.get("active_species_frontier", []) or [])
    ]
    active_qdot = [
        _storage_path(Path(str(path)), repo_root=repo_root_path)
        for path in list(rebalance_plan.get("active_qdot_frontier", []) or [])
    ]
    runtime_cfg["seed_species_miss_artifacts"] = active_species
    runtime_cfg["seed_qdot_miss_artifacts"] = active_qdot
    payload["runtime_chemistry_table"] = runtime_cfg
    _write_json(config_path, payload)
    return {
        "stage_name": str(rebalance_plan["stage_name"]),
        "stage_config_path": str(config_path),
        "selection_reason": str(rebalance_plan.get("selection_reason", "bounded_frontier_rebalance")),
        "active_species_frontier": active_species,
        "active_qdot_frontier": active_qdot,
        "pruned_species_artifacts": list(rebalance_plan.get("pruned_species_artifacts", []) or []),
        "pruned_qdot_artifacts": list(rebalance_plan.get("pruned_qdot_artifacts", []) or []),
    }


__all__ = [
    "MISS_FAMILY_QDOT",
    "MISS_FAMILY_SPECIES_DIAG",
    "apply_stage_local_runtime_table_frontier_rebalance",
    "apply_stage_local_runtime_table_refresh",
    "plan_stage_local_runtime_table_frontier_rebalance",
    "plan_stage_local_runtime_table_refresh",
]
