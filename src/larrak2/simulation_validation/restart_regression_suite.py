"""Artifact-only restart regression analysis suites."""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Any

from .restart_regression_artifacts import extract_restart_run_artifacts
from .restart_regression_numeric import (
    SLOT_FAMILY_CONTEXT,
    SLOT_FAMILY_FAILURE,
    SLOT_FAMILY_OPERATIONAL,
    analyze_dense_series,
    analyze_scalar_slots,
)


def _resolve_run_dirs(
    *,
    runs: list[str] | None = None,
    glob_pattern: str = "",
    latest: int | None = None,
) -> list[Path]:
    explicit = [Path(item).resolve() for item in list(runs or []) if str(item).strip()]
    if explicit and glob_pattern:
        raise ValueError("Use either --runs or --glob/--latest, not both")
    if explicit:
        return explicit
    if not glob_pattern:
        raise ValueError("Either --runs or --glob must be provided")
    matched = sorted(Path(path).resolve() for path in glob.glob(glob_pattern))
    if latest is not None:
        matched = matched[-max(int(latest), 0) :]
    resolved = [path.resolve() for path in matched]
    if not resolved:
        raise FileNotFoundError(f"No restart benchmark runs matched glob '{glob_pattern}'")
    return resolved


def _metric_value(run: dict[str, Any], name: str) -> float | None:
    profile = dict(run.get("selected_profile", {}) or {})
    runtime = dict(profile.get("runtime_chemistry", {}) or {})
    if name in runtime:
        value = runtime.get(name)
    else:
        value = profile.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_variable(run: dict[str, Any]) -> str:
    profile = dict(run.get("selected_profile", {}) or {})
    values = list(profile.get("first_offending_variables", []) or [])
    if values:
        return str(values[0])
    miss = dict(run.get("authority_miss_payload", {}) or {})
    values = list(miss.get("first_offending_variables", []) or [])
    if values:
        return str(values[0])
    reject_variable = str(miss.get("reject_variable", "")).strip()
    return reject_variable


def _first_miss_stage(run: dict[str, Any]) -> str:
    miss = dict(run.get("authority_miss_payload", {}) or {})
    return str(miss.get("stage_name", "")).strip()


def _failure_class(run: dict[str, Any]) -> str:
    miss = dict(run.get("authority_miss_payload", {}) or {})
    value = str(miss.get("failure_class", "")).strip()
    if value:
        return value
    profile = dict(run.get("selected_profile", {}) or {})
    return str(profile.get("first_miss_class", "")).strip()


def _has_checkpoint_progress(run: dict[str, Any]) -> bool:
    profile = dict(run.get("selected_profile", {}) or {})
    latest = dict(profile.get("latest_checkpoint", {}) or {})
    return bool(latest)


def _runtime_summary_dense_rows(dense_output: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in list(dense_output.get("records", []) or [])
        if str(item.get("artifact_slot_id", "")) == "runtimeChemistrySummary.dat"
    ]


def _top_runtime_dense_row(dense_output: dict[str, Any]) -> dict[str, Any] | None:
    rows = [
        item
        for item in list(dense_output.get("top_divergence", []) or [])
        if str(item.get("artifact_slot_id", "")) == "runtimeChemistrySummary.dat"
    ]
    return None if not rows else dict(rows[0])


def _same_area_classification(
    *,
    latest_run: dict[str, Any],
    previous_run: dict[str, Any],
    dense_output: dict[str, Any],
) -> tuple[str, float, dict[str, Any]]:
    latest_profile = dict(latest_run.get("selected_profile", {}) or {})
    previous_profile = dict(previous_run.get("selected_profile", {}) or {})

    point_score = 0.0
    if _first_miss_stage(latest_run) and _first_miss_stage(latest_run) == _first_miss_stage(
        previous_run
    ):
        point_score += 0.25
    if _failure_class(latest_run) and _failure_class(latest_run) == _failure_class(previous_run):
        point_score += 0.25
    latest_branch = str(latest_profile.get("first_miss_branch_id", "")).strip()
    previous_branch = str(previous_profile.get("first_miss_branch_id", "")).strip()
    if latest_branch and latest_branch == previous_branch:
        point_score += 0.15
    latest_variable = _top_variable(latest_run)
    previous_variable = _top_variable(previous_run)
    if latest_variable and previous_variable and latest_variable == previous_variable:
        point_score += 0.35

    runtime_summary_rows = _runtime_summary_dense_rows(dense_output)
    dense_distance = (
        None
        if not runtime_summary_rows
        else sum(float(item.get("priority_score") or 0.0) for item in runtime_summary_rows)
        / len(runtime_summary_rows)
    )
    early_collapse = any(
        bool(item.get("early_collapse_detected", False)) for item in runtime_summary_rows
    )
    first_divergence_early = any(
        item.get("first_divergence_fraction") is not None
        and float(item["first_divergence_fraction"]) <= 0.25
        for item in runtime_summary_rows
    )
    if point_score >= 0.65 and not early_collapse and (dense_distance is None or dense_distance <= 4.0):
        classification = "same_area"
    elif point_score <= 0.35 or early_collapse or first_divergence_early or (
        dense_distance is not None and dense_distance >= 5.0
    ):
        classification = "shifted_area"
    else:
        classification = "undetermined"
    confidence = min(
        1.0,
        point_score
        + (0.2 if dense_distance is not None else 0.0)
        + (0.15 if _first_miss_stage(latest_run) else 0.0),
    )
    details = {
        "point_similarity_score": point_score,
        "dense_distance_score": dense_distance,
        "early_divergence_detected": first_divergence_early,
        "early_collapse_detected": early_collapse,
        "latest_top_variable": latest_variable,
        "previous_top_variable": previous_variable,
        "latest_stage_name": _first_miss_stage(latest_run),
        "previous_stage_name": _first_miss_stage(previous_run),
        "latest_failure_class": _failure_class(latest_run),
        "previous_failure_class": _failure_class(previous_run),
    }
    return classification, float(confidence), details


def _primary_regression_drivers(
    *,
    latest_run: dict[str, Any],
    previous_run: dict[str, Any],
    dense_output: dict[str, Any],
) -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    latest_angle = _metric_value(latest_run, "angle_advance_deg")
    previous_angle = _metric_value(previous_run, "angle_advance_deg")
    if previous_angle is not None and (
        latest_angle is None or float(latest_angle) < float(previous_angle) - 1.0e-6
    ):
        drivers.append(
            {
                "name": "angle_advance_loss",
                "description": "Continuation depth regressed or disappeared.",
                "latest_value": latest_angle,
                "baseline_value": previous_angle,
            }
        )

    if _has_checkpoint_progress(previous_run) and not _has_checkpoint_progress(latest_run):
        drivers.append(
            {
                "name": "checkpoint_loss",
                "description": "Latest run no longer produced checkpoint progress.",
                "latest_value": 0.0,
                "baseline_value": 1.0,
            }
        )

    latest_hit_fraction = _metric_value(latest_run, "table_hit_fraction")
    previous_hit_fraction = _metric_value(previous_run, "table_hit_fraction")
    if (
        latest_hit_fraction is not None
        and previous_hit_fraction not in (None, 0.0)
        and float(latest_hit_fraction) / float(previous_hit_fraction) <= 0.5
    ):
        drivers.append(
            {
                "name": "table_hit_collapse",
                "description": "Table-hit fraction collapsed materially.",
                "latest_value": latest_hit_fraction,
                "baseline_value": previous_hit_fraction,
            }
        )

    top_dense = _top_runtime_dense_row(dense_output)
    if top_dense is not None and (
        bool(top_dense.get("early_collapse_detected", False))
        or str(top_dense.get("collapse_classification", "")) in {"early_collapse", "truncated"}
    ):
        drivers.append(
            {
                "name": "early_dense_collapse",
                "description": "Dense runtime summary indicates early counter collapse.",
                "latest_value": top_dense.get("semantic_label"),
                "baseline_value": top_dense.get("first_divergence_fraction"),
            }
        )

    latest_signature = (_first_miss_stage(latest_run), _top_variable(latest_run), _failure_class(latest_run))
    previous_signature = (
        _first_miss_stage(previous_run),
        _top_variable(previous_run),
        _failure_class(previous_run),
    )
    if latest_signature != previous_signature:
        drivers.append(
            {
                "name": "failure_signature_shift",
                "description": "First-failure family shifted between runs.",
                "latest_value": latest_signature,
                "baseline_value": previous_signature,
            }
        )
    return drivers


def _overall_verdict(
    *,
    latest_run: dict[str, Any],
    previous_run: dict[str, Any],
    dense_output: dict[str, Any],
) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    higher_better = [
        ("angle_advance_deg", 2.5),
        ("table_hit_fraction", 2.0),
    ]
    lower_better = [
        ("trust_region_reject_cells", 1.25),
        ("coverage_reject_cells", 1.25),
        ("fallback_timesteps", 1.25),
        ("total_numeric_hits", 1.0),
    ]
    primary_regressions = _primary_regression_drivers(
        latest_run=latest_run,
        previous_run=previous_run,
        dense_output=dense_output,
    )
    signals: list[dict[str, Any]] = []
    suppressed_secondary_improvements: list[dict[str, Any]] = []
    improvement = 0.0
    regression = float(len(primary_regressions) * 2.0)
    rationale = [item["description"] for item in primary_regressions]

    for metric_name, weight in higher_better:
        latest_value = _metric_value(latest_run, metric_name)
        previous_value = _metric_value(previous_run, metric_name)
        delta = (
            None
            if latest_value is None or previous_value is None
            else latest_value - previous_value
        )
        direction = "flat"
        if latest_value is None and previous_value is not None:
            direction = "regressed"
            regression += weight
        elif latest_value is not None and previous_value is None:
            direction = "improved"
            improvement += weight
        elif delta is not None:
            tolerance = 1.0e-9 if metric_name != "angle_advance_deg" else 1.0e-6
            if delta > tolerance:
                direction = "improved"
                improvement += weight
            elif delta < -tolerance:
                direction = "regressed"
                regression += weight
        signals.append(
            {
                "metric": metric_name,
                "directionality": "higher_better",
                "latest_value": latest_value,
                "baseline_value": previous_value,
                "delta": delta,
                "weight": weight,
                "signal": direction,
            }
        )

    for metric_name, weight in lower_better:
        latest_value = _metric_value(latest_run, metric_name)
        previous_value = _metric_value(previous_run, metric_name)
        delta = (
            None
            if latest_value is None or previous_value is None
            else latest_value - previous_value
        )
        direction = "flat"
        if latest_value is None and previous_value is not None:
            direction = "regressed"
            regression += weight
        elif latest_value is not None and previous_value is None:
            direction = "improved"
            improvement += weight
        elif delta is not None:
            tolerance = 1.0e-9
            if delta < -tolerance:
                direction = "improved"
                improvement += weight
            elif delta > tolerance:
                direction = "regressed"
                regression += weight

        if (
            direction == "improved"
            and primary_regressions
            and metric_name == "total_numeric_hits"
        ):
            direction = "suppressed_improvement"
            suppressed_secondary_improvements.append(
                {
                    "metric": metric_name,
                    "latest_value": latest_value,
                    "baseline_value": previous_value,
                    "reason": "Primary continuation/regression signals dominate lower numeric hits.",
                }
            )
            improvement -= weight
        signals.append(
            {
                "metric": metric_name,
                "directionality": "lower_better",
                "latest_value": latest_value,
                "baseline_value": previous_value,
                "delta": delta,
                "weight": weight,
                "signal": direction,
            }
        )

    latest_profile = dict(latest_run.get("selected_profile", {}) or {})
    previous_profile = dict(previous_run.get("selected_profile", {}) or {})
    latest_solver = bool(latest_profile.get("solver_ok", False))
    previous_solver = bool(previous_profile.get("solver_ok", False))
    if latest_solver and not previous_solver:
        improvement += 1.0
    if previous_solver and not latest_solver:
        regression += 1.5
        rationale.append("Solver state regressed.")

    if primary_regressions:
        classification = "regressed"
    elif improvement == 0.0 and regression == 0.0:
        classification = "insufficient_data"
    elif improvement >= regression + 1.5:
        classification = "improved"
    elif regression >= improvement + 1.5:
        classification = "regressed"
    else:
        classification = "mixed"
    confidence = min(1.0, 0.35 + 0.1 * len(primary_regressions) + (improvement + regression) / 12.0)
    return (
        classification,
        float(confidence),
        signals,
        primary_regressions,
        suppressed_secondary_improvements,
        rationale,
    )


def _cluster_id(*, stage_name: str, top_variable: str, failure_class: str, family: str) -> str:
    return f"{family}:{stage_name or 'unknown'}:{top_variable or 'unknown'}:{failure_class or 'unknown'}"


def _recommended_next_analysis_target(*, stage_name: str, top_variable: str) -> str:
    if stage_name and top_variable:
        return (
            f"Inspect the {stage_name} neighborhood around {top_variable} and the earliest runtime-summary collapse window."
        )
    if top_variable:
        return f"Inspect the authority envelope and local dense-window behavior around {top_variable}."
    return "Inspect the earliest dense divergence window and the dominant failure-family scalars together."


def _focus_clusters(
    *,
    latest_run: dict[str, Any],
    scalar_output: dict[str, Any],
    dense_output: dict[str, Any],
) -> list[dict[str, Any]]:
    latest_miss = dict(latest_run.get("authority_miss_payload", {}) or {})
    stage_name = str(latest_miss.get("stage_name") or _first_miss_stage(latest_run) or "").strip()
    top_variable = _top_variable(latest_run)
    failure_class = _failure_class(latest_run) or "authority_break"
    primary_cluster_id = _cluster_id(
        stage_name=stage_name,
        top_variable=top_variable,
        failure_class=failure_class,
        family="failure",
    )

    scalar_candidates = [
        item
        for item in list(scalar_output.get("top_leverage", []) or [])
        if not bool(item.get("excluded_from_focus", False))
        and str(item.get("slot_family", "")) in {SLOT_FAMILY_FAILURE, SLOT_FAMILY_OPERATIONAL}
    ]
    dense_candidates = [
        item
        for item in list(dense_output.get("top_divergence", []) or [])
        if not bool(item.get("excluded_from_focus", False))
    ]

    related_scalar = [
        item
        for item in scalar_candidates
        if top_variable
        and top_variable.lower() in str(item.get("semantic_label", "")).lower()
        or str(item.get("artifact_slot_id", "")) == "runtimeChemistryAuthorityMiss"
        or "first_miss" in str(item.get("semantic_label", ""))
    ]
    related_dense = [
        item
        for item in dense_candidates
        if str(item.get("artifact_slot_id", "")) == "runtimeChemistrySummary.dat"
    ][:3]
    primary_priority = float(
        max(
            [1.0]
            + [float(item.get("priority_score") or 0.0) for item in related_scalar[:3]]
            + [float(item.get("priority_score") or 0.0) for item in related_dense[:2]]
        )
    )
    cluster_label = (
        f"failure cluster: {stage_name or 'unknown_stage'} / {top_variable or 'unknown_variable'} authority break"
    )
    primary_cluster = {
        "cluster_id": primary_cluster_id,
        "kind": "failure_cluster",
        "label": cluster_label,
        "priority_score": primary_priority,
        "stage_name": stage_name,
        "top_variable": top_variable,
        "failure_class": failure_class,
        "dominant_operational_consequence": (
            "No continued checkpoint advance."
            if not _has_checkpoint_progress(latest_run)
            else "Continuation depth materially weakened."
        ),
        "recommended_next_analysis_target": _recommended_next_analysis_target(
            stage_name=stage_name,
            top_variable=top_variable,
        ),
        "members": [
            *[
                {
                    "member_kind": "scalar",
                    "slot_or_series_id": item.get("slot_or_series_id"),
                    "label": item.get("semantic_label"),
                    "priority_score": item.get("priority_score"),
                }
                for item in related_scalar[:4]
            ],
            *[
                {
                    "member_kind": "dense",
                    "slot_or_series_id": item.get("slot_or_series_id"),
                    "label": item.get("semantic_label"),
                    "priority_score": item.get("priority_score"),
                }
                for item in related_dense[:2]
            ],
        ],
        "supporting_evidence": {
            "max_out_of_bound_by_variable": latest_miss.get("max_out_of_bound_by_variable", {}),
            "stage_name": stage_name,
            "failure_class": failure_class,
        },
    }

    clusters = [primary_cluster]
    top_dense = _top_runtime_dense_row(dense_output)
    if top_dense is not None:
        secondary_cluster_id = _cluster_id(
            stage_name=stage_name,
            top_variable=str(top_dense.get("semantic_label", "")).strip(),
            failure_class=str(top_dense.get("collapse_classification", "")).strip(),
            family="operational",
        )
        clusters.append(
            {
                "cluster_id": secondary_cluster_id,
                "kind": "operational_cluster",
                "label": "operational cluster: early runtime-summary collapse",
                "priority_score": float(top_dense.get("priority_score") or 0.0),
                "stage_name": stage_name,
                "top_variable": str(top_dense.get("semantic_label", "")).strip(),
                "failure_class": str(top_dense.get("collapse_classification", "")).strip(),
                "dominant_operational_consequence": (
                    "Runtime-summary counters diverged before meaningful stage progression."
                ),
                "recommended_next_analysis_target": (
                    "Inspect the earliest runtime-summary divergence window and the paired failure-family scalars."
                ),
                "members": [
                    {
                        "member_kind": "dense",
                        "slot_or_series_id": top_dense.get("slot_or_series_id"),
                        "label": top_dense.get("semantic_label"),
                        "priority_score": top_dense.get("priority_score"),
                    }
                ],
                "supporting_evidence": {
                    "dominant_divergence_windows": top_dense.get("dominant_divergence_windows", []),
                    "collapse_classification": top_dense.get("collapse_classification"),
                    "first_divergence_fraction": top_dense.get("first_divergence_fraction"),
                },
            }
        )

    return sorted(clusters, key=lambda item: float(item.get("priority_score") or 0.0), reverse=True)


def _prioritized_focus(
    *,
    focus_clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    focus: list[dict[str, Any]] = []
    for rank, cluster in enumerate(focus_clusters[:3], start=1):
        focus.append(
            {
                "kind": cluster.get("kind"),
                "cluster_id": cluster.get("cluster_id"),
                "priority": rank,
                "label": cluster.get("label"),
                "details": {
                    "priority_score": cluster.get("priority_score"),
                    "stage_name": cluster.get("stage_name"),
                    "top_variable": cluster.get("top_variable"),
                    "failure_class": cluster.get("failure_class"),
                    "dominant_operational_consequence": cluster.get("dominant_operational_consequence"),
                    "recommended_next_analysis_target": cluster.get("recommended_next_analysis_target"),
                },
            }
        )
    return focus


def analyze_restart_regression_runs(
    *,
    run_dirs: list[str] | None = None,
    glob_pattern: str = "",
    latest: int | None = None,
    profile_name: str | None = None,
    history_window: int = 5,
) -> dict[str, Any]:
    resolved_run_dirs = _resolve_run_dirs(runs=run_dirs, glob_pattern=glob_pattern, latest=latest)
    extracted = [
        extract_restart_run_artifacts(run_dir=path, profile_name=profile_name)
        for path in resolved_run_dirs
    ]
    scalar_output = analyze_scalar_slots(run_items=extracted, history_window=history_window)
    dense_output = analyze_dense_series(run_items=extracted, history_window=history_window)

    general_output: dict[str, Any]
    if len(extracted) < 2:
        general_output = {
            "latest_run_id": extracted[-1]["run_id"] if extracted else "",
            "baseline_run_id": "",
            "latest_run_classification": "insufficient_data",
            "neighborhood_classification": "undetermined",
            "confidence": 0.0,
            "operational_signals": [],
            "dominant_operational_regressions": [],
            "suppressed_secondary_improvements": [],
            "verdict_rationale": [],
            "neighborhood_details": {},
            "focus_clusters": [],
            "prioritized_focus": [],
            "ignored_context_slots_count": 0,
        }
    else:
        latest_run = extracted[-1]
        previous_run = extracted[-2]
        (
            verdict,
            verdict_confidence,
            signals,
            dominant_regressions,
            suppressed_improvements,
            rationale,
        ) = _overall_verdict(
            latest_run=latest_run,
            previous_run=previous_run,
            dense_output=dense_output,
        )
        neighborhood, neighborhood_confidence, neighborhood_details = _same_area_classification(
            latest_run=latest_run,
            previous_run=previous_run,
            dense_output=dense_output,
        )
        focus_clusters = _focus_clusters(
            latest_run=latest_run,
            scalar_output=scalar_output,
            dense_output=dense_output,
        )
        ignored_context_slots_count = sum(
            1
            for item in list(scalar_output.get("records", []) or [])
            if str(item.get("slot_family", "")) == SLOT_FAMILY_CONTEXT
            and bool(item.get("excluded_from_focus", False))
        )
        general_output = {
            "latest_run_id": latest_run["run_id"],
            "baseline_run_id": previous_run["run_id"],
            "latest_run_classification": verdict,
            "neighborhood_classification": neighborhood,
            "confidence": float(min((verdict_confidence + neighborhood_confidence) / 2.0, 1.0)),
            "operational_signals": signals,
            "dominant_operational_regressions": dominant_regressions,
            "suppressed_secondary_improvements": suppressed_improvements,
            "verdict_rationale": rationale,
            "neighborhood_details": neighborhood_details,
            "focus_clusters": focus_clusters,
            "prioritized_focus": _prioritized_focus(focus_clusters=focus_clusters),
            "ignored_context_slots_count": ignored_context_slots_count,
        }

    slot_manifest = {
        "run_ids": [str(item["run_id"]) for item in extracted],
        "profile_names": [str(item["profile_name"]) for item in extracted],
        "scalar_slots": scalar_output.get("slot_manifest", []),
        "dense_series": dense_output.get("series_manifest", []),
    }
    return {
        "run_ids": [str(item["run_id"]) for item in extracted],
        "resolved_run_dirs": [str(path) for path in resolved_run_dirs],
        "history_window": int(history_window),
        "general": general_output,
        "scalars": scalar_output,
        "dense": dense_output,
        "slot_manifest": slot_manifest,
    }


def _write_scalar_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "baseline_run_id",
        "slot_or_series_id",
        "artifact_slot_id",
        "position_index",
        "semantic_label",
        "slot_family",
        "latest_value",
        "baseline_value",
        "delta",
        "normalized_drift_score",
        "priority_score",
        "trend_slope",
        "regression_signal",
        "improvement_signal",
        "appearance_score",
        "disappearance_score",
        "breakpoint_detected",
        "scale_used",
        "scale_floor_applied",
        "focus_eligibility_reason",
        "confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def _render_general_markdown(payload: dict[str, Any]) -> str:
    general = dict(payload.get("general", {}) or {})
    lines = [
        "# Restart Regression Analysis",
        "",
        f"**Latest Run:** {general.get('latest_run_id', '')}",
        f"**Baseline Run:** {general.get('baseline_run_id', '')}",
        f"**Verdict:** {general.get('latest_run_classification', 'insufficient_data')}",
        f"**Neighborhood:** {general.get('neighborhood_classification', 'undetermined')}",
        f"**Confidence:** {float(general.get('confidence', 0.0)):.3f}",
        "",
        "## Why",
        "",
    ]
    for item in list(general.get("verdict_rationale", []) or []):
        lines.append(f"- {item}")
    if not list(general.get("verdict_rationale", []) or []):
        lines.append("- No dominant rationale available.")

    lines.extend(
        [
            "",
            "## Dominant Operational Regressions",
            "",
        ]
    )
    for item in list(general.get("dominant_operational_regressions", []) or []):
        lines.append(
            f"- `{item.get('name', '')}`: {item.get('description', '')} "
            f"(latest={item.get('latest_value')}, baseline={item.get('baseline_value')})"
        )
    if not list(general.get("dominant_operational_regressions", []) or []):
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Suppressed Secondary Improvements",
            "",
        ]
    )
    for item in list(general.get("suppressed_secondary_improvements", []) or []):
        lines.append(
            f"- `{item.get('metric', '')}` was not treated as improvement: {item.get('reason', '')}"
        )
    if not list(general.get("suppressed_secondary_improvements", []) or []):
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Focus Clusters",
            "",
        ]
    )
    for item in list(general.get("focus_clusters", []) or []):
        lines.append(f"- **{item.get('label', '')}**")
        lines.append(f"  Next target: {item.get('recommended_next_analysis_target', '')}")
    if not list(general.get("focus_clusters", []) or []):
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Operational Signals",
            "",
            "| Metric | Latest | Baseline | Delta | Signal |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in list(general.get("operational_signals", []) or []):
        lines.append(
            f"| {item['metric']} | {item.get('latest_value')} | "
            f"{item.get('baseline_value')} | {item.get('delta')} | {item.get('signal')} |"
        )

    lines.extend(
        [
            "",
            "## Report Notes",
            "",
            f"- Context slots ignored for focus ranking: {int(general.get('ignored_context_slots_count', 0))}",
            f"- Focus clusters generated: {len(list(general.get('focus_clusters', []) or []))}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def write_restart_regression_artifacts(
    *,
    analysis: dict[str, Any],
    outdir: str | Path,
    suite: str = "all",
) -> dict[str, str]:
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    general_json_path = output_dir / "restart_regression_general.json"
    general_md_path = output_dir / "restart_regression_general.md"
    scalars_json_path = output_dir / "restart_regression_scalars.json"
    scalars_csv_path = output_dir / "restart_regression_scalars.csv"
    dense_json_path = output_dir / "restart_regression_dense.json"
    slot_manifest_path = output_dir / "restart_regression_slot_manifest.json"

    effective_suite = str(suite).strip().lower()
    if effective_suite in {"general", "all"}:
        general_json_path.write_text(
            json.dumps(analysis["general"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        general_md_path.write_text(_render_general_markdown(analysis), encoding="utf-8")
        written["general_json"] = str(general_json_path)
        written["general_md"] = str(general_md_path)

    if effective_suite in {"scalars", "all", "general"}:
        scalars_json_path.write_text(
            json.dumps(analysis["scalars"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_scalar_csv(scalars_csv_path, list(analysis["scalars"].get("records", []) or []))
        written["scalars_json"] = str(scalars_json_path)
        written["scalars_csv"] = str(scalars_csv_path)

    if effective_suite in {"dense", "all", "general"}:
        dense_json_path.write_text(
            json.dumps(analysis["dense"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        written["dense_json"] = str(dense_json_path)

    slot_manifest_path.write_text(
        json.dumps(analysis["slot_manifest"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    written["slot_manifest_json"] = str(slot_manifest_path)
    return written


__all__ = ["analyze_restart_regression_runs", "write_restart_regression_artifacts"]
