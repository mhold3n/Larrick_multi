"""Combustion-truth workflow for the DOE core corridor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    ComparisonMode,
    SourceType,
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationMetricSpec,
    ValidationSuiteProfile,
)
from .suite import run_suite, suite_to_json, suite_to_markdown

DEFAULT_PROFILE_PATH = Path("data/training/f2_nn_overnight_core_edge_v1.json")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_dataset_and_case(
    config: dict[str, Any],
    regime: str,
) -> tuple[ValidationDatasetManifest, ValidationCaseSpec]:
    ds_cfg = dict(config.get("dataset", {}) or {})
    metrics = [
        ValidationMetricSpec(
            metric_id=str(metric["metric_id"]),
            units=str(metric.get("units", "")),
            comparison_mode=ComparisonMode(str(metric.get("comparison_mode", "absolute"))),
            tolerance_band=float(metric.get("tolerance_band", 0.0)),
            source_type=SourceType(str(metric.get("source_type", "measured"))),
            required=bool(metric.get("required", True)),
            description=str(metric.get("description", "")),
        )
        for metric in list(ds_cfg.get("metrics", []) or [])
    ]

    dataset = ValidationDatasetManifest(
        dataset_id=str(ds_cfg.get("dataset_id", f"{regime}_dataset")),
        regime=regime,
        fuel_family=str(ds_cfg.get("fuel_family", "gasoline")),
        source_type=SourceType(str(ds_cfg.get("source_type", "measured"))),
        provenance=dict(ds_cfg.get("provenance", {}) or {}),
        operating_bounds=dict(ds_cfg.get("operating_bounds", {}) or {}),
        metrics=metrics,
        measured_anchor_ids=list(ds_cfg.get("measured_anchor_ids", []) or []),
        governing_basis=str(ds_cfg.get("governing_basis", "")),
        literature_reference=str(ds_cfg.get("literature_reference", "")),
        standard_reference=str(ds_cfg.get("standard_reference", "")),
    )

    cs_cfg = dict(config.get("case_spec", {}) or {})
    case_spec = ValidationCaseSpec(
        case_id=str(cs_cfg.get("case_id", f"{regime}_case")),
        regime=regime,
        operating_point=dict(cs_cfg.get("operating_point", {}) or {}),
        geometry_revision=str(cs_cfg.get("geometry_revision", "")),
        motion_profile_revision=str(cs_cfg.get("motion_profile_revision", "")),
        solver_config=dict(cs_cfg.get("solver_config", {}) or {}),
        dataset_binding=str(cs_cfg.get("dataset_binding", "")),
    )
    return dataset, case_spec


def _build_suite_profile(config: dict[str, Any]) -> ValidationSuiteProfile:
    regime_order = list(config.get("regime_order", []) or [])
    prerequisites = {
        str(regime): [str(dep) for dep in deps]
        for regime, deps in dict(config.get("prerequisites", {}) or {}).items()
    }
    return ValidationSuiteProfile(
        suite_id=str(config.get("suite_id", "canonical_v1")),
        regime_order=regime_order,
        prerequisites=prerequisites,
        description=str(config.get("description", "")),
    )


def _is_inside_core(point: dict[str, float], core: dict[str, Any]) -> bool:
    rpm = float(point["rpm"])
    torque = float(point["torque"])
    return float(core.get("rpm_min", rpm)) <= rpm <= float(core.get("rpm_max", rpm)) and float(
        core.get("torque_min", torque)
    ) <= torque <= float(core.get("torque_max", torque))


def load_core_operating_points(
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
) -> list[dict[str, float]]:
    """Return core-corridor operating points from the overnight campaign profile."""
    profile = _load_json(profile_path)
    openfoam_cfg = dict(profile.get("openfoam", {}) or {})
    core = dict(openfoam_cfg.get("core_corridor", {}) or {})
    truth_points = [
        {"rpm": float(point["rpm"]), "torque": float(point["torque"])}
        for point in list(openfoam_cfg.get("truth_operating_points", []) or [])
    ]
    filtered = [point for point in truth_points if _is_inside_core(point, core)]
    if filtered:
        return filtered
    return [
        {"rpm": float(core.get("rpm_min", 1800.0)), "torque": float(core.get("torque_min", 80.0))},
        {"rpm": float(core.get("rpm_min", 1800.0)), "torque": float(core.get("torque_max", 160.0))},
        {"rpm": float(core.get("rpm_max", 2800.0)), "torque": float(core.get("torque_min", 80.0))},
        {"rpm": float(core.get("rpm_max", 2800.0)), "torque": float(core.get("torque_max", 160.0))},
    ]


def _deepcopy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def _resolved_payload(run_manifest) -> dict[str, Any]:
    raw = str(run_manifest.solver_artifacts.get("resolved_simulation_data_json", "")).strip()
    if not raw:
        return {}
    return json.loads(raw)


def _metric_map(run_manifest) -> dict[str, float]:
    return {
        result.metric_id: float(result.simulated_value) for result in run_manifest.metric_results
    }


def _estimate_trapped_o2_mass(bundle: dict[str, Any]) -> float:
    species = dict(bundle.get("species_mole_fractions", {}) or {})
    total_mass = float(bundle.get("total_mass_kg", 0.0))
    x_o2 = float(species.get("O2", 0.0))
    return max(0.0, total_mass * x_o2 * 0.95)


def _build_truth_record(
    *,
    point: dict[str, float],
    candidate_id: str,
    suite_manifest,
) -> dict[str, Any]:
    regime_statuses = {
        regime: run.status.value for regime, run in suite_manifest.regime_results.items()
    }
    resolved = {
        regime: _resolved_payload(run) for regime, run in suite_manifest.regime_results.items()
    }
    chemistry_run = suite_manifest.regime_results.get("chemistry")
    spray_run = suite_manifest.regime_results.get("spray")
    reacting_run = suite_manifest.regime_results.get("reacting_flow")
    closed_run = suite_manifest.regime_results.get("closed_cylinder")
    handoff_run = suite_manifest.regime_results.get("full_handoff")
    handoff_payload = resolved.get("full_handoff", {})
    handoff_bundle = dict(handoff_payload.get("handoff_bundle", {}) or {})

    trapped_mass = float(
        resolved.get("reacting_flow", {}).get(
            "trapped_mass",
            resolved.get("spray", {}).get("trapped_mass", handoff_bundle.get("total_mass_kg", 0.0)),
        )
    )
    residual_fraction = float(
        resolved.get("reacting_flow", {}).get(
            "residual_fraction",
            max(0.0, 1.0 - float(handoff_bundle.get("mixture_homogeneity_index", 1.0))),
        )
    )
    trapped_o2_mass = float(
        resolved.get("reacting_flow", {}).get(
            "trapped_o2_mass",
            _estimate_trapped_o2_mass(handoff_bundle),
        )
    )

    return {
        "candidate_id": candidate_id,
        "operating_point": {"rpm": float(point["rpm"]), "torque": float(point["torque"])},
        "suite_id": suite_manifest.suite_id,
        "suite_passed": bool(suite_manifest.overall_passed),
        "truth_valid": bool(suite_manifest.overall_passed),
        "first_blocking_regime": str(suite_manifest.first_blocking_regime),
        "first_blocking_metric_group": str(suite_manifest.first_blocking_metric_group),
        "regime_statuses": regime_statuses,
        "chemistry_outputs": _metric_map(chemistry_run) if chemistry_run is not None else {},
        "openfoam_outputs": {
            "spray_metrics": _metric_map(spray_run) if spray_run is not None else {},
            "reacting_flow_metrics": _metric_map(reacting_run) if reacting_run is not None else {},
            "trapped_mass": trapped_mass,
            "residual_fraction": residual_fraction,
            "trapped_o2_mass": trapped_o2_mass,
        },
        "closed_cylinder_outputs": _metric_map(closed_run) if closed_run is not None else {},
        "handoff_bundle_id": str(handoff_bundle.get("bundle_id", "")),
        "handoff_conservation_status": (
            handoff_run.status.value if handoff_run is not None else "not_run"
        ),
        "handoff_bundle": handoff_bundle,
        "suite_scoreboard": [
            {
                "regime": entry.regime,
                "status": entry.status.value,
                "n_metrics_total": int(entry.n_metrics_total),
                "n_metrics_failed": int(entry.n_metrics_failed),
            }
            for entry in suite_manifest.scoreboard
        ],
        "provenance": {
            "mechanism_path": str(
                dict(resolved.get("chemistry", {}).get("mechanism_provenance", {}) or {}).get(
                    "mechanism_file",
                    "",
                )
            ),
            "spray_template": str(
                dict(resolved.get("spray", {}).get("spray_provenance", {}) or {}).get(
                    "template_dir",
                    "",
                )
            ),
            "reacting_template": str(
                dict(
                    resolved.get("reacting_flow", {}).get("reacting_flow_provenance", {}) or {}
                ).get("template_dir", "")
            ),
            "handoff_bundle_id": str(handoff_bundle.get("bundle_id", "")),
            "closed_cylinder_fixture": str(
                dict(
                    resolved.get("closed_cylinder", {}).get("closed_cylinder_provenance", {}) or {}
                ).get("fixture_results_path", "")
            ),
        },
    }


def run_combustion_truth_workflow(
    *,
    suite_config_path: str | Path,
    outdir: str | Path,
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
    max_points: int | None = None,
) -> dict[str, Any]:
    """Run the four-regime gas-combustion truth suite over the DOE core corridor."""
    suite_cfg = _load_json(suite_config_path)
    suite_profile = _build_suite_profile(suite_cfg)
    operating_points = load_core_operating_points(profile_path)
    if max_points is not None:
        operating_points = operating_points[: max(0, int(max_points))]

    out_root = Path(outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    truth_records_path = out_root / "combustion_truth_records.jsonl"
    truth_records_path.write_text("", encoding="utf-8")
    suite_index: list[dict[str, Any]] = []

    for index, point in enumerate(operating_points):
        candidate_id = (
            f"core_rpm{int(round(point['rpm']))}_tq{int(round(point['torque']))}_{index:02d}"
        )
        point_root = out_root / candidate_id
        validation_root = point_root / "validation"
        validation_root.mkdir(parents=True, exist_ok=True)

        regime_configs: dict[str, dict[str, Any]] = {}
        for regime_name in suite_profile.normalized_regime_order():
            regime_cfg = _deepcopy_payload(
                dict(suite_cfg.get("regimes", {}).get(regime_name, {}) or {})
            )
            if not regime_cfg:
                continue
            case_cfg = regime_cfg.setdefault("case_spec", {})
            operating_point = dict(case_cfg.get("operating_point", {}) or {})
            operating_point.update(
                {
                    "rpm": float(point["rpm"]),
                    "torque": float(point["torque"]),
                }
            )
            case_cfg["operating_point"] = operating_point
            case_cfg["case_id"] = f"{case_cfg.get('case_id', regime_name)}_{candidate_id}"

            solver_cfg = dict(case_cfg.get("solver_config", {}) or {})
            adapter_cfg = dict(solver_cfg.get("simulation_adapter", {}) or {})
            if adapter_cfg.get("backend") in {"openfoam_case", "docker_openfoam"}:
                adapter_cfg["run_dir"] = str(point_root / "cases" / regime_name)
                solver_cfg["simulation_adapter"] = adapter_cfg
                case_cfg["solver_config"] = solver_cfg
            regime_cfg["case_spec"] = case_cfg

            dataset, case_spec = _build_dataset_and_case(regime_cfg, regime_name)
            regime_configs[regime_name] = {
                "dataset": dataset,
                "case_spec": case_spec,
                "simulation_data": dict(regime_cfg.get("simulation_data", {}) or {}),
            }

        suite_manifest = run_suite(regime_configs, suite_profile=suite_profile)
        suite_to_json(suite_manifest, validation_root / "suite_manifest.json")
        (validation_root / "suite_summary.md").write_text(
            suite_to_markdown(suite_manifest),
            encoding="utf-8",
        )

        truth_record = _build_truth_record(
            point=point,
            candidate_id=candidate_id,
            suite_manifest=suite_manifest,
        )
        with truth_records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(truth_record, ensure_ascii=True) + "\n")
        suite_index.append(
            {
                "candidate_id": candidate_id,
                "rpm": float(point["rpm"]),
                "torque": float(point["torque"]),
                "suite_passed": bool(suite_manifest.overall_passed),
                "suite_manifest": str(validation_root / "suite_manifest.json"),
            }
        )

    summary = {
        "suite_id": suite_profile.suite_id,
        "profile_path": str(Path(profile_path)),
        "suite_config_path": str(Path(suite_config_path)),
        "n_points": int(len(operating_points)),
        "n_passed": int(sum(1 for rec in suite_index if rec["suite_passed"])),
        "truth_records_path": str(truth_records_path),
        "points": suite_index,
    }
    (out_root / "combustion_truth_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    summary_md = [
        "# Gas Combustion Truth Summary",
        "",
        f"Suite ID: `{suite_profile.suite_id}`",
        f"Points run: {summary['n_points']}",
        f"Points passed: {summary['n_passed']}",
        f"Truth records: `{truth_records_path}`",
        "",
    ]
    (out_root / "combustion_truth_summary.md").write_text(
        "\n".join(summary_md),
        encoding="utf-8",
    )
    return summary
