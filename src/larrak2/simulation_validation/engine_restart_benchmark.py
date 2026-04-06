"""Benchmark staged engine restart profiles from an existing checkpoint."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.pipelines.openfoam import OpenFoamPipeline

from .engine_results import build_engine_progress_summary, emit_engine_progress_artifacts
from .engine_runtime_mechanism import resolve_engine_runtime_package
from .runtime_chemistry_table import build_runtime_chemistry_table
from .tuning_characterization_study import maybe_log_tuning_observation


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _materialize_runtime_coverage_corpus_npz(engine_case_dir: Path) -> Path | None:
    json_path = engine_case_dir / "runtimeChemistryCoverageCorpus.json"
    if not json_path.exists():
        return None
    payload = _load_json(json_path)
    rows = [item for item in list(payload.get("rows", []) or []) if isinstance(item, dict)]
    if not rows:
        return None
    state_variables = [
        str(value) for value in list(payload.get("state_variables", []) or []) if str(value)
    ]
    if not state_variables:
        first_raw = dict(rows[0].get("raw_state", {}) or {})
        state_variables = list(first_raw.keys())
    bucket_width = max(len(list(row.get("coverage_bucket_key", []) or [])) for row in rows)

    def _numeric(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        return float(value)

    raw_states = np.asarray(
        [
            [
                _numeric(dict(row.get("raw_state", {}) or {}).get(label), float("nan"))
                for label in state_variables
            ]
            for row in rows
        ],
        dtype=float,
    )
    transformed_states = np.asarray(
        [
            [
                _numeric(dict(row.get("transformed_state", {}) or {}).get(label), float("nan"))
                for label in state_variables
            ]
            for row in rows
        ],
        dtype=float,
    )
    coverage_bucket_keys = np.asarray(
        [
            [
                _numeric(
                    (
                        list(row.get("coverage_bucket_key", []) or [])
                        + [float("nan")] * bucket_width
                    )[index],
                    float("nan"),
                )
                for index in range(bucket_width)
            ]
            for row in rows
        ],
        dtype=float,
    )
    npz_path = engine_case_dir / "runtimeChemistryCoverageCorpus.npz"
    np.savez_compressed(
        npz_path,
        state_variables=np.asarray(state_variables),
        raw_states=raw_states,
        transformed_states=transformed_states,
        coverage_bucket_keys=coverage_bucket_keys,
        query_count=np.asarray([_numeric(row.get("query_count")) for row in rows], dtype=float),
        table_hit_count=np.asarray(
            [_numeric(row.get("table_hit_count")) for row in rows], dtype=float
        ),
        coverage_reject_count=np.asarray(
            [_numeric(row.get("coverage_reject_count")) for row in rows], dtype=float
        ),
        trust_reject_count=np.asarray(
            [_numeric(row.get("trust_reject_count")) for row in rows], dtype=float
        ),
        worst_reject_excess=np.asarray(
            [_numeric(row.get("worst_reject_excess")) for row in rows], dtype=float
        ),
        nearest_sample_distance_min=np.asarray(
            [_numeric(row.get("nearest_sample_distance_min")) for row in rows], dtype=float
        ),
        nearest_sample_distance_max=np.asarray(
            [_numeric(row.get("nearest_sample_distance_max")) for row in rows], dtype=float
        ),
        high_fidelity_trust_reject=np.asarray(
            [float(bool(row.get("high_fidelity_trust_reject"))) for row in rows], dtype=float
        ),
    )
    return npz_path


def _resolve_restart_profile(profile_name: str) -> str:
    normalized = str(profile_name).strip().lower()
    mapping = {
        "default": "closed_valve_ignition_v1",
        "current": "closed_valve_ignition_v1",
        "multitable": "closed_valve_ignition_multitable_v1",
        "fast_runtime": "closed_valve_ignition_fast_runtime_v1",
        "low_clamp": "closed_valve_ignition_low_clamp_v1",
        "closed_valve_ignition_v1": "closed_valve_ignition_v1",
        "closed_valve_ignition_multitable_v1": "closed_valve_ignition_multitable_v1",
        "closed_valve_ignition_fast_runtime_v1": "closed_valve_ignition_fast_runtime_v1",
        "closed_valve_ignition_low_clamp_v1": "closed_valve_ignition_low_clamp_v1",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported restart profile: {profile_name}")
    return mapping[normalized]


def _resolve_stage_runtime_table_dirs(
    strategy: dict[str, Any],
    *,
    refresh_runtime_tables: bool,
) -> dict[str, Path]:
    runtime_entry = dict(strategy.get("runtime_package", {}) or {})
    resolved: dict[str, Path] = {}
    for stage_name, raw_entry in dict(runtime_entry.get("stage_runtime_tables", {}) or {}).items():
        table_dir = _ensure_runtime_table_dir(
            dict(raw_entry or {}),
            refresh=refresh_runtime_tables,
        )
        if table_dir is not None:
            resolved[str(stage_name).strip()] = table_dir
    return resolved


def _ensure_runtime_table_dir(entry: dict[str, Any], *, refresh: bool = False) -> Path | None:
    table_dir = (
        Path(str(entry.get("runtime_table_dir", "")).strip())
        if str(entry.get("runtime_table_dir", "")).strip()
        else None
    )
    if table_dir and (table_dir / "runtime_chemistry_table_manifest.json").exists() and not refresh:
        return table_dir
    config_path = str(entry.get("runtime_table_config_path", "")).strip()
    if not config_path:
        return table_dir
    manifest = build_runtime_chemistry_table(
        config_path=config_path,
        refresh=refresh,
        repo_root=Path.cwd(),
    )
    return Path(str(manifest["files"]["runtimeChemistryTable"])).parent


def _build_case_params(
    *,
    tuned_params: dict[str, Any],
    handoff_artifact: dict[str, Any],
    solver_name: str,
    engine_stage_profile: str,
    chemistry_package_dir: str,
) -> dict[str, Any]:
    handoff_bundle = dict(handoff_artifact["handoff_bundle"])
    params = dict(tuned_params)
    params.update(
        {
            "engine_proof_mode": "reacting_staged_ignition",
            "engine_stage_profile": str(engine_stage_profile),
            "engine_start_angle_deg": float(handoff_bundle["cycle_coordinate_deg"]),
            "engine_end_angle_deg": 0.0,
            "solver_name": str(solver_name),
            "derive_end_time_from_angles": True,
            "openfoam_chemistry_package_dir": str(chemistry_package_dir),
        }
    )
    return params


def _latest_logsummary(run_dir: Path) -> dict[str, float]:
    files = sorted(
        run_dir.glob("logSummary.*.dat"),
        key=lambda path: float(path.name[len("logSummary.") : -len(".dat")]),
    )
    if not files:
        raise FileNotFoundError(f"No logSummary artifacts found in {run_dir}")
    latest = files[-1]
    rows = [
        line.strip()
        for line in latest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    crank, pressure, temperature, velocity = rows[-1].split()
    return {
        "time_s": float(latest.name[len("logSummary.") : -len(".dat")]),
        "crank_angle_deg": float(crank),
        "mean_pressure_Pa": float(pressure),
        "mean_temperature_K": float(temperature),
        "mean_velocity_magnitude_m_s": float(velocity),
    }


def _resolve_numeric_time_dir(run_dir: Path, target_time_s: float) -> Path:
    numeric_dirs: list[tuple[float, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            numeric_dirs.append((float(child.name), child))
        except ValueError:
            continue
    if not numeric_dirs:
        raise FileNotFoundError(f"No numeric time directories found in {run_dir}")
    numeric_time, path = min(numeric_dirs, key=lambda item: abs(item[0] - float(target_time_s)))
    tolerance = max(1.0e-12, abs(float(target_time_s)) * 1.0e-9)
    if abs(numeric_time - float(target_time_s)) > tolerance:
        raise FileNotFoundError(f"No checkpoint time directory found near t={target_time_s:.12g}")
    return path


def _copy_checkpoint_case(
    *,
    base_run_dir: Path,
    output_run_dir: Path,
    checkpoint_dir: Path,
) -> None:
    if output_run_dir.exists():
        shutil.rmtree(output_run_dir)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("0", "constant", "system", "chemistry"):
        source = base_run_dir / name
        if source.exists():
            shutil.copytree(source, output_run_dir / name)
    for file_name in ("engine_stage_manifest.json", "engine_stage_resume_summary.json"):
        source = base_run_dir / file_name
        if source.exists():
            shutil.copy2(source, output_run_dir / file_name)
    shutil.copytree(checkpoint_dir, output_run_dir / checkpoint_dir.name)


def _stage_inputs(
    *,
    run_dir: Path,
    staged_inputs: list[dict[str, str]],
) -> None:
    for entry in staged_inputs:
        source = Path(str(entry["source"]))
        target = run_dir / str(entry["target"])
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def _total_numeric_hits(summary: dict[str, Any]) -> int:
    numeric = dict(summary.get("numeric_stability", {}) or {})
    return sum(int(numeric.get(key, 0)) for key in numeric)


def benchmark_engine_restart_profiles(
    *,
    run_dir: str | Path,
    tuned_params_path: str | Path,
    handoff_artifact_path: str | Path,
    outdir: str | Path,
    profiles: list[str],
    window_angle_deg: float = 0.01,
    solver_name: str = "larrakEngineFoam",
    docker_timeout_s: int = 1800,
    runtime_strategy_config: str
    | Path = "data/simulation_validation/engine_runtime_mechanism_strategy.json",
    package_label: str = "",
    docker_image: str | None = None,
    docker_bin: str | None = None,
    refresh_runtime_tables: bool = False,
    continue_across_remaining_stages: bool = False,
    refresh_custom_solver: bool = False,
    tuning_characterization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_run_dir = Path(run_dir)
    tuned_params = _load_json(tuned_params_path)
    handoff_artifact = _load_json(handoff_artifact_path)
    strategy = _load_json(runtime_strategy_config)
    runtime_package_dir, runtime_manifest = resolve_engine_runtime_package(
        config_path=runtime_strategy_config,
        package_label=package_label,
    )

    base_latest = _latest_logsummary(base_run_dir)
    checkpoint_dir = _resolve_numeric_time_dir(base_run_dir, float(base_latest["time_s"]))
    output_root = Path(outdir)
    output_root.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    runtime_entry = dict(strategy.get("runtime_package", {}) or {})
    checkpoint_entries = [
        dict(item) for item in list(strategy.get("checkpoint_packages", []) or [])
    ]
    stage_runtime_table_dirs = _resolve_stage_runtime_table_dirs(
        strategy,
        refresh_runtime_tables=refresh_runtime_tables,
    )

    for profile_name in profiles:
        normalized_profile = str(profile_name).strip()
        mode_name = normalized_profile.lower()
        if (
            mode_name in {"chem323_lookup_strict", "chem323_lookup", "chem323_lookup_permissive"}
            and stage_runtime_table_dirs
        ):
            resolved_profile = "closed_valve_ignition_multitable_v1"
        elif normalized_profile in {
            "default",
            "current",
            "multitable",
            "fast_runtime",
            "low_clamp",
            "closed_valve_ignition_v1",
            "closed_valve_ignition_multitable_v1",
            "closed_valve_ignition_fast_runtime_v1",
            "closed_valve_ignition_low_clamp_v1",
        }:
            resolved_profile = _resolve_restart_profile(profile_name)
        else:
            resolved_profile = "closed_valve_ignition_fast_runtime_v1"

        selected_package_dir = runtime_package_dir
        selected_manifest = runtime_manifest
        runtime_table_dir: Path | None = None
        runtime_mode_override: str | None = None

        if mode_name == "chem323_direct":
            runtime_mode_override = "fullReducedKinetics"
        elif mode_name in {"chem323_lookup", "chem323_lookup_permissive"}:
            runtime_mode_override = "lookupTablePermissive"
            runtime_table_dir = _ensure_runtime_table_dir(
                runtime_entry,
                refresh=refresh_runtime_tables,
            )
        elif mode_name == "chem323_lookup_strict":
            runtime_mode_override = "lookupTableStrict"
            runtime_table_dir = _ensure_runtime_table_dir(
                runtime_entry,
                refresh=refresh_runtime_tables,
            )
        elif (
            mode_name
            in {
                "chem679_direct",
                "chem679_direct_reference",
                "chem679_lookup",
                "chem679_lookup_permissive",
                "chem679_lookup_strict",
            }
            and checkpoint_entries
        ):
            selected_package_dir, selected_manifest = resolve_engine_runtime_package(
                config_path=runtime_strategy_config,
                package_label=str(checkpoint_entries[0].get("label", "")),
                refresh_packages=False,
            )
            if mode_name in {"chem679_lookup_strict"}:
                runtime_mode_override = "lookupTableStrict"
            elif mode_name in {"chem679_lookup", "chem679_lookup_permissive"}:
                runtime_mode_override = "lookupTablePermissive"
            else:
                runtime_mode_override = "fullReducedKinetics"
            if runtime_mode_override != "fullReducedKinetics":
                runtime_table_dir = _ensure_runtime_table_dir(
                    checkpoint_entries[0],
                    refresh=refresh_runtime_tables,
                )
        elif mode_name in {"fast_runtime", "closed_valve_ignition_fast_runtime_v1"}:
            runtime_mode_override = "lookupTableStrict"
            runtime_table_dir = _ensure_runtime_table_dir(
                runtime_entry, refresh=refresh_runtime_tables
            )
        else:
            runtime_mode_override = "fullReducedKinetics"

        case_params = _build_case_params(
            tuned_params=tuned_params,
            handoff_artifact=handoff_artifact,
            solver_name=solver_name,
            engine_stage_profile=resolved_profile,
            chemistry_package_dir=str(selected_package_dir),
        )
        if runtime_mode_override:
            case_params["runtime_chemistry_mode"] = runtime_mode_override
        if runtime_table_dir is not None:
            case_params["openfoam_runtime_chemistry_table_dir"] = str(runtime_table_dir)
        if stage_runtime_table_dirs:
            case_params["engine_stage_runtime_chemistry_table_dirs"] = {
                stage_name: str(path) for stage_name, path in stage_runtime_table_dirs.items()
            }
        pipeline = OpenFoamPipeline(
            template_dir=Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case"),
            solver_cmd=str(solver_name),
            docker_timeout_s=int(docker_timeout_s),
            docker_image=docker_image,
            docker_bin=docker_bin,
            chemistry_package_dir=selected_package_dir,
            runtime_chemistry_table_dir=runtime_table_dir,
            refresh_custom_solver=bool(refresh_custom_solver),
        )
        case_params, case_staged_inputs, _, package_manifest, runtime_table_manifest = (
            pipeline._engine_case_assets(  # noqa: SLF001
                case_params,
                handoff_bundle=dict(handoff_artifact["handoff_bundle"]),
                staged_inputs=None,
            )
        )
        remaining = pipeline._remaining_engine_stages(  # noqa: SLF001
            base_params=case_params,
            manifest=pipeline._load_engine_stage_manifest(base_run_dir),  # noqa: SLF001
        )
        if not remaining:
            raise RuntimeError("No remaining stages found to benchmark")
        target_window_end_angle = float(base_latest["crank_angle_deg"]) + abs(
            float(window_angle_deg)
        )
        selected_stages: list[dict[str, Any]] = []
        if continue_across_remaining_stages:
            for stage in remaining:
                stage_payload = dict(stage)
                stage_payload["writeInterval"] = 1
                if runtime_mode_override:
                    stage_payload["runtime_chemistry_mode"] = runtime_mode_override
                if float(stage_payload["end_angle_deg"]) >= target_window_end_angle:
                    stage_payload["end_angle_deg"] = min(
                        float(stage_payload["end_angle_deg"]),
                        target_window_end_angle,
                    )
                    selected_stages.append(stage_payload)
                    break
                selected_stages.append(stage_payload)
            if not selected_stages:
                stage_payload = dict(remaining[0])
                stage_payload["end_angle_deg"] = min(
                    float(stage_payload["end_angle_deg"]),
                    target_window_end_angle,
                )
                stage_payload["writeInterval"] = 1
                if runtime_mode_override:
                    stage_payload["runtime_chemistry_mode"] = runtime_mode_override
                selected_stages = [stage_payload]
        else:
            stage_payload = dict(remaining[0])
            stage_payload["end_angle_deg"] = min(
                float(stage_payload["end_angle_deg"]),
                target_window_end_angle,
            )
            stage_payload["writeInterval"] = 1
            if runtime_mode_override:
                stage_payload["runtime_chemistry_mode"] = runtime_mode_override
            selected_stages = [stage_payload]
        target_end_angle = float(selected_stages[-1]["end_angle_deg"])

        benchmark_run_dir = output_root / str(profile_name)
        _copy_checkpoint_case(
            base_run_dir=base_run_dir,
            output_run_dir=benchmark_run_dir,
            checkpoint_dir=checkpoint_dir,
        )
        _stage_inputs(
            run_dir=benchmark_run_dir,
            staged_inputs=case_staged_inputs,
        )
        preflight = pipeline.docker_preflight(log_file=benchmark_run_dir / "docker_preflight.log")
        if not bool(preflight.get("ok", False)):
            runs.append(
                {
                    "profile_name": str(profile_name),
                    "resolved_profile": resolved_profile,
                    "runtime_mode": str(
                        runtime_mode_override or case_params.get("runtime_chemistry_mode", "")
                    ),
                    "benchmark_run_dir": str(benchmark_run_dir),
                    "solver_ok": False,
                    "stage_result": str(
                        preflight.get("failure_class", "") or "docker_launch_failed"
                    ),
                    "failed_stage_name": "",
                    "target_end_angle_deg": float(target_end_angle),
                    "baseline_start": base_latest,
                    "latest_checkpoint": {},
                    "wall_elapsed_s": 0.0,
                    "wall_seconds_per_0p01deg": None,
                    "angle_advance_deg": None,
                    "sim_time_advance_s": None,
                    "speed_score_deg_per_s": None,
                    "total_numeric_hits": 0,
                    "clamp_score_hits_per_deg": None,
                    "floor_hits_per_deg": None,
                    "executed_stage_names": [],
                    "executed_stage_count": 0,
                    "executed_stage_runtime_tables": [],
                    "first_miss_class": "",
                    "first_miss_branch_id": "",
                    "first_miss_sign_flip": False,
                    "first_offending_variables": [],
                    "miss_counts_by_variable": {},
                    "max_out_of_bound_by_variable": {},
                    "max_qdot_relative_error": None,
                    "max_qdot_transformed_envelope_excess": None,
                    "chem323_maturity_gate_passed": False,
                    "chem323_runtime_replacement_gate_passed": False,
                    "runtime_package_id": str(selected_manifest.get("package_id", "")),
                    "runtime_table_id": (
                        str(runtime_table_manifest.get("table_id", ""))
                        if runtime_table_dir is not None
                        else ""
                    ),
                    "runtime_table_hash": str(
                        dict(runtime_table_manifest.get("generated_file_hashes", {}) or {}).get(
                            "runtimeChemistryTable", ""
                        )
                    ),
                    "runtime_table_dir": ""
                    if runtime_table_dir is None
                    else str(runtime_table_dir),
                    "runtime_chemistry_authority_miss_path": "",
                    "runtime_coverage_corpus_path": "",
                    "runtime_coverage_corpus_npz_path": "",
                    "numeric_stability": {},
                    "runtime_chemistry": {},
                    "docker_bin": str(preflight.get("docker_bin", "") or ""),
                    "docker_preflight_ok": False,
                    "docker_autostart_attempted": bool(
                        preflight.get("docker_autostart_attempted", False)
                    ),
                    "docker_autostart_succeeded": bool(
                        preflight.get("docker_autostart_succeeded", False)
                    ),
                    "launch_failure_class": str(preflight.get("failure_class", "") or ""),
                    "launch_failure_details": str(preflight.get("message", "") or ""),
                    "launch_failure_candidates": list(preflight.get("candidate_paths", []) or []),
                }
            )
            continue

        solver_metadata = pipeline._ensure_custom_solver(
            log_file=benchmark_run_dir / "custom_solver_build.benchmark.log"
        )
        custom_solver_dirs = (
            [Path(str(solver_metadata["binary_path"])).parent]
            if solver_metadata.get("binary_path")
            else None
        )
        wall_start = time.perf_counter()
        ok, stage_result, manifest_entries = pipeline._run_selected_engine_stages(  # noqa: SLF001
            benchmark_run_dir,
            base_params=case_params,
            stages=selected_stages,
            custom_solver_dirs=custom_solver_dirs,
        )
        wall_elapsed_s = time.perf_counter() - wall_start
        emit_engine_progress_artifacts(
            engine_case_dir=benchmark_run_dir,
            params=case_params,
            solver_name=str(solver_name),
            openfoam_chemistry_package_id=str(package_manifest.get("package_id", "")),
            openfoam_chemistry_package_hash=str(package_manifest.get("package_hash", "")),
            custom_solver_source_hash=str(solver_metadata.get("source_hash", "")),
        )
        _materialize_runtime_coverage_corpus_npz(benchmark_run_dir)
        summary = build_engine_progress_summary(
            engine_case_dir=benchmark_run_dir,
            params=case_params,
            solver_name=str(solver_name),
            openfoam_chemistry_package_id=str(package_manifest.get("package_id", "")),
            openfoam_chemistry_package_hash=str(package_manifest.get("package_hash", "")),
            custom_solver_source_hash=str(solver_metadata.get("source_hash", "")),
        )
        latest = dict(summary.get("latest_checkpoint", {}) or {})
        angle_advance = (
            None
            if "crank_angle_deg" not in latest
            else float(latest["crank_angle_deg"]) - float(base_latest["crank_angle_deg"])
        )
        sim_time_advance = (
            None
            if "time_s" not in latest
            else float(latest["time_s"]) - float(base_latest["time_s"])
        )
        total_hits = _total_numeric_hits(summary)
        runtime_counters = dict(summary.get("runtime_chemistry", {}) or {})
        first_miss_class = str(summary.get("first_miss_class", "") or "").strip()
        first_miss_branch_id = str(summary.get("first_miss_branch_id", "") or "").strip()
        first_miss_sign_flip = bool(summary.get("first_miss_sign_flip", False))
        latest_angle = latest.get("crank_angle_deg")
        reached_target_window = (
            latest_angle is not None and float(latest_angle) >= float(target_end_angle) - 1.0e-4
        )
        replacement_gate = bool(
            str(selected_manifest.get("package_id", "")).startswith("chem323")
            and str(runtime_mode_override or case_params.get("runtime_chemistry_mode", ""))
            == "lookupTableStrict"
            and reached_target_window
            and not first_miss_class
            and float(runtime_counters.get("coverage_reject_cells", 0.0) or 0.0) == 0.0
            and float(runtime_counters.get("fallback_timesteps", 0.0) or 0.0) == 0.0
            and float(runtime_counters.get("trust_region_reject_cells", 0.0) or 0.0) == 0.0
            and not first_miss_sign_flip
            and total_hits == 0
        )
        maturity_gate = bool(
            str(selected_manifest.get("package_id", "")).startswith("chem323")
            and str(runtime_mode_override or case_params.get("runtime_chemistry_mode", ""))
            == "lookupTableStrict"
            and float(runtime_counters.get("coverage_reject_cells", 0.0) or 0.0) == 0.0
            and float(runtime_counters.get("fallback_timesteps", 0.0) or 0.0) == 0.0
            and not first_miss_sign_flip
            and total_hits == 0
        )
        speed_score = None
        if wall_elapsed_s > 0.0 and angle_advance is not None:
            speed_score = float(angle_advance) / float(wall_elapsed_s)
        clamp_score = None
        if angle_advance is not None and abs(float(angle_advance)) > 0.0:
            clamp_score = float(total_hits) / abs(float(angle_advance))
        wall_seconds_per_0p01deg = None
        if wall_elapsed_s > 0.0 and angle_advance is not None and abs(float(angle_advance)) > 0.0:
            wall_seconds_per_0p01deg = float(wall_elapsed_s) / (abs(float(angle_advance)) / 0.01)
        floor_hits_per_deg = None
        if angle_advance is not None and abs(float(angle_advance)) > 0.0:
            floor_hits_per_deg = float(total_hits) / abs(float(angle_advance))
        executed_stage_runtime_tables = [
            {
                "stage_name": str(entry.get("name", "")),
                "runtime_table_id": str(entry.get("runtime_table_id", "")),
                "runtime_table_hash": str(entry.get("runtime_table_hash", "")),
                "runtime_table_dir": str(entry.get("runtime_table_dir", "")),
                "target_end_angle_deg": float(entry.get("end_angle_deg", 0.0)),
                "latest_angle_deg": (
                    None
                    if not dict(entry.get("completion_status", {}) or {}).get("latest_angle_deg")
                    else float(dict(entry.get("completion_status", {}) or {})["latest_angle_deg"])
                ),
                "ok": bool(entry.get("ok", False)),
            }
            for entry in manifest_entries
        ]
        failed_stage_name = str(stage_result) if not ok else ""
        stage_failure_class = (
            str(manifest_entries[-1].get("stage_result", ""))
            if (manifest_entries and not ok)
            else ""
        )
        run_record = {
            "profile_name": str(profile_name),
            "resolved_profile": resolved_profile,
            "runtime_mode": str(
                runtime_mode_override or case_params.get("runtime_chemistry_mode", "")
            ),
            "benchmark_run_dir": str(benchmark_run_dir),
            "solver_ok": bool(ok),
            "stage_result": stage_failure_class,
            "failed_stage_name": failed_stage_name,
            "target_end_angle_deg": float(target_end_angle),
            "baseline_start": base_latest,
            "latest_checkpoint": latest,
            "wall_elapsed_s": float(wall_elapsed_s),
            "wall_seconds_per_0p01deg": wall_seconds_per_0p01deg,
            "angle_advance_deg": angle_advance,
            "sim_time_advance_s": sim_time_advance,
            "speed_score_deg_per_s": speed_score,
            "total_numeric_hits": int(total_hits),
            "clamp_score_hits_per_deg": clamp_score,
            "floor_hits_per_deg": floor_hits_per_deg,
            "executed_stage_names": [str(entry.get("name", "")) for entry in manifest_entries],
            "executed_stage_count": int(len(manifest_entries)),
            "executed_stage_runtime_tables": executed_stage_runtime_tables,
            "first_miss_class": first_miss_class,
            "first_miss_branch_id": first_miss_branch_id,
            "first_miss_sign_flip": first_miss_sign_flip,
            "first_offending_variables": list(summary.get("first_offending_variables", []) or []),
            "miss_counts_by_variable": dict(summary.get("miss_counts_by_variable", {}) or {}),
            "max_out_of_bound_by_variable": dict(
                summary.get("max_out_of_bound_by_variable", {}) or {}
            ),
            "max_qdot_relative_error": summary.get("max_qdot_relative_error"),
            "max_qdot_transformed_envelope_excess": summary.get(
                "max_qdot_transformed_envelope_excess"
            ),
            "chem323_maturity_gate_passed": maturity_gate,
            "chem323_runtime_replacement_gate_passed": replacement_gate,
            "runtime_package_id": str(selected_manifest.get("package_id", "")),
            "runtime_table_id": (
                str(executed_stage_runtime_tables[-1]["runtime_table_id"])
                if executed_stage_runtime_tables
                else str(runtime_table_manifest.get("table_id", ""))
            ),
            "runtime_table_hash": (
                str(executed_stage_runtime_tables[-1]["runtime_table_hash"])
                if executed_stage_runtime_tables
                else ""
            ),
            "runtime_table_dir": (
                str(executed_stage_runtime_tables[-1]["runtime_table_dir"])
                if executed_stage_runtime_tables
                else ("" if runtime_table_dir is None else str(runtime_table_dir))
            ),
            "runtime_chemistry_authority_miss_path": (
                str(benchmark_run_dir / "runtimeChemistryAuthorityMiss.json")
                if (benchmark_run_dir / "runtimeChemistryAuthorityMiss.json").exists()
                else ""
            ),
            "runtime_coverage_corpus_path": str(
                runtime_counters.get("runtime_coverage_corpus_path", "") or ""
            ),
            "runtime_coverage_corpus_npz_path": str(
                runtime_counters.get("runtime_coverage_corpus_npz_path", "") or ""
            ),
            "numeric_stability": dict(summary.get("numeric_stability", {}) or {}),
            "runtime_chemistry": runtime_counters,
            "docker_bin": str(preflight.get("docker_bin", "") or ""),
            "docker_preflight_ok": True,
            "docker_autostart_attempted": bool(preflight.get("docker_autostart_attempted", False)),
            "docker_autostart_succeeded": bool(preflight.get("docker_autostart_succeeded", False)),
            "launch_failure_class": "",
            "launch_failure_details": "",
            "launch_failure_candidates": list(preflight.get("candidate_paths", []) or []),
        }
        runs.append(run_record)

    fastest = max(
        (item for item in runs if item.get("speed_score_deg_per_s") is not None),
        key=lambda item: float(item["speed_score_deg_per_s"]),
        default=None,
    )
    cleanest = min(
        (item for item in runs if item.get("clamp_score_hits_per_deg") is not None),
        key=lambda item: float(item["clamp_score_hits_per_deg"]),
        default=None,
    )
    summary = {
        "base_run_dir": str(base_run_dir),
        "checkpoint_time_s": float(base_latest["time_s"]),
        "checkpoint_angle_deg": float(base_latest["crank_angle_deg"]),
        "runtime_package_id": str(runtime_manifest.get("package_id", "")),
        "runtime_package_hash": str(runtime_manifest.get("package_hash", "")),
        "window_angle_deg": float(window_angle_deg),
        "profiles": runs,
        "recommendation": {
            "fastest_profile": None if fastest is None else str(fastest["profile_name"]),
            "lowest_clamp_profile": None if cleanest is None else str(cleanest["profile_name"]),
        },
    }
    output_path = output_root / "engine_restart_benchmark_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    tc = tuning_characterization
    if tc and bool(tc.get("enabled")):
        profile_names = [str(r.get("profile_name", "")) for r in runs]
        tuning_payload = maybe_log_tuning_observation(
            benchmark_outdir=output_root,
            experiments_jsonl=tc.get("experiments_jsonl", ""),
            knob_schema_path=tc.get("knob_schema_path", ""),
            table_config_path=tc.get("table_config_path", ""),
            strategy_config_path=tc.get("strategy_config_path"),
            profile_name=tc.get("profile_name"),
            profile_names_in_benchmark=profile_names,
            repo_root=tc.get("repo_root"),
        )
        summary["tuning_characterization"] = tuning_payload
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return summary


__all__ = ["benchmark_engine_restart_profiles"]
