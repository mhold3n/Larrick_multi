from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from larrak2.cli.validate_simulation import main as validate_simulation_main
from larrak2.simulation_validation.restart_regression_artifacts import (
    extract_restart_run_artifacts,
)
from larrak2.simulation_validation.restart_regression_suite import (
    analyze_restart_regression_runs,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_runtime_summary(path: Path, *, rows: list[list[float]]) -> None:
    path.write_text(
        "\n".join(
            [
                "# tableQueryCells tableHitCells fallbackTimesteps interpolationCacheHits coverageRejectCells trustRegionRejectCells",
                *["\t".join(str(value) for value in row) for row in rows],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_engine_results(path: Path) -> None:
    _write_json(
        path,
        {
            "trace": [
                {
                    "time_s": 0.1,
                    "crank_angle_deg": -6.95,
                    "mean_pressure_Pa": 9.2e5,
                    "mean_temperature_K": 1200.0,
                    "apparent_heat_release_step_J": 0.1,
                },
                {
                    "time_s": 0.10001,
                    "crank_angle_deg": -6.9495,
                    "mean_pressure_Pa": 9.3e5,
                    "mean_temperature_K": 1210.0,
                    "apparent_heat_release_step_J": 0.2,
                },
                {
                    "time_s": 0.10002,
                    "crank_angle_deg": -6.9490,
                    "mean_pressure_Pa": 9.4e5,
                    "mean_temperature_K": 1220.0,
                    "apparent_heat_release_step_J": 0.05,
                },
            ]
        },
    )


def _profile_payload(
    *,
    benchmark_run_dir: Path,
    miss_path: Path,
    corpus_path: Path,
    corpus_npz_path: Path | None,
    table_hit_fraction: float,
    angle_advance_deg: float | None,
    total_numeric_hits: int,
    first_miss_class: str,
    first_miss_branch_id: str,
    first_offending_variables: list[str],
    max_out_of_bound_by_variable: dict[str, float],
    baseline_mean_pressure_pa: float,
) -> dict:
    return {
        "profile_name": "chem323_lookup_strict",
        "resolved_profile": "closed_valve_ignition_multitable_v1",
        "runtime_mode": "lookupTableStrict",
        "benchmark_run_dir": str(benchmark_run_dir),
        "solver_ok": False,
        "stage_result": "solver",
        "target_end_angle_deg": -6.9,
        "baseline_start": {
            "time_s": 0.1,
            "crank_angle_deg": -6.94906,
            "mean_pressure_Pa": baseline_mean_pressure_pa,
        },
        "latest_checkpoint": {}
        if angle_advance_deg is None
        else {"time_s": 0.10001, "crank_angle_deg": -6.94906 + angle_advance_deg},
        "wall_elapsed_s": 2.0,
        "wall_seconds_per_0p01deg": None if angle_advance_deg is None else 1.0,
        "angle_advance_deg": angle_advance_deg,
        "sim_time_advance_s": None if angle_advance_deg is None else 1.0e-5,
        "speed_score_deg_per_s": None if angle_advance_deg is None else angle_advance_deg / 2.0,
        "total_numeric_hits": total_numeric_hits,
        "clamp_score_hits_per_deg": None,
        "floor_hits_per_deg": None,
        "runtime_package_id": "chem323_reduced_v2512",
        "runtime_table_id": "chem323_engine_ignition_entry_v1",
        "runtime_table_hash": "table-hash",
        "runtime_table_dir": str(benchmark_run_dir / "chemistry"),
        "runtime_chemistry_authority_miss_path": str(miss_path),
        "runtime_coverage_corpus_path": str(corpus_path),
        "runtime_coverage_corpus_npz_path": "" if corpus_npz_path is None else str(corpus_npz_path),
        "numeric_stability": {},
        "runtime_chemistry": {
            "coverage_reject_cell_fraction": 0.0,
            "coverage_reject_cells": 0.0,
            "fallback_hit_fraction": None,
            "fallback_timesteps": 0.0,
            "interpolation_cache_hit_fraction": 0.25,
            "interpolation_cache_hits": 25.0,
            "runtime_coverage_corpus_path": str(corpus_path),
            "runtime_summary_count": 2.0,
            "table_hit_cells": 100.0,
            "table_hit_fraction": table_hit_fraction,
            "table_query_cells": 150.0,
            "trust_region_reject_cells": 1.0,
        },
        "chem323_maturity_gate_passed": table_hit_fraction >= 0.6,
        "chem323_runtime_replacement_gate_passed": False,
        "first_miss_class": first_miss_class,
        "first_miss_branch_id": first_miss_branch_id,
        "first_miss_sign_flip": False,
        "first_offending_variables": first_offending_variables,
        "max_out_of_bound_by_variable": max_out_of_bound_by_variable,
        "max_qdot_relative_error": 0.0,
        "max_qdot_transformed_envelope_excess": 0.0,
        "executed_stage_count": 1,
        "executed_stage_names": ["ignition_entry"],
        "executed_stage_runtime_tables": [
            {
                "stage_name": "ignition_entry",
                "runtime_table_id": "chem323_engine_ignition_entry_v1",
                "runtime_table_hash": "table-hash",
                "runtime_table_dir": str(benchmark_run_dir / "chemistry"),
                "target_end_angle_deg": -6.9,
                "latest_angle_deg": -6.94902 if angle_advance_deg is not None else -6.94906,
                "ok": False,
            }
        ],
    }


def _write_run_fixture(
    tmp_path: Path,
    *,
    run_name: str,
    table_hit_fraction: float,
    angle_advance_deg: float | None,
    total_numeric_hits: int,
    first_miss_class: str,
    first_offending_variables: list[str],
    max_out_of_bound_by_variable: dict[str, float],
    miss_payload: dict,
    runtime_rows: list[tuple[float, list[float]]],
    use_npz_corpus: bool,
    baseline_mean_pressure_pa: float = 9.2e5,
) -> Path:
    run_dir = tmp_path / run_name
    benchmark_run_dir = run_dir / "chem323_lookup_strict"
    chemistry_dir = benchmark_run_dir / "chemistry"
    chemistry_dir.mkdir(parents=True, exist_ok=True)

    miss_path = benchmark_run_dir / "runtimeChemistryAuthorityMiss.json"
    _write_json(miss_path, miss_payload)

    stage_manifest_path = benchmark_run_dir / "engine_stage_manifest.json"
    _write_json(
        stage_manifest_path,
        {
            "stages": [
                {
                    "name": "ignition_entry",
                    "end_angle_deg": -6.9,
                    "ok": False,
                    "stage_result": "solver",
                    "engine_min_pressure_Pa": 0.000287037037037037,
                }
            ]
        },
    )
    _write_engine_results(benchmark_run_dir / "engine_results.json")

    for time_s, row in runtime_rows:
        _write_runtime_summary(
            benchmark_run_dir / f"runtimeChemistrySummary.{time_s}.dat",
            rows=[row],
        )

    if use_npz_corpus:
        corpus_npz_path = benchmark_run_dir / "runtimeChemistryCoverageCorpus.npz"
        np.savez(
            corpus_npz_path,
            table_id=np.asarray("chem323_engine_ignition_entry_v1"),
            package_id=np.asarray("chem323_reduced_v2512"),
            runtime_mode=np.asarray("lookupTableStrict"),
            coverage_quantization=np.asarray(0.05),
            state_variables=np.asarray(["Temperature", "Pressure", "IC8H18"]),
            raw_states=np.asarray([[1308.2, 1.38531e6, 0.04], [1350.0, 3.9e6, 0.03]], dtype=float),
            transformed_states=np.asarray(
                [[1308.2, 6.14, 0.04], [1350.0, 6.59, 0.03]], dtype=float
            ),
            coverage_bucket_keys=np.asarray([[1, 2, 3], [2, 3, 4]], dtype=int),
            query_counts=np.asarray([2.0, 4.0], dtype=float),
            table_hit_counts=np.asarray([2.0, 3.0], dtype=float),
            coverage_reject_counts=np.asarray([0.0, 0.0], dtype=float),
            trust_reject_counts=np.asarray([0.0, 1.0], dtype=float),
            worst_reject_excess=np.asarray(
                [0.0, max(max_out_of_bound_by_variable.values())], dtype=float
            ),
            nearest_sample_distance_min=np.asarray([0.0, 0.01], dtype=float),
            nearest_sample_distance_max=np.asarray([0.0, 0.05], dtype=float),
            stage_names_json=np.asarray(['["ignition_entry"]', '["ignition_entry"]']),
        )
        corpus_path = benchmark_run_dir / "runtimeChemistryCoverageCorpus.json"
        _write_json(
            corpus_path,
            {
                "table_id": "chem323_engine_ignition_entry_v1",
                "package_id": "chem323_reduced_v2512",
                "runtime_mode": "lookupTableStrict",
                "state_variables": ["Temperature", "Pressure", "IC8H18"],
                "coverage_quantization": 0.05,
                "rows": [],
            },
        )
    else:
        corpus_npz_path = None
        corpus_path = benchmark_run_dir / "runtimeChemistryCoverageCorpus.json"
        _write_json(
            corpus_path,
            {
                "table_id": "chem323_engine_ignition_entry_v1",
                "package_id": "chem323_reduced_v2512",
                "runtime_mode": "lookupTableStrict",
                "state_variables": ["Temperature", "Pressure", "IC8H18"],
                "coverage_quantization": 0.05,
                "rows": [
                    {
                        "coverage_bucket_key": [1, 2, 3],
                        "raw_state": {
                            "Temperature": 1350.0,
                            "Pressure": 3.90866e6,
                            "IC8H18": 0.03,
                        },
                        "transformed_state": {
                            "Temperature": 1350.0,
                            "Pressure": 6.59,
                            "IC8H18": 0.03,
                        },
                        "query_count": 3,
                        "table_hit_count": 0,
                        "coverage_reject_count": 0,
                        "trust_reject_count": 1,
                        "worst_reject_variable": "CY3C5H8O_diag",
                        "worst_reject_excess": max(max_out_of_bound_by_variable.values()),
                        "nearest_sample_distance_min": 0.0,
                        "nearest_sample_distance_max": 0.05,
                        "stage_names": ["ignition_entry"],
                    }
                ],
            },
        )

    summary_payload = {
        "base_run_dir": str(run_dir / "base"),
        "checkpoint_angle_deg": -6.94906,
        "checkpoint_time_s": 0.1,
        "runtime_package_id": "chem323_reduced_v2512",
        "runtime_package_hash": "runtime-hash",
        "window_angle_deg": 2.5,
        "recommendation": {"fastest_profile": None, "lowest_clamp_profile": None},
        "profiles": [
            _profile_payload(
                benchmark_run_dir=benchmark_run_dir,
                miss_path=miss_path,
                corpus_path=corpus_path,
                corpus_npz_path=corpus_npz_path,
                table_hit_fraction=table_hit_fraction,
                angle_advance_deg=angle_advance_deg,
                total_numeric_hits=total_numeric_hits,
                first_miss_class=first_miss_class,
                first_miss_branch_id="default"
                if first_miss_class != "undetailed_authority_miss"
                else "",
                first_offending_variables=first_offending_variables,
                max_out_of_bound_by_variable=max_out_of_bound_by_variable,
                baseline_mean_pressure_pa=baseline_mean_pressure_pa,
            )
        ],
    }
    _write_json(run_dir / "engine_restart_benchmark_summary.json", summary_payload)
    return run_dir


def _make_recent_runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    v63 = _write_run_fixture(
        tmp_path,
        run_name="engine_restart_benchmark_live_parallel_v63_chem323_multitable_entry_refreshed",
        table_hit_fraction=0.625,
        angle_advance_deg=4.0e-05,
        total_numeric_hits=3477,
        first_miss_class="qdot",
        first_offending_variables=["Qdot"],
        max_out_of_bound_by_variable={"Qdot": 0.00176476},
        miss_payload={
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "failure_branch_id": "default",
            "first_offending_variables": ["Qdot"],
            "max_out_of_bound_by_variable": {"Qdot": 0.00176476},
            "reject_variable": "Qdot",
            "qdot_reject_variable": "Qdot",
            "reject_excess": 0.00176476,
            "qdot_transformed_excess": 0.00176476,
            "reject_nearest_sample_distance": 0.0137735,
            "qdot_nearest_sample_distance": 0.0137735,
            "table_id": "chem323_engine_ignition_entry_v1",
            "table_hash": "table-hash",
            "runtime_mode": "lookupTableStrict",
            "checkpoint_time_s": 0.1,
            "crank_angle_deg": -6.94902,
            "total_queried_cells": 2712,
            "trust_reject_cell_count": 1,
            "uncovered_cell_count": 0,
            "max_untracked_mass_fraction": 0.0,
            "miss_counts_by_variable": {"Qdot": 1},
            "crossed_local_stencil_sign": False,
            "sign_flip": False,
            "reject_state": {"Temperature": 1308.2, "Pressure": 1.38531e6},
        },
        runtime_rows=[
            (0.00028249794086964591, [1356, 1356, 0, 511, 0, 0]),
            (0.00028250201081343209, [2712, 1783, 0, 781, 0, 1]),
        ],
        use_npz_corpus=True,
    )
    v64 = _write_run_fixture(
        tmp_path,
        run_name="engine_restart_benchmark_live_parallel_v64_chem323_multitable_entry_qdot_refreshed",
        table_hit_fraction=0.657448377581121,
        angle_advance_deg=4.0e-05,
        total_numeric_hits=3477,
        first_miss_class="qdot",
        first_offending_variables=["Qdot"],
        max_out_of_bound_by_variable={"Qdot": 0.0435392},
        miss_payload={
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "failure_branch_id": "default",
            "first_offending_variables": ["Qdot"],
            "max_out_of_bound_by_variable": {"Qdot": 0.0435392},
            "reject_variable": "Qdot",
            "qdot_reject_variable": "Qdot",
            "reject_excess": 0.0435392,
            "qdot_transformed_excess": 0.0435392,
            "reject_nearest_sample_distance": 0.0505193,
            "qdot_nearest_sample_distance": 0.0505193,
            "table_id": "chem323_engine_ignition_entry_v1",
            "table_hash": "table-hash",
            "runtime_mode": "lookupTableStrict",
            "checkpoint_time_s": 0.1,
            "crank_angle_deg": -6.94902,
            "total_queried_cells": 2712,
            "trust_reject_cell_count": 1,
            "uncovered_cell_count": 0,
            "max_untracked_mass_fraction": 0.0,
            "miss_counts_by_variable": {"Qdot": 1},
            "crossed_local_stencil_sign": False,
            "sign_flip": False,
            "reject_state": {"Temperature": 1350.0, "Pressure": 3.90866e6},
        },
        runtime_rows=[
            (0.00028249794086964591, [1356, 1356, 0, 511, 0, 0]),
            (0.00028250201081343209, [2712, 1783, 0, 781, 0, 1]),
        ],
        use_npz_corpus=True,
    )
    v65 = _write_run_fixture(
        tmp_path,
        run_name="engine_restart_benchmark_live_parallel_v65_chem323_multitable_qdot_multitarget",
        table_hit_fraction=0.014749262536873156,
        angle_advance_deg=None,
        total_numeric_hits=0,
        first_miss_class="undetailed_authority_miss",
        first_offending_variables=["CY3C5H8O_diag"],
        max_out_of_bound_by_variable={"CY3C5H8O_diag": 158.821},
        miss_payload={
            "stage_name": "ignition_entry",
            "first_offending_variables": ["CY3C5H8O_diag"],
            "max_out_of_bound_by_variable": {"CY3C5H8O_diag": 158.821},
            "table_id": "chem323_engine_ignition_entry_v1",
            "table_hash": "table-hash",
            "runtime_mode": "lookupTableStrict",
            "checkpoint_time_s": 0.1,
            "crank_angle_deg": -6.94906,
            "total_queried_cells": 1356,
            "trust_reject_cell_count": 1,
            "uncovered_cell_count": 0,
            "max_untracked_mass_fraction": 0.0,
            "miss_counts_by_variable": {"CY3C5H8O_diag": 1},
        },
        runtime_rows=[
            (0.00028249794086964591, [1356, 20, 0, 6, 0, 1]),
        ],
        use_npz_corpus=False,
        baseline_mean_pressure_pa=9.42194e5,
    )
    return v63, v64, v65


def _make_same_area_runs(tmp_path: Path) -> tuple[Path, Path]:
    a = _write_run_fixture(
        tmp_path,
        run_name="engine_restart_benchmark_same_area_v1",
        table_hit_fraction=0.66,
        angle_advance_deg=5.0e-05,
        total_numeric_hits=3200,
        first_miss_class="qdot",
        first_offending_variables=["Qdot"],
        max_out_of_bound_by_variable={"Qdot": 0.01},
        miss_payload={
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "failure_branch_id": "default",
            "first_offending_variables": ["Qdot"],
            "max_out_of_bound_by_variable": {"Qdot": 0.01},
            "reject_variable": "Qdot",
            "miss_counts_by_variable": {"Qdot": 1},
        },
        runtime_rows=[
            (0.00028249794086964591, [1356, 1300, 0, 500, 0, 0]),
            (0.00028250201081343209, [2712, 1700, 0, 780, 0, 1]),
        ],
        use_npz_corpus=True,
    )
    b = _write_run_fixture(
        tmp_path,
        run_name="engine_restart_benchmark_same_area_v2",
        table_hit_fraction=0.63,
        angle_advance_deg=4.5e-05,
        total_numeric_hits=3100,
        first_miss_class="qdot",
        first_offending_variables=["Qdot"],
        max_out_of_bound_by_variable={"Qdot": 0.015},
        miss_payload={
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "failure_branch_id": "default",
            "first_offending_variables": ["Qdot"],
            "max_out_of_bound_by_variable": {"Qdot": 0.015},
            "reject_variable": "Qdot",
            "miss_counts_by_variable": {"Qdot": 1},
        },
        runtime_rows=[
            (0.00028249794086964591, [1356, 1290, 0, 500, 0, 0]),
            (0.00028250201081343209, [2712, 1650, 0, 770, 0, 1]),
        ],
        use_npz_corpus=True,
    )
    return a, b


def test_extract_restart_run_artifacts_flattens_deterministically(tmp_path: Path) -> None:
    v63, _, _ = _make_recent_runs(tmp_path)
    first = extract_restart_run_artifacts(run_dir=v63)
    second = extract_restart_run_artifacts(run_dir=v63)

    first_summary_labels = [
        item["metadata"]["semantic_label"]
        for item in first["scalar_slots"]
        if item["artifact_slot_id"] == "engine_restart_benchmark_summary"
    ]
    second_summary_labels = [
        item["metadata"]["semantic_label"]
        for item in second["scalar_slots"]
        if item["artifact_slot_id"] == "engine_restart_benchmark_summary"
    ]
    assert first_summary_labels == second_summary_labels
    assert any(
        item["artifact_slot_id"] == "runtimeChemistrySummary.dat" for item in first["dense_series"]
    )
    assert any(
        item["artifact_slot_id"] == "runtimeChemistryCoverageCorpus.npz.raw_states"
        for item in first["dense_series"]
    )


def test_extract_restart_run_artifacts_supports_json_coverage_corpus_fallback(
    tmp_path: Path,
) -> None:
    _, _, v65 = _make_recent_runs(tmp_path)
    extracted = extract_restart_run_artifacts(run_dir=v65)
    assert any(
        item["artifact_slot_id"] == "runtimeChemistryCoverageCorpus.json.raw_states"
        for item in extracted["dense_series"]
    )


def test_analyze_restart_regression_runs_detects_shifted_regression_and_clusters_focus(
    tmp_path: Path,
) -> None:
    v63, v64, v65 = _make_recent_runs(tmp_path)
    analysis = analyze_restart_regression_runs(
        run_dirs=[str(v63), str(v64), str(v65)],
        history_window=3,
    )

    assert analysis["general"]["latest_run_classification"] == "regressed"
    assert analysis["general"]["neighborhood_classification"] == "shifted_area"
    assert "CY3C5H8O_diag" in analysis["general"]["prioritized_focus"][0]["label"]
    assert analysis["general"]["focus_clusters"][0]["cluster_id"]
    assert analysis["general"]["ignored_context_slots_count"] > 0


def test_scalar_priority_downweights_context_outliers(tmp_path: Path) -> None:
    earlier, latest = _make_same_area_runs(tmp_path)
    payload = json.loads((latest / "engine_restart_benchmark_summary.json").read_text(encoding="utf-8"))
    payload["profiles"][0]["baseline_start"]["mean_pressure_Pa"] = 9.42194e5
    _write_json(latest / "engine_restart_benchmark_summary.json", payload)

    analysis = analyze_restart_regression_runs(
        run_dirs=[str(earlier), str(latest)],
        history_window=2,
    )
    top = analysis["scalars"]["top_leverage"][0]
    assert top["slot_family"] != "context"
    assert "baseline_start.mean_pressure_Pa" not in str(top["semantic_label"])


def test_dense_priority_prefers_runtime_summary_collapse(tmp_path: Path) -> None:
    _, v64, v65 = _make_recent_runs(tmp_path)
    analysis = analyze_restart_regression_runs(
        run_dirs=[str(v64), str(v65)],
        history_window=2,
    )
    top = analysis["dense"]["top_divergence"][0]
    assert top["artifact_slot_id"] == "runtimeChemistrySummary.dat"
    assert top["collapse_classification"] in {"early_collapse", "truncated", "hit_collapse"}
    assert top["priority_score"] > 0.0


def test_general_verdict_suppresses_secondary_improvements(tmp_path: Path) -> None:
    _, v64, v65 = _make_recent_runs(tmp_path)
    analysis = analyze_restart_regression_runs(
        run_dirs=[str(v64), str(v65)],
        history_window=2,
    )
    suppressed = analysis["general"]["suppressed_secondary_improvements"]
    assert any(item["metric"] == "total_numeric_hits" for item in suppressed)


def test_same_area_regression_classification(tmp_path: Path) -> None:
    a, b = _make_same_area_runs(tmp_path)
    analysis = analyze_restart_regression_runs(
        run_dirs=[str(a), str(b)],
        history_window=2,
    )
    assert analysis["general"]["neighborhood_classification"] == "same_area"


def test_schema_drift_slots_preserve_presence_masks(tmp_path: Path) -> None:
    v63, v64, _ = _make_recent_runs(tmp_path)
    payload = json.loads((v64 / "engine_restart_benchmark_summary.json").read_text(encoding="utf-8"))
    payload["profiles"][0]["experimental_probe_value"] = 42.0
    _write_json(v64 / "engine_restart_benchmark_summary.json", payload)

    analysis = analyze_restart_regression_runs(
        run_dirs=[str(v63), str(v64)],
        history_window=2,
    )
    assert any(
        item["presence_mask"] == [False, True]
        for item in analysis["slot_manifest"]["scalar_slots"]
        if item["artifact_slot_id"] == "engine_restart_benchmark_summary"
    )


def test_focus_clusters_collapse_duplicate_failure_slots(tmp_path: Path) -> None:
    _, v64, v65 = _make_recent_runs(tmp_path)
    analysis = analyze_restart_regression_runs(
        run_dirs=[str(v64), str(v65)],
        history_window=2,
    )
    primary_cluster = analysis["general"]["focus_clusters"][0]
    member_labels = [str(item["label"]) for item in primary_cluster["members"]]
    assert any("CY3C5H8O_diag" in label for label in member_labels)
    assert "authority break" in primary_cluster["label"]


def test_restart_regression_cli_supports_explicit_runs(tmp_path: Path) -> None:
    v63, v64, v65 = _make_recent_runs(tmp_path)
    outdir = tmp_path / "analysis_out"
    exit_code = validate_simulation_main(
        [
            "restart-regression-analysis",
            "--runs",
            str(v63),
            "--runs",
            str(v64),
            "--runs",
            str(v65),
            "--outdir",
            str(outdir),
        ]
    )
    assert exit_code == 0
    assert (outdir / "restart_regression_general.json").exists()
    assert (outdir / "restart_regression_scalars.csv").exists()
    assert (outdir / "restart_regression_dense.json").exists()
    assert (outdir / "restart_regression_slot_manifest.json").exists()


def test_restart_regression_cli_supports_glob_and_latest(tmp_path: Path) -> None:
    _make_recent_runs(tmp_path)
    outdir = tmp_path / "analysis_glob"
    exit_code = validate_simulation_main(
        [
            "restart-regression-analysis",
            "--glob",
            str(tmp_path / "engine_restart_benchmark_live_parallel_v*"),
            "--latest",
            "2",
            "--suite",
            "all",
            "--outdir",
            str(outdir),
        ]
    )
    assert exit_code == 0
    payload = json.loads((outdir / "restart_regression_general.json").read_text(encoding="utf-8"))
    assert payload["baseline_run_id"].endswith("_v64_chem323_multitable_entry_qdot_refreshed")
    assert payload["latest_run_id"].endswith("_v65_chem323_multitable_qdot_multitarget")
