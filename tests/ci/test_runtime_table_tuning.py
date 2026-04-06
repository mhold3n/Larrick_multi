from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.cli.validate_simulation import main as validate_simulation_main
from larrak2.simulation_validation.runtime_table_tuning import (
    MISS_FAMILY_QDOT,
    MISS_FAMILY_SPECIES_DIAG,
    apply_stage_local_runtime_table_frontier_rebalance,
    apply_stage_local_runtime_table_refresh,
    plan_stage_local_runtime_table_frontier_rebalance,
    plan_stage_local_runtime_table_refresh,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_stage_config(path: Path) -> None:
    _write_json(
        path,
        {
            "runtime_chemistry_table": {
                "state_species": ["IC8H18", "O2", "CO2", "H2O", "H2O2"],
                "transformed_state_variables": ["Pressure", "H2O2"],
                "seed_species_miss_artifacts": ["existing_species.json"],
                "seed_qdot_miss_artifacts": ["existing_qdot.json"],
                "coverage_corpora": ["existing_corpus.json"],
                "current_window_diag_target_limit": 4,
                "current_window_qdot_target_limit": 2,
            }
        },
    )


def _write_strategy(path: Path, *, entry_config: Path, ramp_config: Path) -> None:
    _write_json(
        path,
        {
            "runtime_package": {
                "package_dir": "mechanisms/openfoam/v2512/chem323_reduced",
                "stage_runtime_tables": {
                    "ignition_entry": {
                        "runtime_table_dir": "mechanisms/openfoam/v2512/runtime_tables/chem323_engine_ignition_entry_v1",
                        "runtime_table_config_path": str(entry_config),
                    },
                    "ignition_ramp": {
                        "runtime_table_dir": "mechanisms/openfoam/v2512/runtime_tables/chem323_engine_ignition_ramp_v1",
                        "runtime_table_config_path": str(ramp_config),
                    },
                },
            }
        },
    )


def _write_latest_run(
    tmp_path: Path,
    *,
    miss_payload: dict,
    coverage_exists: bool,
) -> Path:
    run_dir = tmp_path / "latest_run"
    benchmark_run_dir = run_dir / "chem323_lookup_strict"
    benchmark_run_dir.mkdir(parents=True, exist_ok=True)
    miss_path = benchmark_run_dir / "runtimeChemistryAuthorityMiss.json"
    _write_json(miss_path, miss_payload)
    if coverage_exists:
        _write_json(
            benchmark_run_dir / "runtimeChemistryCoverageCorpus.json",
            {"rows": [], "state_variables": ["Temperature", "Pressure"]},
        )
    _write_json(
        run_dir / "engine_restart_benchmark_summary.json",
        {
            "profiles": [
                {
                    "profile_name": "chem323_lookup_strict",
                    "benchmark_run_dir": str(benchmark_run_dir),
                    "runtime_chemistry_authority_miss_path": str(miss_path),
                    "runtime_coverage_corpus_path": "",
                    "runtime_coverage_corpus_npz_path": "",
                }
            ]
        },
    )
    return run_dir


def test_cli_exposes_runtime_chemistry_table_and_restart_benchmark_help() -> None:
    for command in ("runtime-chemistry-table", "engine-restart-benchmark"):
        with pytest.raises(SystemExit) as exc:
            validate_simulation_main([command, "--help"])
        assert exc.value.code == 0


def test_plan_marks_sparse_diag_family_miss_as_non_seedable_and_skips_config_edit(
    tmp_path: Path,
) -> None:
    entry_config = tmp_path / "entry.json"
    ramp_config = tmp_path / "ramp.json"
    _write_stage_config(entry_config)
    _write_stage_config(ramp_config)
    strategy_path = tmp_path / "strategy.json"
    _write_strategy(strategy_path, entry_config=entry_config, ramp_config=ramp_config)
    latest_run = _write_latest_run(
        tmp_path,
        miss_payload={
            "stage_name": "ignition_entry",
            "first_offending_variables": ["CY3C5H8O_diag"],
            "max_out_of_bound_by_variable": {"CY3C5H8O_diag": 158.821},
        },
        coverage_exists=False,
    )
    analysis_payload = {
        "general": {
            "focus_clusters": [
                {
                    "kind": "failure_cluster",
                    "stage_name": "ignition_entry",
                    "top_variable": "CY3C5H8O_diag",
                    "failure_class": "undetailed_authority_miss",
                },
                {
                    "kind": "operational_cluster",
                    "stage_name": "ignition_entry",
                    "top_variable": "tableHitCells",
                    "failure_class": "early_collapse",
                },
            ]
        }
    }

    refresh_plan = plan_stage_local_runtime_table_refresh(
        run_dirs=[latest_run],
        strategy_config_path=str(strategy_path),
        analysis_payload=analysis_payload,
    )

    assert refresh_plan["stage_name"] == "ignition_entry"
    assert refresh_plan["miss_family"] == MISS_FAMILY_SPECIES_DIAG
    assert refresh_plan["target_list_name"] == "seed_species_miss_artifacts"
    assert refresh_plan["seedable"] is False
    assert refresh_plan["seedability_reason"] == "missing_reject_state"
    assert refresh_plan["coverage_corpus_exists"] is False

    result = apply_stage_local_runtime_table_refresh(
        refresh_plan=refresh_plan,
        repo_root=tmp_path,
    )

    updated = json.loads(entry_config.read_text(encoding="utf-8"))["runtime_chemistry_table"]
    assert result["seedable"] is False
    assert result["skipped"] is True
    assert result["appended_miss_artifact"] is None
    assert result["appended_coverage_corpus"] is None
    assert updated["seed_species_miss_artifacts"] == ["existing_species.json"]
    assert updated["coverage_corpora"] == ["existing_corpus.json"]
    untouched = json.loads(ramp_config.read_text(encoding="utf-8"))["runtime_chemistry_table"]
    assert untouched["seed_species_miss_artifacts"] == ["existing_species.json"]


def test_plan_and_apply_routes_qdot_family_and_appends_existing_corpus(tmp_path: Path) -> None:
    entry_config = tmp_path / "entry.json"
    ramp_config = tmp_path / "ramp.json"
    _write_stage_config(entry_config)
    _write_stage_config(ramp_config)
    strategy_path = tmp_path / "strategy.json"
    _write_strategy(strategy_path, entry_config=entry_config, ramp_config=ramp_config)
    latest_run = _write_latest_run(
        tmp_path,
        miss_payload={
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "reject_variable": "Qdot",
            "reject_state": {"Temperature": 1308.2, "Pressure": 1.38531e6},
            "first_offending_variables": ["Qdot"],
            "max_out_of_bound_by_variable": {"Qdot": 0.04},
        },
        coverage_exists=True,
    )
    analysis_payload = {
        "general": {
            "focus_clusters": [
                {
                    "kind": "failure_cluster",
                    "stage_name": "ignition_entry",
                    "top_variable": "Qdot",
                    "failure_class": "qdot",
                }
            ]
        }
    }

    refresh_plan = plan_stage_local_runtime_table_refresh(
        run_dirs=[latest_run],
        strategy_config_path=str(strategy_path),
        analysis_payload=analysis_payload,
    )

    assert refresh_plan["miss_family"] == MISS_FAMILY_QDOT
    assert refresh_plan["target_list_name"] == "seed_qdot_miss_artifacts"
    assert refresh_plan["seedable"] is True
    assert refresh_plan["seedability_reason"] == "seedable_qdot_reject_state"
    assert refresh_plan["coverage_corpus_exists"] is True

    result = apply_stage_local_runtime_table_refresh(
        refresh_plan=refresh_plan,
        repo_root=tmp_path,
    )

    updated = json.loads(entry_config.read_text(encoding="utf-8"))["runtime_chemistry_table"]
    assert result["seedable"] is True
    assert result["skipped"] is False
    assert updated["seed_qdot_miss_artifacts"][-1] == (
        "latest_run/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json"
    )
    assert result["appended_coverage_corpus"] == (
        "latest_run/chem323_lookup_strict/runtimeChemistryCoverageCorpus.json"
    )


def test_plan_prefers_json_coverage_corpus_path_when_json_and_npz_both_exist(tmp_path: Path) -> None:
    entry_config = tmp_path / "entry.json"
    ramp_config = tmp_path / "ramp.json"
    _write_stage_config(entry_config)
    _write_stage_config(ramp_config)
    strategy_path = tmp_path / "strategy.json"
    _write_strategy(strategy_path, entry_config=entry_config, ramp_config=ramp_config)
    latest_run = _write_latest_run(
        tmp_path,
        miss_payload={
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "reject_variable": "Qdot",
            "reject_state": {"Temperature": 1308.2, "Pressure": 1.38531e6},
            "first_offending_variables": ["Qdot"],
            "max_out_of_bound_by_variable": {"Qdot": 0.04},
        },
        coverage_exists=True,
    )
    npz_path = (
        latest_run
        / "chem323_lookup_strict"
        / "runtimeChemistryCoverageCorpus.npz"
    )
    npz_path.write_bytes(b"fake")

    refresh_plan = plan_stage_local_runtime_table_refresh(
        run_dirs=[latest_run],
        strategy_config_path=str(strategy_path),
        analysis_payload={
            "general": {
                "focus_clusters": [
                    {
                        "kind": "failure_cluster",
                        "stage_name": "ignition_entry",
                        "top_variable": "Qdot",
                        "failure_class": "qdot",
                    }
                ]
            }
        },
    )

    assert refresh_plan["coverage_corpus_exists"] is True
    assert str(refresh_plan["coverage_corpus_path"]).endswith(
        "runtimeChemistryCoverageCorpus.json"
    )


def test_plan_marks_tracked_species_same_sign_overshoot_as_seedable_species_family(
    tmp_path: Path,
) -> None:
    entry_config = tmp_path / "entry.json"
    ramp_config = tmp_path / "ramp.json"
    _write_stage_config(entry_config)
    _write_stage_config(ramp_config)
    strategy_path = tmp_path / "strategy.json"
    _write_strategy(strategy_path, entry_config=entry_config, ramp_config=ramp_config)
    latest_run = _write_latest_run(
        tmp_path,
        miss_payload={
            "stage_name": "ignition_entry",
            "failure_class": "same_sign_overshoot",
            "reject_variable": "H2O2",
            "reject_state": {"Temperature": 1120.0, "Pressure": 8.0e5, "H2O2": 1.0e-10},
            "first_offending_variables": ["H2O2"],
            "max_out_of_bound_by_variable": {"H2O2": 4.2e-4},
        },
        coverage_exists=False,
    )
    analysis_payload = {
        "general": {
            "focus_clusters": [
                {
                    "kind": "failure_cluster",
                    "stage_name": "ignition_entry",
                    "top_variable": "H2O2",
                    "failure_class": "same_sign_overshoot",
                }
            ]
        }
    }

    refresh_plan = plan_stage_local_runtime_table_refresh(
        run_dirs=[latest_run],
        strategy_config_path=str(strategy_path),
        analysis_payload=analysis_payload,
    )

    assert refresh_plan["miss_family"] == MISS_FAMILY_SPECIES_DIAG
    assert refresh_plan["target_list_name"] == "seed_species_miss_artifacts"
    assert refresh_plan["seedable"] is True
    assert refresh_plan["seedability_reason"] == "seedable_species_state_reject_state"


def test_plan_accepts_bare_solver_species_same_sign_overshoot_as_seedable_species_family(
    tmp_path: Path,
) -> None:
    entry_config = tmp_path / "entry.json"
    ramp_config = tmp_path / "ramp.json"
    _write_stage_config(entry_config)
    _write_stage_config(ramp_config)
    strategy_path = tmp_path / "strategy.json"
    _write_strategy(strategy_path, entry_config=entry_config, ramp_config=ramp_config)
    latest_run = _write_latest_run(
        tmp_path,
        miss_payload={
            "stage_name": "ignition_entry",
            "failure_class": "same_sign_overshoot",
            "reject_variable": "CH2OH",
            "reject_state": {"Temperature": 1120.0, "Pressure": 8.0e5},
            "first_offending_variables": ["CH2OH"],
            "max_out_of_bound_by_variable": {"CH2OH": 1.0e-8},
        },
        coverage_exists=False,
    )
    analysis_payload = {
        "general": {
            "focus_clusters": [
                {
                    "kind": "failure_cluster",
                    "stage_name": "ignition_entry",
                    "top_variable": "CH2OH",
                    "failure_class": "same_sign_overshoot",
                }
            ]
        }
    }

    refresh_plan = plan_stage_local_runtime_table_refresh(
        run_dirs=[latest_run],
        strategy_config_path=str(strategy_path),
        analysis_payload=analysis_payload,
    )

    assert refresh_plan["miss_family"] == MISS_FAMILY_SPECIES_DIAG
    assert refresh_plan["seedable"] is True
    assert refresh_plan["seedability_reason"] == "seedable_species_output_reject_state"


def test_plan_resolves_repo_relative_stage_config_paths_from_strategy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    entry_config = tmp_path / "configs" / "entry.json"
    ramp_config = tmp_path / "configs" / "ramp.json"
    _write_stage_config(entry_config)
    _write_stage_config(ramp_config)
    strategy_dir = tmp_path / "strategy"
    strategy_dir.mkdir()
    strategy_path = strategy_dir / "strategy.json"
    _write_json(
        strategy_path,
        {
            "runtime_package": {
                "stage_runtime_tables": {
                    "ignition_entry": {
                        "runtime_table_dir": "mechanisms/openfoam/v2512/runtime_tables/chem323_engine_ignition_entry_v1",
                        "runtime_table_config_path": "configs/entry.json",
                    },
                    "ignition_ramp": {
                        "runtime_table_dir": "mechanisms/openfoam/v2512/runtime_tables/chem323_engine_ignition_ramp_v1",
                        "runtime_table_config_path": "configs/ramp.json",
                    },
                }
            }
        },
    )
    latest_run = _write_latest_run(
        tmp_path,
        miss_payload={
            "stage_name": "ignition_entry",
            "first_offending_variables": ["CY3C5H8O_diag"],
            "max_out_of_bound_by_variable": {"CY3C5H8O_diag": 2.0},
        },
        coverage_exists=False,
    )
    analysis_payload = {
        "general": {
            "focus_clusters": [
                {
                    "kind": "failure_cluster",
                    "stage_name": "ignition_entry",
                    "top_variable": "CY3C5H8O_diag",
                    "failure_class": "undetailed_authority_miss",
                }
            ]
        }
    }

    refresh_plan = plan_stage_local_runtime_table_refresh(
        run_dirs=[latest_run],
        strategy_config_path=str(strategy_path),
        analysis_payload=analysis_payload,
    )

    assert Path(refresh_plan["stage_config_path"]).resolve() == entry_config.resolve()


def test_frontier_rebalance_selects_exact_species_and_qdot_frontiers(
    tmp_path: Path,
) -> None:
    entry_config = tmp_path / "entry.json"
    ramp_config = tmp_path / "ramp.json"
    _write_stage_config(entry_config)
    _write_stage_config(ramp_config)

    def _reject_state(*, temperature: float, pressure: float, h2o2: float) -> dict[str, float]:
        return {
            "Temperature": temperature,
            "Pressure": pressure,
            "IC8H18": 0.042,
            "O2": 0.219,
            "CO2": 0.013,
            "H2O": 0.0029,
            "H2O2": h2o2,
        }

    def _artifact(relpath: str, payload: dict) -> str:
        path = tmp_path / relpath
        _write_json(path, payload)
        return str(path.relative_to(tmp_path))

    species_v62 = _artifact(
        "runs/v62/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "same_sign_overshoot",
            "reject_variable": "OLD_diag",
            "reject_state": _reject_state(temperature=650.0, pressure=7.5e5, h2o2=1.0e-12),
        },
    )
    species_v66d = _artifact(
        "runs/v66d/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "same_sign_overshoot",
            "reject_variable": "C5H82OOH4-5_diag",
            "reject_state": _reject_state(temperature=830.44, pressure=3.0e4, h2o2=1.7e-16),
        },
    )
    species_v66g = _artifact(
        "runs/v66g/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "same_sign_overshoot",
            "reject_variable": "H2O2",
            "reject_state": _reject_state(temperature=1120.08, pressure=8.08698e5, h2o2=3.0e-16),
        },
    )
    species_v66h = _artifact(
        "runs/v66h/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "same_sign_overshoot",
            "reject_variable": "C5H82OOH4-5_diag",
            "reject_state": _reject_state(temperature=830.44, pressure=3.0e4, h2o2=1.71e-16),
        },
    )
    species_v66i = _artifact(
        "runs/v66i/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "same_sign_overshoot",
            "reject_variable": "CY3C5H8O_diag",
            "reject_state": _reject_state(temperature=1350.0, pressure=1.5216e6, h2o2=2.48e-15),
        },
    )
    qdot_v63 = _artifact(
        "runs/v63/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "reject_variable": "Qdot",
            "reject_state": _reject_state(temperature=1228.27, pressure=1.3761e6, h2o2=3.1e-16),
        },
    )
    qdot_v64 = _artifact(
        "runs/v64/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "reject_variable": "Qdot",
            "reject_state": _reject_state(temperature=1275.0, pressure=2.2e6, h2o2=3.1e-16),
        },
    )
    qdot_v66a = _artifact(
        "runs/v66a/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "reject_variable": "Qdot",
        },
    )
    qdot_v66c = _artifact(
        "runs/v66c/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "reject_variable": "Qdot",
            "reject_state": _reject_state(temperature=1287.25, pressure=2.39583e6, h2o2=3.1e-16),
        },
    )
    qdot_v66j = _artifact(
        "runs/v66j/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "reject_variable": "Qdot",
            "reject_state": _reject_state(temperature=1228.27, pressure=1.3761e6, h2o2=3.1e-16),
        },
    )
    qdot_v66l = _artifact(
        "runs/v66l/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "reject_variable": "Qdot",
            "reject_state": _reject_state(temperature=953.034, pressure=3.02072e5, h2o2=3.1e-16),
        },
    )
    qdot_v66o = _artifact(
        "runs/v66o/runtimeChemistryAuthorityMiss.json",
        {
            "stage_name": "ignition_entry",
            "failure_class": "qdot",
            "reject_variable": "Qdot",
            "reject_state": _reject_state(temperature=812.226, pressure=1.4786e5, h2o2=3.1e-16),
        },
    )

    _write_json(
        entry_config,
        {
            "runtime_chemistry_table": {
                "state_species": ["IC8H18", "O2", "CO2", "H2O", "H2O2"],
                "transformed_state_variables": ["Pressure", "H2O2"],
                "seed_species_miss_artifacts": [
                    species_v62,
                    species_v66d,
                    species_v66g,
                    species_v66h,
                    species_v66i,
                ],
                "seed_qdot_miss_artifacts": [
                    qdot_v63,
                    qdot_v64,
                    qdot_v66a,
                    qdot_v66c,
                    qdot_v66j,
                    qdot_v66l,
                ],
                "coverage_corpora": ["existing_corpus.json"],
                "current_window_diag_target_limit": 4,
                "current_window_qdot_target_limit": 4,
            }
        },
    )
    strategy_path = tmp_path / "strategy.json"
    _write_strategy(strategy_path, entry_config=entry_config, ramp_config=ramp_config)

    latest_run = _write_latest_run(
        tmp_path,
        miss_payload=json.loads((tmp_path / qdot_v66o).read_text(encoding="utf-8")),
        coverage_exists=False,
    )
    analysis_payload = {
        "general": {
            "focus_clusters": [
                {
                    "kind": "failure_cluster",
                    "stage_name": "ignition_entry",
                    "top_variable": "Qdot",
                    "failure_class": "qdot",
                }
            ]
        }
    }

    rebalance_plan = plan_stage_local_runtime_table_frontier_rebalance(
        run_dirs=[latest_run],
        strategy_config_path=str(strategy_path),
        analysis_payload=analysis_payload,
    )

    assert rebalance_plan["selection_reason"] == "bounded_frontier_rebalance"
    assert rebalance_plan["active_species_frontier"] == [
        str((tmp_path / species_v66d).resolve()),
        str((tmp_path / species_v66g).resolve()),
        str((tmp_path / species_v66h).resolve()),
        str((tmp_path / species_v66i).resolve()),
    ]
    assert rebalance_plan["active_qdot_frontier"] == [
        str((tmp_path / qdot_v66c).resolve()),
        str((tmp_path / qdot_v66j).resolve()),
        str((tmp_path / qdot_v66l).resolve()),
        str((tmp_path / "latest_run/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json").resolve()),
    ]
    assert {"path": str((tmp_path / species_v62).resolve()), "reason": "older_than_active_species_frontier"} in rebalance_plan["pruned_species_artifacts"]
    assert {"path": str((tmp_path / qdot_v66a).resolve()), "reason": "missing_reject_state"} in rebalance_plan["pruned_qdot_artifacts"]
    assert {"path": str((tmp_path / qdot_v63).resolve()), "reason": "superseded_duplicate_signature"} in rebalance_plan["pruned_qdot_artifacts"]
    assert {"path": str((tmp_path / qdot_v64).resolve()), "reason": "not_selected_for_active_qdot_frontier"} in rebalance_plan["pruned_qdot_artifacts"]

    result = apply_stage_local_runtime_table_frontier_rebalance(
        rebalance_plan=rebalance_plan,
        repo_root=tmp_path,
    )

    updated = json.loads(entry_config.read_text(encoding="utf-8"))["runtime_chemistry_table"]
    assert result["active_species_frontier"] == [species_v66d, species_v66g, species_v66h, species_v66i]
    assert result["active_qdot_frontier"] == [
        qdot_v66c,
        qdot_v66j,
        qdot_v66l,
        "latest_run/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json",
    ]
    assert updated["seed_species_miss_artifacts"] == [species_v66d, species_v66g, species_v66h, species_v66i]
    assert updated["seed_qdot_miss_artifacts"] == [
        qdot_v66c,
        qdot_v66j,
        qdot_v66l,
        "latest_run/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json",
    ]
