from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from larrak2.simulation_validation.coverage_corpus_analysis import (
    analyze_coverage_corpus_vs_targets,
    summarize_authority_miss_cluster,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_coverage_corpus_vs_targets_distances_and_dedupe(tmp_path: Path) -> None:
    state = {
        "Temperature": 1000.0,
        "Pressure": 1.0e6,
        "O2": 0.05,
        "CO2": 0.02,
    }
    far_state = {
        "Temperature": 500.0,
        "Pressure": 5.0e5,
        "O2": 0.2,
        "CO2": 0.1,
    }
    table = {
        "runtime_chemistry_table": {
            "state_species": ["O2", "CO2"],
            "transformed_state_variables": ["Pressure", "O2", "CO2"],
            "state_transform_floors": {"Pressure": 100000.0, "O2": 1e-12, "CO2": 1e-12},
            "current_window_diag_stage_names": ["ignition_entry"],
            "current_window_qdot_stage_names": ["ignition_entry"],
            "current_window_diag_target_limit": 4,
            "current_window_qdot_target_limit": 4,
            "seed_species_miss_artifacts": ["miss_o2.json"],
            "seed_qdot_miss_artifacts": ["miss_qdot.json"],
            "coverage_corpora": ["corpus.json"],
        }
    }
    _write_json(tmp_path / "table.json", table)
    _write_json(
        tmp_path / "miss_o2.json",
        {
            "reject_variable": "O2",
            "failure_class": "out_of_bound",
            "stage_name": "ignition_entry",
            "reject_state": dict(state),
        },
    )
    _write_json(
        tmp_path / "miss_qdot.json",
        {
            "reject_variable": "Qdot",
            "stage_name": "ignition_entry",
            "reject_state": {
                "Temperature": 1100.0,
                "Pressure": 1.1e6,
                "O2": 0.06,
                "CO2": 0.03,
            },
        },
    )
    _write_json(
        tmp_path / "corpus.json",
        {
            "rows": [
                {"raw_state": dict(state), "query_count": 1, "stage_names": ["ignition_entry"]},
                {"raw_state": dict(state), "query_count": 1, "stage_names": ["ignition_entry"]},
                {"raw_state": dict(far_state), "query_count": 1, "stage_names": ["ignition_entry"]},
            ],
            "high_fidelity_rows": [],
        },
    )

    result = analyze_coverage_corpus_vs_targets(
        table_config_path=tmp_path / "table.json",
        repo_root=tmp_path,
        authority_miss_path=tmp_path / "miss_o2.json",
    )

    assert result["corpus"]["loaded_row_count"] == 3
    assert result["corpus"]["unique_point_key_count"] == 2
    assert result["corpus"]["point_key_dedupe_collapsed_rows"] == 1
    species = result["species_miss_targets"]
    assert len(species) == 1
    assert species[0]["min_transformed_distance_to_corpus"] == pytest.approx(0.0, abs=1e-9)
    qdot = result["qdot_miss_targets"]
    assert len(qdot) == 1
    assert qdot[0]["min_transformed_distance_to_corpus"] is not None
    assert qdot[0]["min_transformed_distance_to_corpus"] > 0.01
    assert result["authority_miss_sample"]["reject_variable"] == "O2"


def test_summarize_authority_miss_cluster(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_json(
        a,
        {
            "reject_variable": "H2O2",
            "failure_class": "out_of_bound",
            "stage_name": "ignition_entry",
        },
    )
    _write_json(
        b,
        {
            "reject_variable": "Qdot",
            "failure_class": "envelope",
            "stage_name": "ignition_entry",
        },
    )
    summary = summarize_authority_miss_cluster([str(a), str(b)], repo_root=tmp_path)
    assert summary["artifact_paths_resolved"] == 2
    assert summary["reject_variable_counts"]["H2O2"] == 1
    assert summary["reject_variable_counts"]["Qdot"] == 1


def test_coverage_corpus_analysis_cli_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {
        "Temperature": 1000.0,
        "Pressure": 1.0e6,
        "O2": 0.05,
        "CO2": 0.02,
    }
    table = {
        "runtime_chemistry_table": {
            "state_species": ["O2", "CO2"],
            "transformed_state_variables": ["Pressure", "O2", "CO2"],
            "state_transform_floors": {"Pressure": 100000.0, "O2": 1e-12, "CO2": 1e-12},
            "current_window_diag_stage_names": ["ignition_entry"],
            "current_window_qdot_stage_names": ["ignition_entry"],
            "current_window_diag_target_limit": 2,
            "current_window_qdot_target_limit": 2,
            "seed_species_miss_artifacts": [],
            "seed_qdot_miss_artifacts": [],
            "coverage_corpora": ["corpus.json"],
        }
    }
    _write_json(tmp_path / "table.json", table)
    _write_json(
        tmp_path / "corpus.json",
        {"rows": [{"raw_state": dict(state), "query_count": 0}], "high_fidelity_rows": []},
    )
    monkeypatch.chdir(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "larrak2.cli.validate_simulation",
            "coverage-corpus-analysis",
            "--config",
            "table.json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = json.loads(proc.stdout)
    assert out["corpus"]["loaded_row_count"] == 1
