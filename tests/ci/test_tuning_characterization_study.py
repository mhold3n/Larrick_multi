from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from larrak2.cli.validate_simulation import main as validate_simulation_main
from larrak2.simulation_validation.tuning_characterization_study import (
    ExpectedImprovementGPSearchStrategy,
    NSGAIISurrogateSearchStrategy,
    RandomSearchStrategy,
    append_experiments_jsonl,
    apply_knobs_to_table_config,
    build_tuning_manifest,
    compute_objective_vector,
    ingest_benchmark_run_directory,
    load_experiments_jsonl,
    load_knob_schema,
    load_tuning_manifest,
    run_tuning_batch,
    sha256_file,
    write_tuning_manifest,
)


def _minimal_summary(
    tmp_path: Path,
    *,
    trust_cells: float = 0.0,
    numeric_hits: float = 50.0,
    gate: bool = True,
    miss_excess: float = 0.0,
    miss_var: str = "",
) -> Path:
    bench = tmp_path / "bench_case"
    bench.mkdir()
    authority = bench / "runtimeChemistryAuthorityMiss.json"
    authority.write_text(
        json.dumps(
            {
                "reject_variable": miss_var,
                "reject_excess": miss_excess,
                "failure_class": "same_sign_overshoot",
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "bench_out"
    root.mkdir()
    summary = {
        "profiles": [
            {
                "profile_name": "chem323_lookup_strict",
                "benchmark_run_dir": str(bench.resolve()),
                "trust_region_reject_cells": trust_cells,
                "total_numeric_hits": numeric_hits,
                "chem323_runtime_replacement_gate_passed": gate,
                "runtime_chemistry_authority_miss_path": str(authority.resolve()),
            }
        ]
    }
    (root / "engine_restart_benchmark_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return root


def test_apply_knobs_to_table_config_preserves_comment_and_sets_nested(tmp_path: Path) -> None:
    schema_path = Path(
        "data/simulation_validation/tuning_knob_schema_chem323_ignition_entry_v1.json"
    )
    schema = load_knob_schema(schema_path)
    cfg = tmp_path / "table.json"
    cfg.write_text(
        json.dumps(
            {
                "_comment": "keep",
                "runtime_chemistry_table": {
                    "adaptive_sampling": {"rbf_envelope_scale": 1.0, "rbf_neighbor_count": 10},
                    "table_id": "t",
                },
            }
        ),
        encoding="utf-8",
    )
    out = apply_knobs_to_table_config(
        table_config_path=cfg,
        knob_schema=schema,
        knobs={
            "rbf_envelope_scale": 1.2,
            "rbf_neighbor_count": 16,
            "rbf_epsilon": 1.5,
            "rbf_diag_envelope_scale_ho2": 1.1,
            "max_samples": 500,
        },
    )
    assert out["_comment"] == "keep"
    assert out["runtime_chemistry_table"]["adaptive_sampling"]["rbf_envelope_scale"] == 1.2
    assert out["runtime_chemistry_table"]["adaptive_sampling"]["rbf_neighbor_count"] == 16
    assert out["runtime_chemistry_table"]["rbf_diag_envelope_scale_ho2"] == 1.1


def test_tuning_manifest_round_trip(tmp_path: Path) -> None:
    m = build_tuning_manifest(
        knob_schema_id="s1",
        knobs={"a": 1.0},
        input_artifact_hashes={"table_config_sha256": "abc"},
        experiment_id="e1",
        staged_table_config_path="/tmp/x.json",
    )
    p = tmp_path / "tuning_experiment_manifest.json"
    write_tuning_manifest(p, m)
    loaded = load_tuning_manifest(p.parent)
    assert loaded is not None
    assert loaded["knob_schema_id"] == "s1"
    assert loaded["knobs"]["a"] == 1.0


def test_compute_objective_vector_penalized_scalar() -> None:
    artifacts = {
        "run_dir": "/tmp/r",
        "selected_profile": {
            "trust_region_reject_cells": 2.0,
            "total_numeric_hits": 1000.0,
            "chem323_runtime_replacement_gate_passed": False,
        },
        "authority_miss_payload": {"reject_excess": 0.5, "reject_variable": "HO2_diag"},
    }
    obj = compute_objective_vector(
        artifacts, lambda_numeric=1e-3, mu_gate=5.0, mu_reject_excess=2.0
    )
    assert obj["trust_region_reject_cells"] == 2.0
    assert obj["chem323_runtime_replacement_gate_passed"] == 0.0
    assert obj["penalized_scalar"] > 2.0


def test_ingest_benchmark_run_directory_with_manifest(tmp_path: Path) -> None:
    root = _minimal_summary(tmp_path, trust_cells=1.0, gate=False, miss_excess=0.1, miss_var="X")
    manifest = build_tuning_manifest(
        knob_schema_id="chem323_ignition_entry_rbf_v1",
        knobs={"rbf_envelope_scale": 1.1},
        input_artifact_hashes={},
        experiment_id="x",
    )
    write_tuning_manifest(root / "tuning_experiment_manifest.json", manifest)
    rec = ingest_benchmark_run_directory(root)
    assert rec["knobs_trusted"] is True
    assert rec["knobs"]["rbf_envelope_scale"] == 1.1
    assert rec["objectives"]["trust_region_reject_cells"] == 1.0


def test_append_and_load_experiments_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    append_experiments_jsonl(p, [{"a": 1}])
    append_experiments_jsonl(p, [{"b": 2}])
    rows = load_experiments_jsonl(p)
    assert len(rows) == 2


def test_gp_ei_proposes_after_two_trusted_points(tmp_path: Path) -> None:
    schema_path = Path(
        "data/simulation_validation/tuning_knob_schema_chem323_ignition_entry_v1.json"
    )
    schema = load_knob_schema(schema_path)
    names = [str(k["name"]) for k in schema["knobs"]]

    def full_knobs(
        rbf_env: float, rbf_n: int, eps: float, ho2: float, ms: int
    ) -> dict[str, float | int]:
        return {
            "rbf_envelope_scale": rbf_env,
            "rbf_neighbor_count": rbf_n,
            "rbf_epsilon": eps,
            "rbf_diag_envelope_scale_ho2": ho2,
            "max_samples": ms,
        }

    experiments = [
        {
            "knobs_trusted": True,
            "knobs": full_knobs(1.1, 10, 1.0, 1.0, 400),
            "objectives": {"penalized_scalar": 5.0},
        },
        {
            "knobs_trusted": True,
            "knobs": full_knobs(1.2, 12, 1.0, 1.0, 400),
            "objectives": {"penalized_scalar": 3.0},
        },
    ]
    rng = np.random.default_rng(0)
    strat = ExpectedImprovementGPSearchStrategy()
    proposals = strat.propose(knob_schema=schema, experiments=experiments, n=2, rng=rng)
    assert len(proposals) == 2
    assert "knobs" in proposals[0]
    assert set(proposals[0]["knobs"]) == set(names)


def test_nsga2_surrogate_proposes_with_three_trusted_points() -> None:
    schema_path = Path("data/simulation_validation/tuning_knob_schema_chem323_ignition_entry_v1.json")
    schema = load_knob_schema(schema_path)

    def k(rbf_env: float, n: int) -> dict[str, float | int]:
        return {
            "rbf_envelope_scale": rbf_env,
            "rbf_neighbor_count": n,
            "rbf_epsilon": 1.0,
            "rbf_diag_envelope_scale_ho2": 1.0,
            "max_samples": 500,
        }

    experiments = [
        {
            "knobs_trusted": True,
            "knobs": k(1.05, 10),
            "objectives": {"trust_region_reject_cells": 2.0, "total_numeric_hits": 100.0},
        },
        {
            "knobs_trusted": True,
            "knobs": k(1.1, 12),
            "objectives": {"trust_region_reject_cells": 1.0, "total_numeric_hits": 200.0},
        },
        {
            "knobs_trusted": True,
            "knobs": k(1.15, 14),
            "objectives": {"trust_region_reject_cells": 0.0, "total_numeric_hits": 300.0},
        },
    ]
    rng = np.random.default_rng(3)
    prop = NSGAIISurrogateSearchStrategy(pop_size=8, n_gen=5).propose(
        knob_schema=schema, experiments=experiments, n=2, rng=rng
    )
    assert len(prop) == 2


def test_random_search_proposes(tmp_path: Path) -> None:
    schema_path = Path(
        "data/simulation_validation/tuning_knob_schema_chem323_ignition_entry_v1.json"
    )
    schema = load_knob_schema(schema_path)
    rng = np.random.default_rng(1)
    out = RandomSearchStrategy().propose(knob_schema=schema, experiments=[], n=3, rng=rng)
    assert len(out) == 3


def test_run_tuning_batch_dry_run(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "schema_id": "test_v1",
                "config_root_key": "runtime_chemistry_table",
                "knobs": [
                    {
                        "name": "rbf_envelope_scale",
                        "path": "adaptive_sampling.rbf_envelope_scale",
                        "type": "float",
                        "low": 1.0,
                        "high": 1.2,
                        "scale": "linear",
                        "group": "rbf",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "runtime_chemistry_table": {
                    "adaptive_sampling": {"rbf_envelope_scale": 1.05},
                    "table_id": "t",
                }
            }
        ),
        encoding="utf-8",
    )
    exp_path = tmp_path / "exp.jsonl"
    summary = run_tuning_batch(
        knob_schema_path=schema,
        base_table_config_path=base,
        strategy_config_path=None,
        experiments_jsonl_path=exp_path,
        study_outdir=tmp_path / "study",
        search_strategy="random",
        n_trials=2,
        refresh_table=False,
        run_benchmark=False,
        benchmark_run_dir=None,
        tuned_params_path=None,
        handoff_artifact_path=None,
        runtime_strategy_config=None,
        benchmark_profiles=None,
        window_angle_deg=0.01,
        docker_timeout_s=60,
        refresh_custom_solver=False,
        max_benchmarks=0,
        dry_run=True,
        profile_name=None,
        rng_seed=2,
    )
    assert summary["proposals"] == 2
    assert not exp_path.exists()


def test_cli_tuning_characterization_propose_smoke(tmp_path: Path, monkeypatch) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "schema_id": "cli_test",
                "config_root_key": "runtime_chemistry_table",
                "knobs": [
                    {
                        "name": "x",
                        "path": "adaptive_sampling.rbf_envelope_scale",
                        "type": "float",
                        "low": 1.0,
                        "high": 1.1,
                        "scale": "linear",
                        "group": "rbf",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    rc = validate_simulation_main(
        [
            "tuning-characterization",
            "--mode",
            "propose",
            "--knob-schema",
            str(schema),
            "--experiments-jsonl",
            str(tmp_path / "missing.jsonl"),
            "--n-proposals",
            "1",
            "--strategy",
            "random",
        ]
    )
    assert rc == 0


def test_cli_tuning_characterization_ingest_requires_glob_or_runs() -> None:
    rc = validate_simulation_main(
        [
            "tuning-characterization",
            "--mode",
            "ingest",
            "--experiments-jsonl",
            "/tmp/x.jsonl",
        ]
    )
    assert rc == 2


def test_sha256_file_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    h = sha256_file(p)
    assert len(h) == 64
