from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from larrak2.core.constraints import get_constraint_names
from larrak2.core.encoding import N_TOTAL, mid_bounds_candidate
from larrak2.core.types import BreathingConfig, EvalContext
from larrak2.orchestration.adapters.simulation_adapter import candidate_openfoam_params
from larrak2.training import overnight_campaign as oc


def test_candidate_openfoam_projection_uses_runtime_derived_timing() -> None:
    x = np.asarray(mid_bounds_candidate(), dtype=np.float64)
    ctx = EvalContext(
        rpm=2300.0,
        torque=120.0,
        fidelity=0,
        breathing=BreathingConfig(fuel_name="gasoline", valve_timing_mode="candidate"),
        surrogate_validation_mode="off",
        thermo_model="two_zone_eq_v1",
        thermo_timing_profile_path="data/thermo/valve_timing_profile_v1.json",
        thermo_chemistry_profile_path="data/thermo/hybrid_chemistry_profile_v1.json",
    )
    res = oc.evaluate_candidate(x, ctx)
    timing = ((res.diag or {}).get("thermo", {}) or {}).get("valve_timing", {})
    params = candidate_openfoam_params({"id": "cand", "x": x}, ctx, eval_diag=res.diag)
    assert params["intake_open_deg"] == float(timing["intake_open_deg"])
    assert params["intake_close_deg"] == float(timing["intake_close_deg"])
    assert params["exhaust_open_deg"] == float(timing["exhaust_open_deg"])
    assert params["exhaust_close_deg"] == float(timing["exhaust_close_deg"])
    assert params["overlap_deg"] == float(timing["overlap_deg"])


def test_build_stack_dataset_emits_schema(tmp_path: Path, monkeypatch) -> None:
    profile = {
        "profile_id": "test_campaign",
        "fuel_name": "gasoline",
        "principles_profile_path": "data/optimization/principles_frontier_profile_v2.json",
        "breathing_defaults": {},
        "openfoam": {},
        "calculix": {},
        "stack": {
            "operating_points": [{"rpm": 1800.0, "torque": 90.0}],
            "per_point": {"quasi_random": 2, "principles": 1, "local_perturb": 1},
            "min_success_rows": 1,
            "fail_fast_max_failure_fraction": 0.8,
            "dataset_seed": 7,
        },
    }

    class _FakeResult:
        def __init__(self) -> None:
            self.F = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
            self.G = np.zeros(len(get_constraint_names(2)), dtype=np.float64)
            self.diag = {
                "objectives": {"names": ["eta_comb_gap", "eta_exp_gap", "motion_law_penalty"]},
                "constraints": [
                    {"name": name, "scaled_raw": 0.0} for name in get_constraint_names(2)
                ],
            }

    monkeypatch.setattr(oc, "evaluate_candidate", lambda x, ctx: _FakeResult())

    dataset_path, meta = oc.build_stack_dataset(
        profile=profile,
        outdir=tmp_path / "stack_dataset",
        openfoam_model_path="outputs/artifacts/surrogates/openfoam_nn/openfoam_breathing.pt",
        calculix_model_path="outputs/artifacts/surrogates/calculix_nn/calculix_stress.pt",
        anchor_manifest_path="data/thermo/anchor_manifest_v1.json",
        log_fn=lambda *args, **kwargs: None,
    )
    assert dataset_path.exists()
    with np.load(dataset_path, allow_pickle=True) as data:
        assert data["X"].shape[1] == N_TOTAL + 2
        assert data["Y"].shape[1] == 3 + len(get_constraint_names(2))
        assert tuple(str(v) for v in data["feature_names"].tolist())[-2:] == ("rpm", "torque")
    assert meta["n_success"] >= 1


def test_run_overnight_campaign_stages_all_artifacts_without_install(tmp_path: Path, monkeypatch) -> None:
    profile = {
        "profile_id": "test_campaign",
        "profile_version": "1.0",
        "fuel_name": "gasoline",
        "principles_profile_path": "data/optimization/principles_frontier_profile_v2.json",
        "breathing_defaults": {},
        "openfoam": {"min_success_rows": 1, "truth_anchor_target": 1},
        "calculix": {"min_success_rows": 1},
        "stack": {"split": {"val_frac": 0.15, "test_frac": 0.15}},
    }
    monkeypatch.setattr(oc, "load_overnight_campaign_profile", lambda path=None: profile)

    openfoam_template = tmp_path / "openfoam_template"
    (openfoam_template / "system").mkdir(parents=True)
    (openfoam_template / "system" / "controlDict").write_text("application rhoPimpleFoam;\n", encoding="utf-8")
    calculix_template = tmp_path / "gear.inp"
    calculix_template.write_text("*HEADING\n", encoding="utf-8")

    truth_dir = tmp_path / "truth_bundle"
    truth_dir.mkdir(parents=True)
    truth_jsonl = truth_dir / "truth_records.jsonl"
    truth_jsonl.write_text('{"truth_ok": true, "rpm": 2000, "torque": 80}\n', encoding="utf-8")
    anchor_manifest = truth_dir / "anchor_manifest_truth.json"
    anchor_manifest.write_text(
        json.dumps(
            {
                "version": "thermo_anchor_v1",
                "provenance": {
                    "source_type": "truth_runs",
                    "input_files": [str(truth_jsonl)],
                },
                "anchors": [{"rpm": 2000.0, "torque": 80.0, "source": "truth_runs"}],
            }
        ),
        encoding="utf-8",
    )

    openfoam_dataset = tmp_path / "openfoam_results.jsonl"
    openfoam_dataset.write_text('{"ok": true}\n', encoding="utf-8")
    calculix_dataset = tmp_path / "calculix_train.npz"
    np.savez(calculix_dataset, X=np.ones((2, 8)), Y=np.ones((2, 1)))
    stack_dataset = tmp_path / "stack_dataset.npz"
    np.savez(
        stack_dataset,
        X=np.ones((4, N_TOTAL + 2)),
        Y=np.ones((4, 3 + len(get_constraint_names(2)))),
        feature_names=np.array([f"x_{i:03d}" for i in range(N_TOTAL)] + ["rpm", "torque"], dtype=object),
        objective_names=np.array(["eta_comb_gap", "eta_exp_gap", "motion_law_penalty"], dtype=object),
        constraint_names=np.array(get_constraint_names(2), dtype=object),
    )

    openfoam_artifact_dir = tmp_path / "openfoam_artifact"
    openfoam_artifact_dir.mkdir()
    openfoam_artifact = openfoam_artifact_dir / "openfoam_breathing.pt"
    openfoam_artifact.write_bytes(b"pt")
    (openfoam_artifact_dir / "quality_report.json").write_text("{}", encoding="utf-8")
    staged_dir = tmp_path / "openfoam_authority" / "bundle"
    staged_dir.mkdir(parents=True)

    calculix_artifact_dir = tmp_path / "calculix_artifact"
    calculix_artifact_dir.mkdir()
    (calculix_artifact_dir / "calculix_stress.pt").write_bytes(b"ccx")
    (calculix_artifact_dir / "quality_report.json").write_text("{}", encoding="utf-8")

    stack_artifact_dir = tmp_path / "stack_artifact"
    stack_artifact_dir.mkdir()
    stack_artifact = stack_artifact_dir / "stack_f2_surrogate.npz"
    np.savez(stack_artifact, X=np.ones((1, 1)), __meta_json__=json.dumps({"fidelity": 2}))
    (stack_artifact_dir / "quality_report.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        oc,
        "build_truth_anchor_bundle",
        lambda **kwargs: {
            "truth_records_path": str(truth_jsonl),
            "anchor_manifest_path": str(anchor_manifest),
            "anchor_count": 1,
            "truth_successes": 1,
            "truth_cases": 1,
        },
    )
    monkeypatch.setattr(
        oc,
        "build_openfoam_training_dataset",
        lambda **kwargs: (openfoam_dataset, {"source": "doe_generated", "n_success_cases": 1}),
    )
    monkeypatch.setattr(
        oc,
        "train_openfoam_workflow",
        lambda args: {
            "artifact_path": str(openfoam_artifact),
            "authority_bundle": {"promotable": True, "staged_dir": str(staged_dir)},
        },
    )
    monkeypatch.setattr(
        oc,
        "build_calculix_training_dataset",
        lambda **kwargs: (calculix_dataset, {"source": "doe_generated", "n_success_cases": 2}),
    )
    monkeypatch.setattr(oc, "train_calculix_workflow", lambda args: None)
    monkeypatch.setattr(
        oc,
        "build_stack_dataset",
        lambda **kwargs: (stack_dataset, {"n_success": 4, "n_attempted": 4, "n_failures": 0}),
    )
    monkeypatch.setattr(
        oc,
        "train_stack_surrogate_workflow",
        lambda args: {"artifact_path": str(stack_artifact), "metrics": {"n_test": 1}},
    )
    monkeypatch.setattr(oc, "validate_artifact_quality", lambda *args, **kwargs: {"pass": True})

    args = argparse.Namespace(
        profile="",
        outdir_root=str(tmp_path / "overnight_out"),
        run_id="smoke",
        openfoam_template=str(openfoam_template),
        openfoam_solver="rhoPimpleFoam",
        openfoam_docker_timeout_s=30,
        openfoam_docker_image="",
        openfoam_name="openfoam_breathing.pt",
        openfoam_hidden="64,64",
        openfoam_epochs=1,
        openfoam_lr=1e-3,
        openfoam_weight_decay=0.0,
        openfoam_authority_bundle_root="",
        calculix_template=str(calculix_template),
        calculix_solver="ccx",
        calculix_name="calculix_stress.pt",
        calculix_hidden="64,64",
        calculix_epochs=1,
        calculix_lr=1e-3,
        calculix_weight_decay=0.0,
        stack_name="stack_f2_surrogate.npz",
        stack_hidden="16,16",
        stack_activation="relu",
        stack_leaky_relu_slope=0.01,
        stack_epochs=1,
        stack_lr=1e-3,
        stack_weight_decay=1e-6,
        seed=5,
        install_canonical=False,
    )

    summary = oc.run_overnight_f2_nn_campaign(args, log_fn=lambda *args, **kwargs: None)
    assert summary["status"] == "ok"
    assert summary["promotion"]["install_canonical"] is False
    assert "truth_anchors" in summary["steps"]
    assert "openfoam_train" in summary["steps"]
    assert "calculix_train" in summary["steps"]
    assert "stack_train" in summary["steps"]
