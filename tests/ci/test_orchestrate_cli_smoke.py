"""CLI smoke test for orchestrate run type."""

from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from larrak2.architecture.contracts import CONTRACT_VERSION
from larrak2.cli.run import main as run_main
from larrak2.cli.run_workflows import run_orchestrate_workflow


def test_orchestrate_cli_smoke(tmp_path: Path) -> None:
    pytest.importorskip("casadi")
    outdir = tmp_path / "orchestrate_smoke"
    stack_model = tmp_path / "stack_f2_surrogate.npz"
    stack_model.write_bytes(b"placeholder")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "larrak2.cli.run",
            "orchestrate",
            "--outdir",
            str(outdir),
            "--rpm",
            "2200",
            "--torque",
            "120",
            "--seed",
            "123",
            "--sim-budget",
            "4",
            "--batch-size",
            "4",
            "--max-iterations",
            "2",
            "--truth-dispatch-mode",
            "off",
            "--allow-heuristic-surrogate-fallback",
            "--surrogate-validation-mode",
            "off",
            "--thermo-symbolic-mode",
            "off",
            "--stack-model-path",
            str(stack_model),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    manifest_path = outdir / "orchestrate_manifest.json"
    provenance_path = outdir / "provenance_events.jsonl"
    contract_trace_path = outdir / "contract_trace.jsonl"
    contract_summary_path = outdir / "contract_summary.json"
    assert manifest_path.exists()
    assert provenance_path.exists()
    assert contract_trace_path.exists()
    assert contract_summary_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["workflow"] == "orchestrate"
    assert manifest["result"]["n_iterations"] >= 1
    assert manifest["files"]["orchestrate_manifest"] == str(manifest_path)
    assert manifest["contract_version"] == CONTRACT_VERSION
    assert manifest["contract_trace_file"] == str(contract_trace_path)
    assert manifest["contract_summary_file"] == str(contract_summary_path)
    assert isinstance(manifest.get("contract_summary", {}), dict)
    gate = manifest["production_gate"]
    for key in (
        "production_profile",
        "production_gate_pass",
        "production_gate_failures",
        "fallback_paths_used",
        "nonproduction_overrides",
        "n_eval_errors",
        "algorithm_used",
        "fidelity",
        "constraint_phase",
    ):
        assert key in manifest
        assert manifest[key] == gate[key]


def test_orchestrate_cli_defaults(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _mock_workflow(args):
        captured["thermo_symbolic_mode"] = str(args.thermo_symbolic_mode)
        captured["fidelity"] = int(args.fidelity)
        captured["constraint_phase"] = str(args.constraint_phase)
        captured["allow_nonproduction_paths"] = bool(args.allow_nonproduction_paths)
        captured["enforce_contract_routing"] = bool(args.enforce_contract_routing)
        captured["thermo_constants_path"] = str(args.thermo_constants_path)
        captured["thermo_anchor_manifest"] = str(args.thermo_anchor_manifest)
        captured["stack_model_path"] = str(args.stack_model_path)
        captured["ipopt_max_iter"] = args.ipopt_max_iter
        captured["ipopt_tol"] = args.ipopt_tol
        captured["ipopt_linear_solver"] = args.ipopt_linear_solver
        return 0

    monkeypatch.setattr("larrak2.cli.run.run_orchestrate_workflow", _mock_workflow)
    with patch.object(sys, "argv", ["run.py", "orchestrate"]):
        code = run_main()
    assert code == 0
    assert captured["thermo_symbolic_mode"] == "strict"
    assert captured["fidelity"] == 2
    assert captured["constraint_phase"] == "downselect"
    assert captured["allow_nonproduction_paths"] is False
    assert captured["enforce_contract_routing"] is False
    assert captured["thermo_constants_path"] == ""
    assert captured["thermo_anchor_manifest"] == ""
    assert captured["stack_model_path"] == ""
    assert captured["ipopt_max_iter"] is None
    assert captured["ipopt_tol"] is None
    assert captured["ipopt_linear_solver"] is None


def test_orchestrate_cli_passthroughs_casadi_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _mock_workflow(args):
        captured["stack_model_path"] = str(args.stack_model_path)
        captured["ipopt_max_iter"] = args.ipopt_max_iter
        captured["ipopt_tol"] = args.ipopt_tol
        captured["ipopt_linear_solver"] = args.ipopt_linear_solver
        return 0

    monkeypatch.setattr("larrak2.cli.run.run_orchestrate_workflow", _mock_workflow)
    with patch.object(
        sys,
        "argv",
        [
            "run.py",
            "orchestrate",
            "--stack-model-path",
            "outputs/artifacts/surrogates/stack_f2/stack_f2_surrogate.npz",
            "--ipopt-max-iter",
            "123",
            "--ipopt-tol",
            "1e-7",
            "--ipopt-linear-solver",
            "mumps",
        ],
    ):
        code = run_main()
    assert code == 0
    assert (
        captured["stack_model_path"]
        == "outputs/artifacts/surrogates/stack_f2/stack_f2_surrogate.npz"
    )
    assert captured["ipopt_max_iter"] == 123
    assert captured["ipopt_tol"] == 1e-7
    assert captured["ipopt_linear_solver"] == "mumps"


def test_orchestrate_workflow_wires_solver_stack_and_ipopt(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    stack_model = tmp_path / "stack_f2_surrogate.npz"
    stack_model.write_bytes(b"placeholder")

    class _DummyCEMAdapter:
        pass

    class _DummySurrogateAdapter:
        def __init__(self, *args, **kwargs):
            _ = args
            captured["surrogate_kwargs"] = dict(kwargs)

    class _DummySolverAdapter:
        def __init__(self, **kwargs):
            captured["solver_kwargs"] = dict(kwargs)

    class _DummySimulationAdapter:
        def __init__(self, *args, **kwargs):
            _ = args
            _ = kwargs

    class _DummyOrchestrator:
        def __init__(self, **kwargs):
            captured["config"] = kwargs["config"]

        def optimize(self, initial_params=None):
            _ = initial_params
            return SimpleNamespace(
                best_objective=0.0,
                best_source="surrogate",
                manifest_path=str(tmp_path / "orchestrate_manifest.json"),
            )

    monkeypatch.setattr("larrak2.orchestration.adapters.CEMAdapter", _DummyCEMAdapter)
    monkeypatch.setattr(
        "larrak2.orchestration.adapters.HifiSurrogateAdapter", _DummySurrogateAdapter
    )
    monkeypatch.setattr(
        "larrak2.cli.run_workflows._ensure_casadi_available", lambda **_kwargs: None
    )
    monkeypatch.setattr("larrak2.orchestration.adapters.CasadiSolverAdapter", _DummySolverAdapter)
    monkeypatch.setattr(
        "larrak2.orchestration.adapters.PhysicsSimulationAdapter", _DummySimulationAdapter
    )
    monkeypatch.setattr("larrak2.orchestration.Orchestrator", _DummyOrchestrator)

    args = Namespace(
        outdir=str(tmp_path / "orchestrate"),
        rpm=2200.0,
        torque=120.0,
        fidelity=2,
        constraint_phase="downselect",
        enforce_contract_routing=False,
        seed=123,
        sim_budget=4,
        batch_size=4,
        max_iterations=2,
        truth_dispatch_mode="off",
        truth_plan="",
        truth_auto_top_k=2,
        truth_auto_min_uncertainty=0.0,
        truth_auto_min_feasibility=0.0,
        truth_auto_min_pred_quantile=0.0,
        hifi_model_dir=str(tmp_path / "hifi"),
        allow_heuristic_surrogate_fallback=True,
        allow_nonproduction_paths=False,
        surrogate_validation_mode="off",
        thermo_symbolic_mode="off",
        thermo_symbolic_artifact_path="",
        stack_model_path=str(stack_model),
        ipopt_max_iter=123,
        ipopt_tol=1e-7,
        ipopt_linear_solver="mumps",
        thermo_constants_path="",
        thermo_anchor_manifest="",
        strict_data=True,
        strict_tribology_data=None,
        tribology_scuff_method="auto",
        machining_mode="nn",
        machining_model_path="",
        control_backend="file",
        provenance_backend="off",
        cache_path="",
        multi_start=False,
    )
    code = run_orchestrate_workflow(args)
    assert code == 0
    assert captured["solver_kwargs"] == {
        "backend": "casadi",
        "mode": "weighted_sum",
        "stack_model_path": str(stack_model),
        "ipopt_options": {"max_iter": 123, "tol": 1e-7, "linear_solver": "mumps"},
        "trust_radius": None,
    }
    config = captured["config"]
    assert config.stack_model_path == str(stack_model)
    assert config.ipopt_options == {"max_iter": 123, "tol": 1e-7, "linear_solver": "mumps"}
    surrogate_kwargs = captured["surrogate_kwargs"]
    assert "required_quality_artifacts" in surrogate_kwargs
    required = surrogate_kwargs["required_quality_artifacts"]
    assert isinstance(required, list)
    assert any(str(v).endswith("pipeline_readiness_summary.md") for v in required)
    assert any(str(v).endswith("f2_blockers.json") for v in required)
    assert any(str(v).endswith("artifact_contract_audit.json") for v in required)


def test_orchestrate_workflow_enables_openfoam_truth_dispatch(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    stack_model = tmp_path / "stack_f2_surrogate.npz"
    stack_model.write_bytes(b"placeholder")
    template_dir = tmp_path / "openfoam_template"
    template_dir.mkdir()

    class _DummyCEMAdapter:
        pass

    class _DummySurrogateAdapter:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            return None

    class _DummySolverAdapter:
        def __init__(self, **kwargs):  # noqa: ARG002
            return None

    class _DummyDocker:
        def check_availability(self) -> bool:
            return True

    class _DummyOpenFoamPipeline:
        def __init__(self, **kwargs):
            captured["openfoam_runner_kwargs"] = dict(kwargs)
            self.docker = _DummyDocker()
            self.template_dir = kwargs.get("template_dir")

    class _DummySimulationAdapter:
        def __init__(self, *args, **kwargs):
            _ = args
            captured["simulation_kwargs"] = dict(kwargs)

    class _DummyOrchestrator:
        def __init__(self, **kwargs):
            _ = kwargs

        def optimize(self, initial_params=None):  # noqa: ARG002
            return SimpleNamespace(
                best_objective=0.0,
                best_source="truth",
                manifest_path=str(tmp_path / "orchestrate_manifest.json"),
            )

    monkeypatch.setattr("larrak2.orchestration.adapters.CEMAdapter", _DummyCEMAdapter)
    monkeypatch.setattr(
        "larrak2.orchestration.adapters.HifiSurrogateAdapter", _DummySurrogateAdapter
    )
    monkeypatch.setattr(
        "larrak2.cli.run_workflows._ensure_casadi_available", lambda **_kwargs: None
    )
    monkeypatch.setattr("larrak2.orchestration.adapters.CasadiSolverAdapter", _DummySolverAdapter)
    monkeypatch.setattr(
        "larrak2.orchestration.adapters.PhysicsSimulationAdapter", _DummySimulationAdapter
    )
    monkeypatch.setattr("larrak2.orchestration.Orchestrator", _DummyOrchestrator)
    monkeypatch.setattr("larrak2.pipelines.openfoam.OpenFoamPipeline", _DummyOpenFoamPipeline)

    args = Namespace(
        outdir=str(tmp_path / "orchestrate"),
        rpm=2200.0,
        torque=120.0,
        fidelity=2,
        constraint_phase="downselect",
        enforce_contract_routing=False,
        seed=123,
        sim_budget=4,
        batch_size=4,
        max_iterations=2,
        truth_dispatch_mode="auto",
        truth_plan="",
        truth_auto_top_k=2,
        truth_auto_min_uncertainty=0.0,
        truth_auto_min_feasibility=0.0,
        truth_auto_min_pred_quantile=0.0,
        truth_run_openfoam=True,
        truth_records_path="",
        openfoam_template=str(template_dir),
        openfoam_solver="rhoPimpleFoam",
        openfoam_backend="docker",
        openfoam_docker_image="",
        hifi_model_dir=str(tmp_path / "hifi"),
        allow_heuristic_surrogate_fallback=True,
        allow_nonproduction_paths=False,
        surrogate_validation_mode="off",
        thermo_symbolic_mode="off",
        thermo_symbolic_artifact_path="",
        stack_model_path=str(stack_model),
        ipopt_max_iter=None,
        ipopt_tol=None,
        ipopt_linear_solver=None,
        thermo_constants_path="",
        thermo_anchor_manifest="",
        thermo_chemistry_profile_path="",
        strict_data=True,
        strict_tribology_data=None,
        tribology_scuff_method="auto",
        machining_mode="nn",
        machining_model_path="",
        control_backend="file",
        provenance_backend="off",
        cache_path="",
        multi_start=False,
        bore_mm=80.0,
        stroke_mm=90.0,
        intake_port_area_m2=4.0e-4,
        exhaust_port_area_m2=4.0e-4,
        p_manifold_pa=101325.0,
        p_back_pa=101325.0,
        compression_ratio=10.0,
        fuel_name="gasoline",
    )
    code = run_orchestrate_workflow(args)
    assert code == 0
    assert captured["openfoam_runner_kwargs"] == {
        "template_dir": template_dir,
        "solver_cmd": "rhoPimpleFoam",
        "docker_image": None,
    }
    simulation_kwargs = captured["simulation_kwargs"]
    assert simulation_kwargs["run_openfoam"] is True
    assert simulation_kwargs["openfoam_runner"].template_dir == template_dir
