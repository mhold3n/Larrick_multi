"""CLI smoke test for orchestrate run type."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from larrak2.architecture.contracts import CONTRACT_VERSION
from larrak2.cli.run import main as run_main


def test_orchestrate_cli_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "orchestrate_smoke"
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
