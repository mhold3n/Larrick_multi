"""Manual truth-plan behavior for orchestrate workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_orchestrate_manual_truth_plan_filters_truth_evals(tmp_path: Path) -> None:
    outdir = tmp_path / "orchestrate_manual"
    truth_plan = tmp_path / "truth_plan.json"
    truth_plan.write_text(json.dumps(["999999"]), encoding="utf-8")

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
            "1",
            "--truth-dispatch-mode",
            "manual",
            "--truth-plan",
            str(truth_plan),
            "--allow-heuristic-surrogate-fallback",
            "--surrogate-validation-mode",
            "off",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    manifest = json.loads((outdir / "orchestrate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["truth_dispatch_mode"] == "manual"
    assert manifest["iterations"], "expected at least one iteration record"
    assert all(int(it["n_truth_evaluated"]) == 0 for it in manifest["iterations"])
