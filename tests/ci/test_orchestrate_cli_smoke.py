"""CLI smoke test for orchestrate run type."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    manifest_path = outdir / "orchestrate_manifest.json"
    provenance_path = outdir / "provenance_events.jsonl"
    assert manifest_path.exists()
    assert provenance_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["workflow"] == "orchestrate"
    assert manifest["result"]["n_iterations"] >= 1
    assert manifest["files"]["orchestrate_manifest"] == str(manifest_path)
