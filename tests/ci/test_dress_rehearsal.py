"""Integration test for the dress-rehearsal workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_dress_rehearsal_end_to_end_smoke():
    """Dress rehearsal should produce gate artifacts and complete successfully."""
    openfoam_model = os.environ.get("LARRAK2_OPENFOAM_NN_PATH", "")
    calculix_model = os.environ.get("LARRAK2_CALCULIX_NN_PATH", "")
    assert openfoam_model
    assert calculix_model

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "rehearsal"

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "larrak2.cli.run",
                "dress-rehearsal",
                "--outdir",
                str(outdir),
                "--openfoam-model-path",
                openfoam_model,
                "--calculix-model-path",
                calculix_model,
                "--skip-unit-tests",
                "--single-condition",
                "--fidelity",
                "0",
                "--pop",
                "8",
                "--gen",
                "2",
                "--cem-top",
                "5",
                "--cem-min-feasible",
                "0",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

        manifest_path = outdir / "dress_rehearsal_manifest.json"
        report_txt = outdir / "cem_validation_report.txt"
        report_json = outdir / "cem_validation_report.json"

        assert manifest_path.exists()
        assert report_txt.exists()
        assert report_json.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["workflow"] == "dress_rehearsal"
        assert manifest["ready_for_quality_analysis"] is True
        assert manifest["steps"]["verify_surrogates"]["ok"] is True
        assert manifest["steps"]["unit_tests"]["ok"] is True
        assert manifest["steps"]["optimization"]["ok"] is True
        assert manifest["steps"]["cem_validation"]["ok"] is True
