"""Integration test for standalone surrogate training workflow."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def test_train_surrogates_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "surrogate_training"
        openfoam_outdir = Path(tmpdir) / "models_openfoam"
        calculix_outdir = Path(tmpdir) / "models_calculix"
        openfoam_data = Path(tmpdir) / "openfoam_train.npz"
        calculix_data = Path(tmpdir) / "calculix_train.npz"

        X_of = np.array(
            [
                [1200, 40, 1.0, 80, 90, 4e-4, 4e-4, 120000, 101325, 5, -20, 80, -80, 20],
                [2000, 80, 1.0, 80, 90, 4e-4, 4e-4, 140000, 101325, 10, -15, 90, -75, 25],
                [2800, 120, 1.1, 80, 90, 5e-4, 4e-4, 160000, 105000, 15, -10, 95, -70, 30],
            ],
            dtype=np.float64,
        )
        Y_of = np.array(
            [
                [1.0e-3, 0.82, 0.18, 1.9e-4],
                [1.2e-3, 0.85, 0.15, 2.3e-4],
                [1.35e-3, 0.88, 0.12, 2.7e-4],
            ],
            dtype=np.float64,
        )
        np.savez(openfoam_data, X=X_of, Y=Y_of)

        X_ccx = np.array(
            [
                [1200, 40, 35, 12, 2.0, 20, 0.0, 0.0],
                [2000, 80, 42, 14, 2.5, 20, 0.0, 0.1],
                [2800, 120, 50, 16, 3.0, 20, 0.0, 0.2],
            ],
            dtype=np.float64,
        )
        Y_ccx = np.array([[980.0], [1120.0], [1260.0]], dtype=np.float64)
        np.savez(calculix_data, X=X_ccx, Y=Y_ccx)

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "larrak2.cli.run",
                "train-surrogates",
                "--outdir",
                str(outdir),
                "--openfoam-outdir",
                str(openfoam_outdir),
                "--calculix-outdir",
                str(calculix_outdir),
                "--openfoam-data",
                str(openfoam_data),
                "--calculix-data",
                str(calculix_data),
                "--openfoam-epochs",
                "5",
                "--openfoam-hidden",
                "16",
                "--calculix-epochs",
                "5",
                "--calculix-hidden",
                "16",
                "--single-condition",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

        manifest_path = outdir / "surrogate_training_manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["workflow"] == "train_surrogates"
        assert manifest["ready_for_dress_rehearsal"] is True
        assert manifest["steps"]["train_nn_surrogates"]["ok"] is True
