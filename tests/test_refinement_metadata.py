"""Tests that refinement outputs are version-stamped."""

import json
import tempfile
from pathlib import Path

import numpy as np

from larrak2.cli.refine_pareto import main as refine_main
from larrak2.core.encoding import ENCODING_VERSION, mid_bounds_candidate, N_TOTAL


def test_refinement_summary_contains_versions():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Minimal Pareto archive
        X = np.tile(mid_bounds_candidate(), (1, 1))
        F = np.zeros((1, 3))
        G = np.zeros((1, 7))
        np.save(tmp / "pareto_X.npy", X)
        np.save(tmp / "pareto_F.npy", F)
        np.save(tmp / "pareto_G.npy", G)

        exit_code = refine_main(
            [
                "--input",
                tmpdir,
                "--top-k",
                "1",
                "--mode",
                "weighted_sum",
                "--rpm",
                "2000",
                "--torque",
                "150",
            ]
        )
        assert exit_code == 0

        summary_path = tmp / "refinement_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["encoding_version"] == ENCODING_VERSION
        assert "model_versions" in summary
        assert "constraint_names" in summary and len(summary["constraint_names"]) == 10
        assert "constraint_scales" in summary
