"""Tests that refinement outputs are version-stamped."""

import json
import tempfile
from pathlib import Path

import numpy as np

from larrak2.cli.refine_pareto import main as refine_main
from larrak2.core.constraints import get_constraint_names
from larrak2.core.encoding import ENCODING_VERSION, mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def test_refinement_summary_contains_versions():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Minimal Pareto archive
        probe_ctx = EvalContext(rpm=2000.0, torque=150.0, fidelity=0, seed=1)
        n_obj = int(evaluate_candidate(mid_bounds_candidate(), probe_ctx).F.size)
        n_constr = len(get_constraint_names(probe_ctx.fidelity))
        X = np.tile(mid_bounds_candidate(), (1, 1))
        F = np.zeros((1, n_obj))
        G = np.zeros((1, n_constr))
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
        expected_constraints = get_constraint_names(int(summary["fidelity"]))
        assert "constraint_names" in summary and summary["constraint_names"] == expected_constraints
        assert "constraint_scales" in summary
