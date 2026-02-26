"""Refine Pareto candidates with slice metadata checks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from larrak2.cli.refine_pareto import main as refine_main
from larrak2.core.encoding import N_TOTAL, mid_bounds_candidate


def test_refine_pareto_slice_metadata_and_full_dimensionality():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        x0 = mid_bounds_candidate()
        X = np.tile(x0, (1, 1))
        F = np.zeros((1, 3), dtype=np.float64)
        G = np.zeros((1, 10), dtype=np.float64)
        np.save(tmp / "pareto_X.npy", X)
        np.save(tmp / "pareto_F.npy", F)
        np.save(tmp / "pareto_G.npy", G)

        code = refine_main(
            [
                "--input",
                tmpdir,
                "--top-k",
                "1",
                "--backend",
                "casadi",
                "--slice-method",
                "sensitivity",
                "--active-k",
                "6",
            ]
        )
        assert code == 0

        refined_X = np.load(tmp / "refined_X.npy")
        assert refined_X.shape == (1, N_TOTAL)

        summary = json.loads((tmp / "refinement_summary.json").read_text(encoding="utf-8"))
        assert summary["backend"] == "casadi"
        assert summary["slice_method"] == "sensitivity"
        assert summary["n_refined"] == 1

        row = summary["results"][0]
        active = row["active_indices"]
        frozen = row["frozen_indices"]
        assert isinstance(active, list)
        assert isinstance(frozen, list)
        assert sorted(active + frozen) == list(range(N_TOTAL))
        assert row["backend_used"] in {"casadi", "scipy_fallback"}
