"""Explicit SciPy refinement mode smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from larrak2.cli.refine_pareto import main as refine_main
from larrak2.core.encoding import N_TOTAL, mid_bounds_candidate


def test_refine_pareto_scipy_backend_annotation(tmp_path: Path) -> None:
    x0 = mid_bounds_candidate()
    X = np.tile(x0, (1, 1))
    F = np.zeros((1, 6), dtype=np.float64)
    G = np.zeros((1, 10), dtype=np.float64)
    np.save(tmp_path / "pareto_X.npy", X)
    np.save(tmp_path / "pareto_F.npy", F)
    np.save(tmp_path / "pareto_G.npy", G)

    code = refine_main(
        [
            "--input",
            str(tmp_path),
            "--top-k",
            "1",
            "--backend",
            "scipy",
            "--mode",
            "weighted_sum",
            "--active-k",
            "6",
        ]
    )
    assert code == 0

    refined_X = np.load(tmp_path / "refined_X.npy")
    assert refined_X.shape == (1, N_TOTAL)

    summary = json.loads((tmp_path / "refinement_summary.json").read_text(encoding="utf-8"))
    assert summary["backend"] == "scipy"
    assert summary["results"][0]["backend_used"] == "scipy"
