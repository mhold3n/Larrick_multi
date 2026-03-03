"""CandidateStore tests for explore->exploit pipeline."""

from __future__ import annotations

import json

import numpy as np

from larrak2.core.archive_io import save_archive
from larrak2.core.encoding import N_TOTAL, mid_bounds_candidate
from larrak2.optimization.candidate_store import CandidateStore


def test_candidate_store_feasible_filter_rank_and_export(tmp_path):
    x0 = mid_bounds_candidate()
    X = np.vstack([x0, x0 * 0.99, x0 * 1.01])
    F = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 5.0, 5.0],
            [0.8, 1.8, 2.8],
        ],
        dtype=np.float64,
    )
    G = np.array(
        [
            [0.0, -1.0],
            [0.2, 0.0],  # infeasible
            [-0.1, -0.2],
        ],
        dtype=np.float64,
    )
    save_archive(tmp_path, X, F, G, {"fidelity": 0, "seed": 1, "rpm": 2000.0, "torque": 120.0})

    store = CandidateStore.from_archive_dir(tmp_path)
    assert store.n_candidates == 3

    feasible = store.feasible_indices()
    assert feasible == [0, 2]

    ranked = store.rank_indices(objective_weights=np.array([1.0, 0.0, 0.0]), feasible_only=True)
    assert ranked == [2, 0]

    out = store.export_x_full_star(2, tmp_path / "x_full_star.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["candidate_index"] == 2
    assert len(payload["x_full"]) == N_TOTAL
    assert payload["n_var"] == N_TOTAL
