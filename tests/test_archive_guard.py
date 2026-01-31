"""Archive version guard tests."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from larrak2.core.archive_io import load_archive, save_archive
from larrak2.core.encoding import ENCODING_VERSION, N_TOTAL


def test_archive_load_version_match():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        X = np.zeros((1, N_TOTAL))
        F = np.zeros((1, 3))
        G = np.zeros((1, 10))
        save_archive(
            out,
            X,
            F,
            G,
            {
                "fidelity": 0,
                "seed": 1,
                "n_pareto": 1,
                "n_constr": 10,
                "n_obj": 3,
            },
        )
        X2, F2, G2, summary = load_archive(out)
        assert summary["encoding_version"] == ENCODING_VERSION
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(F, F2)
        np.testing.assert_array_equal(G, G2)


def test_archive_load_version_mismatch_raises(tmp_path, monkeypatch):
    out = tmp_path
    X = np.zeros((1, N_TOTAL))
    F = np.zeros((1, 3))
    G = np.zeros((1, 10))
    save_archive(
        out,
        X,
        F,
        G,
        {
            "fidelity": 0,
            "seed": 1,
            "n_pareto": 1,
            "n_constr": 10,
            "n_obj": 3,
        },
    )
    # Corrupt version
    summary_path = out / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    summary["encoding_version"] = "bad_version"
    with open(summary_path, "w") as f:
        json.dump(summary, f)

    with pytest.raises(ValueError, match="Encoding version mismatch"):
        load_archive(out)
