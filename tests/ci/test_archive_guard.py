"""Archive version guard tests."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from larrak2.core.archive_io import load_archive, save_archive
from larrak2.core.constraints import get_constraint_names
from larrak2.core.encoding import ENCODING_VERSION, N_TOTAL
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.types import EvalContext


def test_archive_load_version_match():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        ctx = EvalContext(rpm=2000.0, torque=100.0, fidelity=0, seed=1)
        n_obj = int(evaluate_candidate(mid_bounds_candidate(), ctx).F.size)
        n_constr = len(get_constraint_names(ctx.fidelity))
        X = np.zeros((1, N_TOTAL))
        F = np.zeros((1, n_obj))
        G = np.zeros((1, n_constr))
        save_archive(
            out,
            X,
            F,
            G,
            {
                "fidelity": 0,
                "seed": 1,
                "n_pareto": 1,
                "n_constr": n_constr,
                "n_obj": n_obj,
            },
        )
        X2, F2, G2, summary = load_archive(out)
        assert summary["encoding_version"] == ENCODING_VERSION
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(F, F2)
        np.testing.assert_array_equal(G, G2)


def test_archive_load_version_mismatch_raises(tmp_path, monkeypatch):
    out = tmp_path
    ctx = EvalContext(rpm=2000.0, torque=100.0, fidelity=0, seed=1)
    n_obj = int(evaluate_candidate(mid_bounds_candidate(), ctx).F.size)
    n_constr = len(get_constraint_names(ctx.fidelity))
    X = np.zeros((1, N_TOTAL))
    F = np.zeros((1, n_obj))
    G = np.zeros((1, n_constr))
    save_archive(
        out,
        X,
        F,
        G,
        {
            "fidelity": 0,
            "seed": 1,
            "n_pareto": 1,
            "n_constr": n_constr,
            "n_obj": n_obj,
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
