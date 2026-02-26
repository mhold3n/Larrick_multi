"""Tests for archive migration handling."""

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from larrak2.core.archive_io import load_archive, save_archive
from larrak2.core.constraints import get_constraint_names
from larrak2.core.encoding import ENCODING_VERSION, N_TOTAL
from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def _make_archive(tmp_path: Path, encoding_version: str):
    ctx = EvalContext(rpm=2000.0, torque=100.0, fidelity=0, seed=1)
    n_obj = int(evaluate_candidate(mid_bounds_candidate(), ctx).F.size)
    n_constr = len(get_constraint_names(ctx.fidelity))
    X = np.zeros((1, N_TOTAL))
    F = np.zeros((1, n_obj))
    G = np.zeros((1, n_constr))
    save_archive(
        tmp_path,
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
    # Overwrite encoding version
    summary_path = tmp_path / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    summary["encoding_version"] = encoding_version
    with open(summary_path, "w") as f:
        json.dump(summary, f)


def test_archive_migration_warns_and_loads():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _make_archive(out, "0.0")
        with warnings.catch_warnings(record=True) as w:
            X, F, G, summary = load_archive(out)
        assert any("Migrating archive encoding" in str(wi.message) for wi in w)
        assert summary["encoding_version"] == ENCODING_VERSION


def test_archive_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _make_archive(out, "bad_version")
        with pytest.raises(ValueError, match="Encoding version mismatch"):
            load_archive(out)
