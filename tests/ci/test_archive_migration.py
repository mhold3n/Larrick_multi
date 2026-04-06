"""Tests for archive migration handling."""

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from larrak2.core.archive_io import load_archive, save_archive
from larrak2.core.constraints import get_constraint_names
from larrak2.core.encoding import ENCODING_VERSION, N_TOTAL, N_TOTAL_V0_4, mid_bounds_candidate
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


def test_legacy_archive_rows_upgrade_to_current_width(tmp_path: Path) -> None:
    out = tmp_path
    n_obj = 6
    n_constr = len(get_constraint_names(0))
    np.save(out / "pareto_X.npy", np.zeros((1, N_TOTAL_V0_4), dtype=np.float64))
    np.save(out / "pareto_F.npy", np.zeros((1, n_obj), dtype=np.float64))
    np.save(out / "pareto_G.npy", np.zeros((1, n_constr), dtype=np.float64))
    (out / "summary.json").write_text(
        json.dumps(
            {
                "encoding_version": "0.4",
                "fidelity": 0,
                "seed": 1,
                "n_pareto": 1,
                "n_constr": n_constr,
                "n_obj": n_obj,
                "n_var": N_TOTAL_V0_4,
            }
        ),
        encoding="utf-8",
    )

    X, _F, _G, summary = load_archive(out)

    assert X.shape == (1, N_TOTAL)
    assert summary["encoding_version"] == ENCODING_VERSION
    assert summary["legacy_upgrade"]["timing_defaults_injected"] is True
