"""Archive IO with version guards and migration stubs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .constants import MODEL_VERSION_GEAR_V1, MODEL_VERSION_THERMO_V1
from .encoding import ENCODING_VERSION, N_TOTAL
from .constraints import get_constraint_names, get_constraint_scales


META_FILENAME = "summary.json"


def save_archive(
    outdir: Path,
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    summary: Dict[str, Any],
) -> None:
    """Save archive arrays plus metadata with guards."""
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "pareto_X.npy", X)
    np.save(outdir / "pareto_F.npy", F)
    np.save(outdir / "pareto_G.npy", G)

    summary = {
        **summary,
        "encoding_version": ENCODING_VERSION,
        "model_versions": {
            "thermo_v1": MODEL_VERSION_THERMO_V1,
            "gear_v1": MODEL_VERSION_GEAR_V1,
        },
        "constraint_names": get_constraint_names(summary.get("fidelity", 0)),
        "constraint_scales": get_constraint_scales(),
        "n_var": N_TOTAL,
    }
    with open(outdir / META_FILENAME, "w") as f:
        json.dump(summary, f, indent=2)


def load_archive(outdir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load archive with version validation. Raises on incompatible encoding."""
    summary_path = outdir / META_FILENAME
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {META_FILENAME} in {outdir}")

    with open(summary_path) as f:
        summary = json.load(f)

    enc = summary.get("encoding_version")
    if enc != ENCODING_VERSION:
        raise ValueError(f"Encoding version mismatch: archive {enc}, expected {ENCODING_VERSION}")

    X = np.load(outdir / "pareto_X.npy", allow_pickle=False)
    F = np.load(outdir / "pareto_F.npy", allow_pickle=False)
    G = np.load(outdir / "pareto_G.npy", allow_pickle=False)

    # Basic shape validation
    if X.shape[1] != summary.get("n_var", N_TOTAL):
        raise ValueError(f"n_var mismatch: {X.shape[1]} vs {summary.get('n_var')}")

    return X, F, G, summary


def migrate_archive(summary: dict) -> dict:
    """Placeholder migration hook."""
    # Currently no migrations; would translate fields here.
    return summary
