"""Archive IO with version guards and migration stubs."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .constants import MODEL_VERSION_GEAR_V1, MODEL_VERSION_THERMO_V1
from .constraints import (
    get_constraint_kinds_for_phase,
    get_constraint_names,
    get_constraint_scales,
    get_material_constraint_names,
)
from .encoding import (
    ENCODING_VERSION,
    LEGACY_ENCODING_VERSION,
    LEGACY_N_TOTAL,
    N_TOTAL,
    PRECHEM_ENCODING_VERSION,
    PRECHEM_N_TOTAL,
    upgrade_legacy_candidate_matrix,
    upgrade_prechem_candidate_matrix,
)

META_FILENAME = "summary.json"


def save_archive(
    outdir: Path,
    X: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    summary: dict[str, Any],
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
        "constraint_kinds": get_constraint_kinds_for_phase(
            str(summary.get("constraint_phase", "explore"))
        ),
        "material_constraint_names": get_material_constraint_names(),
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
        summary = migrate_archive(summary)

    X = np.load(outdir / "pareto_X.npy", allow_pickle=False)
    F = np.load(outdir / "pareto_F.npy", allow_pickle=False)
    G = np.load(outdir / "pareto_G.npy", allow_pickle=False)

    expected_n_var = int(summary.get("n_var", N_TOTAL))
    if X.shape[1] == LEGACY_N_TOTAL and expected_n_var == N_TOTAL:
        X = upgrade_legacy_candidate_matrix(X)
    elif X.shape[1] == PRECHEM_N_TOTAL and expected_n_var == N_TOTAL:
        X = upgrade_prechem_candidate_matrix(X)
    elif X.shape[1] != expected_n_var:
        raise ValueError(f"n_var mismatch: {X.shape[1]} vs {summary.get('n_var')}")

    return X, F, G, summary


def migrate_archive(summary: dict) -> dict:
    """Migration hook for legacy archive summaries."""
    enc = summary.get("encoding_version")
    if enc == ENCODING_VERSION:
        return summary

    # Example migration map; extend as schemas evolve
    MIGRATION_MAP = {
        "0.0": lambda s: {**s, "encoding_version": ENCODING_VERSION, "n_var": N_TOTAL},
        LEGACY_ENCODING_VERSION: lambda s: {
            **s,
            "encoding_version": ENCODING_VERSION,
            "n_var": N_TOTAL,
            "legacy_upgrade": {
                "timing_defaults_injected": True,
                "legacy_n_var": LEGACY_N_TOTAL,
                "current_n_var": N_TOTAL,
            },
        },
        PRECHEM_ENCODING_VERSION: lambda s: {
            **s,
            "encoding_version": ENCODING_VERSION,
            "n_var": N_TOTAL,
            "legacy_upgrade": {
                "spark_default_injected": True,
                "legacy_n_var": PRECHEM_N_TOTAL,
                "current_n_var": N_TOTAL,
            },
        },
    }

    if enc in MIGRATION_MAP:
        warnings.warn(f"Migrating archive encoding {enc} -> {ENCODING_VERSION}", UserWarning)
        return MIGRATION_MAP[enc](summary)

    raise ValueError(f"Encoding version mismatch: archive {enc}, expected {ENCODING_VERSION}")
