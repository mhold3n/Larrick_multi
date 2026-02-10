"""Pytest configuration for larrak2.

We make fidelity=2 strict (OpenFOAM NN required). To keep the unit/integration
tests self-contained and deterministic, we generate a tiny synthetic OpenFOAM NN
artifact once per test session and point `LARRAK2_OPENFOAM_NN_PATH` at it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_openfoam_nn_artifact_for_tests() -> None:
    # If the user explicitly sets a model path for their environment, respect it.
    if os.environ.get("LARRAK2_OPENFOAM_NN_PATH"):
        return

    from larrak2.surrogate.openfoam_nn import (
        DEFAULT_FEATURE_KEYS,
        DEFAULT_TARGET_KEYS,
        save_artifact,
        train_openfoam_surrogate,
    )

    rng = np.random.default_rng(0)
    n = 80
    d_in = len(DEFAULT_FEATURE_KEYS)
    d_out = len(DEFAULT_TARGET_KEYS)

    # Create synthetic features in plausible ranges.
    X = np.zeros((n, d_in), dtype=np.float64)
    key_to_col = {k: i for i, k in enumerate(DEFAULT_FEATURE_KEYS)}

    def col(k: str) -> int:
        return key_to_col[k]  # type: ignore[index]

    X[:, col("rpm")] = rng.uniform(1000, 7000, size=n)
    X[:, col("torque")] = rng.uniform(50, 400, size=n)
    X[:, col("lambda_af")] = rng.uniform(0.6, 1.6, size=n)
    X[:, col("bore_mm")] = 80.0
    X[:, col("stroke_mm")] = 90.0
    X[:, col("intake_port_area_m2")] = rng.uniform(2e-4, 8e-4, size=n)
    X[:, col("exhaust_port_area_m2")] = rng.uniform(2e-4, 8e-4, size=n)
    X[:, col("p_manifold_Pa")] = rng.uniform(30_000, 250_000, size=n)
    X[:, col("p_back_Pa")] = rng.uniform(80_000, 200_000, size=n)
    X[:, col("overlap_deg")] = rng.uniform(0, 80, size=n)
    X[:, col("intake_open_deg")] = rng.uniform(-60, 20, size=n)
    X[:, col("intake_close_deg")] = rng.uniform(20, 120, size=n)
    X[:, col("exhaust_open_deg")] = rng.uniform(-120, -20, size=n)
    X[:, col("exhaust_close_deg")] = rng.uniform(-20, 60, size=n)

    # Synthetic targets:
    # - trapped mass scales with manifold pressure and displacement proxy
    p_man = X[:, col("p_manifold_Pa")]
    p_back = X[:, col("p_back_Pa")]
    overlap = X[:, col("overlap_deg")]

    m_air_trapped = 1.2e-3 * (p_man / 101325.0) * (1.0 - 0.002 * overlap)
    residual_fraction = np.clip(0.05 + 0.2 * (p_back / np.maximum(p_man, 1.0) - 0.6), 0.0, 0.6)
    scav_eff = np.clip(1.0 - residual_fraction, 0.0, 1.0)
    trapped_o2_mass = m_air_trapped * 0.233 * (1.0 - residual_fraction)

    Y = np.zeros((n, d_out), dtype=np.float64)
    tmap = {k: i for i, k in enumerate(DEFAULT_TARGET_KEYS)}
    Y[:, tmap["m_air_trapped"]] = m_air_trapped
    Y[:, tmap["scavenging_efficiency"]] = scav_eff
    Y[:, tmap["residual_fraction"]] = residual_fraction
    Y[:, tmap["trapped_o2_mass"]] = trapped_o2_mass

    artifact = train_openfoam_surrogate(
        X,
        Y,
        feature_keys=DEFAULT_FEATURE_KEYS,
        target_keys=DEFAULT_TARGET_KEYS,
        hidden_layers=(32, 32),
        seed=0,
        epochs=200,
        lr=5e-3,
        val_frac=0.2,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="larrak2_openfoam_nn_test_"))
    model_path = tmp_dir / "openfoam_breathing.pt"
    save_artifact(artifact, model_path)

    os.environ["LARRAK2_OPENFOAM_NN_PATH"] = str(model_path)

