"""Pytest configuration for larrak2.

We make fidelity=2 strict (OpenFOAM + CalculiX NN required). To keep the
unit/integration tests self-contained and deterministic, we generate tiny
synthetic artifacts once per test session and point environment variables
at them.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Patch external `larrak_runtime` data resolution early.
#
# Some external packages import `larrak_runtime.core.encoding` at module-import time,
# which calls into thermo timing bounds and loads a JSON profile from a `data/`
# directory adjacent to the installed wheel. In this repo, we keep `data/` at the
# repo root; we patch the default path *during conftest import* so test collection
# does not fail before fixtures run.
_REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import larrak_runtime.thermo.timing_profile as _tp

    _profile = _REPO_ROOT / "data" / "thermo" / "valve_timing_profile_v1.json"
    if _profile.exists():
        _tp.DEFAULT_THERMO_TIMING_PROFILE_PATH = _profile
        # Clear caches so the new path is used even if imported earlier.
        if hasattr(_tp.load_thermo_timing_profile, "cache_clear"):
            _tp.load_thermo_timing_profile.cache_clear()  # type: ignore[attr-defined]
        if hasattr(_tp.thermo_timing_bounds, "cache_clear"):
            _tp.thermo_timing_bounds.cache_clear()  # type: ignore[attr-defined]

    import larrak_runtime.thermo.chemistry_profile as _cp

    _chem = _REPO_ROOT / "data" / "thermo" / "hybrid_chemistry_profile_v1.json"
    if _chem.exists():
        _cp.DEFAULT_THERMO_CHEMISTRY_PROFILE_PATH = _chem
        if hasattr(_cp.load_thermo_chemistry_profile, "cache_clear"):
            _cp.load_thermo_chemistry_profile.cache_clear()  # type: ignore[attr-defined]
        if hasattr(_cp.spark_timing_bounds, "cache_clear"):
            _cp.spark_timing_bounds.cache_clear()  # type: ignore[attr-defined]

    # Patch CEM dataset discovery root to the repo `data/cem/`.
    import larrak_runtime.cem.registry as _cem_registry

    _cem_root = _REPO_ROOT / "data" / "cem"
    if _cem_root.exists():
        _cem_registry._DATA_CEM_ROOT = _cem_root  # type: ignore[attr-defined]
        # Drop any cached empty tables picked up before the patch.
        try:
            _cem_registry.get_registry()._cache.clear()  # type: ignore[attr-defined]
        except Exception:
            pass

    # Default external EvalContext settings for repo CI.
    #
    # The integration repo does not vendor full datasets/artifacts required for
    # strict runtime evaluation. CI should run in "warn" degradation mode unless
    # individual tests explicitly request strict behavior.
    # Dataclass field defaults do not affect the generated __init__ defaults at runtime,
    # so we patch the generated __init__.__defaults__ tuple directly.
    import inspect

    from larrak_runtime.core.types import EvalContext as _EvalContext

    params = list(inspect.signature(_EvalContext).parameters.values())
    default_params = [p for p in params if p.default is not inspect._empty]
    defaults = list(_EvalContext.__init__.__defaults__ or ())

    def _set_default(name: str, value: object) -> None:
        for i, p in enumerate(default_params):
            if p.name == name:
                defaults[i] = value
                return

    _set_default("strict_data", False)
    _set_default("surrogate_validation_mode", "warn")
    _set_default("thermo_symbolic_mode", "warn")
    _EvalContext.__init__.__defaults__ = tuple(defaults)
except Exception:
    # If externals aren't installed yet, collection will fail elsewhere anyway;
    # keep conftest import resilient.
    pass


@pytest.fixture(scope="session", autouse=True)
def _ensure_repo_root_env_for_external_packages() -> None:
    """Point external packages at this repo's `data/` directory.

    `larrak_runtime` resolves its `data/` directory relative to the installed
    wheel location by default. In `Larrick_multi`, we keep `data/` at the repo
    root; setting `LARRICK_MULTI_ROOT` makes that location discoverable in a
    deterministic way for tests.
    """

    os.environ.setdefault("LARRICK_MULTI_ROOT", str(Path(__file__).resolve().parents[1]))


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


@pytest.fixture(scope="session", autouse=True)
def _ensure_calculix_nn_artifact_for_tests() -> None:
    if os.environ.get("LARRAK2_CALCULIX_NN_PATH"):
        return

    from larrak2.surrogate.calculix_nn import (
        DEFAULT_FEATURE_KEYS,
        DEFAULT_TARGET_KEYS,
        save_artifact,
        train_calculix_surrogate,
    )

    rng = np.random.default_rng(1)
    n = 80
    d_in = len(DEFAULT_FEATURE_KEYS)
    d_out = len(DEFAULT_TARGET_KEYS)

    X = np.zeros((n, d_in), dtype=np.float64)
    key_to_col = {k: i for i, k in enumerate(DEFAULT_FEATURE_KEYS)}

    def col(k: str) -> int:
        return key_to_col[k]  # type: ignore[index]

    X[:, col("rpm")] = rng.uniform(1000, 7000, size=n)
    X[:, col("torque")] = rng.uniform(50, 400, size=n)
    X[:, col("base_radius_mm")] = rng.uniform(25.0, 120.0, size=n)
    X[:, col("face_width_mm")] = rng.uniform(8.0, 35.0, size=n)
    X[:, col("module_mm")] = rng.uniform(1.0, 5.0, size=n)
    X[:, col("pressure_angle_deg")] = rng.uniform(15.0, 30.0, size=n)
    X[:, col("helix_angle_deg")] = rng.uniform(-20.0, 20.0, size=n)
    X[:, col("profile_shift")] = rng.uniform(-0.5, 0.5, size=n)

    stress = (
        700.0
        + 0.07 * X[:, col("rpm")]
        + 0.9 * X[:, col("torque")]
        + 120.0 / np.maximum(X[:, col("module_mm")], 0.5)
        + 80.0 * np.abs(X[:, col("profile_shift")])
        + 0.8 * np.abs(X[:, col("helix_angle_deg")])
    )
    Y = stress.reshape(n, d_out).astype(np.float64)

    artifact = train_calculix_surrogate(
        X,
        Y,
        feature_keys=DEFAULT_FEATURE_KEYS,
        target_keys=DEFAULT_TARGET_KEYS,
        hidden_layers=(32, 32),
        seed=1,
        epochs=200,
        lr=5e-3,
        val_frac=0.2,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="larrak2_calculix_nn_test_"))
    model_path = tmp_dir / "calculix_stress.pt"
    save_artifact(artifact, model_path)
    os.environ["LARRAK2_CALCULIX_NN_PATH"] = str(model_path)


@pytest.fixture(scope="session", autouse=True)
def _ensure_machining_nn_artifact_for_tests() -> None:
    from larrak_engines.gear.manufacturability_limits import PROFILE_NAMES
    from larrak_runtime.core.artifact_paths import DEFAULT_MACHINING_NN_ARTIFACT

    from larrak2.surrogate.machining_inference import MachiningSurrogateNet

    model_path = Path(DEFAULT_MACHINING_NN_ARTIFACT)
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    input_dim = 2 + len(PROFILE_NAMES)
    model = MachiningSurrogateNet(input_dim=input_dim)
    import torch

    torch.save(model.state_dict(), model_path)
