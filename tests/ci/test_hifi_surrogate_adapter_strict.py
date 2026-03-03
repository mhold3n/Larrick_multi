"""Strict runtime behavior for HiFi surrogate adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from larrak2.core.encoding import bounds
from larrak2.orchestration.adapters.surrogate_adapter import HifiSurrogateAdapter


def _candidate() -> dict[str, np.ndarray]:
    xl, xu = bounds()
    return {"x": (xl + xu) * 0.5}


def test_hifi_adapter_fails_hard_without_assets(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        HifiSurrogateAdapter(
            model_dir=tmp_path / "missing_hifi",
            allow_heuristic_fallback=False,
            validation_mode="strict",
        )


def test_hifi_adapter_allows_explicit_nonprod_fallback(tmp_path: Path) -> None:
    adapter = HifiSurrogateAdapter(
        model_dir=tmp_path / "missing_hifi",
        allow_heuristic_fallback=True,
        validation_mode="off",
    )
    pred, unc = adapter.predict([_candidate()])
    assert pred.shape == (1,)
    assert unc.shape == (1,)
    assert np.isfinite(pred[0])
    assert np.isfinite(unc[0])
