"""Hard guard tests for artifact path policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from larrak2.artifacts import planned_model_layout
from larrak2.core.artifact_paths import assert_not_legacy_models_path


def test_runtime_guard_rejects_models_write_paths() -> None:
    with pytest.raises(ValueError):
        assert_not_legacy_models_path("models/surrogate_v1", purpose="test artifact")
    with pytest.raises(ValueError):
        assert_not_legacy_models_path(
            "src/larrak2/surrogate/machining_surrogate.pth",
            purpose="test artifact",
        )

    p = assert_not_legacy_models_path(
        "outputs/artifacts/surrogates/v1_gbr",
        purpose="test artifact",
    )
    assert str(p).startswith("outputs/")


def test_cli_defaults_do_not_target_models_folder() -> None:
    src_root = Path("src/larrak2")
    allow = {
        src_root / "core" / "artifact_paths.py",
    }
    banned_tokens = ['default="models', "default='models", 'Path("models', "Path('models"]

    offenders: list[str] = []
    for py in src_root.rglob("*.py"):
        if py in allow:
            continue
        text = py.read_text(encoding="utf-8")
        if any(token in text for token in banned_tokens):
            offenders.append(str(py))

    assert not offenders, f"legacy models defaults detected in: {offenders}"


def test_model_layout_policy_is_explicit_and_canonical() -> None:
    specs = planned_model_layout()
    keys = {s.key for s in specs}
    assert {"openfoam_nn", "calculix_nn", "v1_gbr", "hifi", "initialization_voxel"} <= keys
    for spec in specs:
        assert str(spec.canonical).startswith("outputs/")
        assert spec.purpose


def test_no_runtime_legacy_fallback_markers_in_src() -> None:
    src_root = Path("src/larrak2")
    banned_markers = ["LEGACY_", "fallback_paths=["]
    allow = {
        src_root / "core" / "artifact_paths.py",  # contains deprecated-path guard
    }

    offenders: list[str] = []
    for py in src_root.rglob("*.py"):
        if py in allow:
            continue
        text = py.read_text(encoding="utf-8")
        if any(marker in text for marker in banned_markers):
            offenders.append(str(py))

    assert not offenders, f"legacy fallback markers detected: {offenders}"
