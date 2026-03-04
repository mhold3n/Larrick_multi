"""Fidelity-aware artifact path resolution tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from larrak2.core.artifact_paths import (
    resolve_stack_artifact_path,
    resolve_thermo_symbolic_artifact_path,
    stack_artifact_path_for_fidelity,
    thermo_symbolic_artifact_path_for_fidelity,
)


def test_canonical_artifact_paths_by_fidelity() -> None:
    assert stack_artifact_path_for_fidelity(2) == Path(
        "outputs/artifacts/surrogates/stack_f2/stack_f2_surrogate.npz"
    )
    assert thermo_symbolic_artifact_path_for_fidelity(2) == Path(
        "outputs/artifacts/surrogates/thermo_symbolic_f2/thermo_symbolic_f2.npz"
    )


def test_stack_resolver_missing_includes_remediation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_stack_artifact_path(fidelity=2, must_exist=True)
    msg = str(excinfo.value)
    assert "train-stack-surrogate --fidelity 2" in msg
    assert "stack_f2_surrogate.npz" in msg


def test_thermo_resolver_missing_includes_remediation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_thermo_symbolic_artifact_path(fidelity=2, must_exist=True)
    msg = str(excinfo.value)
    assert "train-thermo-symbolic --fidelity 2" in msg
    assert "thermo_symbolic_f2.npz" in msg


def test_thermo_resolver_legacy_f1_fallback_warns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    legacy = Path("outputs/artifacts/surrogates/thermo_symbolic/thermo_symbolic_f1.npz")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"legacy")

    with pytest.warns(UserWarning, match="deprecated legacy thermo symbolic artifact path"):
        resolved = resolve_thermo_symbolic_artifact_path(fidelity=1, must_exist=True)

    assert resolved == legacy


def test_stack_resolver_fidelity_mismatch_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    wrong = Path("outputs/artifacts/surrogates/stack_f1/stack_f1_surrogate.npz")
    wrong.parent.mkdir(parents=True, exist_ok=True)
    wrong.write_bytes(b"placeholder")

    with pytest.raises(ValueError) as excinfo:
        resolve_stack_artifact_path(fidelity=2, explicit_path=str(wrong), must_exist=True)
    msg = str(excinfo.value)
    assert "requested_fidelity=2" in msg
    assert "detected_fidelity=1" in msg
    assert "train-stack-surrogate --fidelity 2" in msg


def test_thermo_resolver_fidelity_mismatch_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    wrong = Path("outputs/artifacts/surrogates/thermo_symbolic_f1/thermo_symbolic_f1.npz")
    wrong.parent.mkdir(parents=True, exist_ok=True)
    wrong.write_bytes(b"placeholder")

    with pytest.raises(ValueError) as excinfo:
        resolve_thermo_symbolic_artifact_path(fidelity=2, explicit_path=str(wrong), must_exist=True)
    msg = str(excinfo.value)
    assert "requested_fidelity=2" in msg
    assert "detected_fidelity=1" in msg
    assert "train-thermo-symbolic --fidelity 2" in msg
