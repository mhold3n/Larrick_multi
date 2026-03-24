"""Training CLI validation-preflight gate tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from larrak2.cli.train import main as train_main


def test_train_cli_runs_validation_preflight_before_training(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def _fake_preflight(regime_or_suite: str, *, config_path: str, outdir: str) -> int:
        captured["regime"] = regime_or_suite
        captured["config_path"] = config_path
        captured["outdir"] = outdir
        return 0

    def _fake_train(args) -> None:
        captured["trained_model_type"] = args.model_type
        captured["train_data"] = args.data

    monkeypatch.setattr("larrak2.cli.train.train_openfoam_workflow", _fake_train)
    monkeypatch.setattr(
        "larrak2.cli.validate_simulation.run_validation_preflight",
        _fake_preflight,
    )

    config_path = tmp_path / "chemistry_config.json"
    config_path.write_text("{}", encoding="utf-8")
    data_path = tmp_path / "openfoam_data.npz"
    data_path.write_bytes(b"placeholder")

    with patch(
        "sys.argv",
        [
            "train.py",
            "openfoam",
            "--data",
            str(data_path),
            "--validation-regime",
            "chemistry",
            "--validation-config",
            str(config_path),
        ],
    ):
        train_main()

    assert captured["regime"] == "chemistry"
    assert captured["config_path"] == str(config_path)
    assert captured["outdir"].endswith("outputs/validation/pretrain/openfoam")
    assert captured["trained_model_type"] == "openfoam"
    assert captured["train_data"] == str(data_path)


def test_train_cli_blocks_training_when_validation_fails(monkeypatch, tmp_path: Path) -> None:
    called = {"train": False}

    def _fake_preflight(regime_or_suite: str, *, config_path: str, outdir: str) -> int:
        _ = regime_or_suite, config_path, outdir
        return 1

    def _fake_train(args) -> None:
        _ = args
        called["train"] = True

    monkeypatch.setattr("larrak2.cli.train.train_openfoam_workflow", _fake_train)
    monkeypatch.setattr(
        "larrak2.cli.validate_simulation.run_validation_preflight",
        _fake_preflight,
    )

    config_path = tmp_path / "chemistry_config.json"
    config_path.write_text("{}", encoding="utf-8")
    data_path = tmp_path / "openfoam_data.npz"
    data_path.write_bytes(b"placeholder")

    with patch(
        "sys.argv",
        [
            "train.py",
            "openfoam",
            "--data",
            str(data_path),
            "--validation-regime",
            "chemistry",
            "--validation-config",
            str(config_path),
        ],
    ):
        with pytest.raises(
            RuntimeError, match="Training blocked by simulation validation preflight"
        ):
            train_main()

    assert called["train"] is False


def test_train_cli_requires_both_validation_args(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("larrak2.cli.train.train_openfoam_workflow", lambda args: None)

    data_path = tmp_path / "openfoam_data.npz"
    data_path.write_bytes(b"placeholder")

    with patch(
        "sys.argv",
        [
            "train.py",
            "openfoam",
            "--data",
            str(data_path),
            "--validation-regime",
            "chemistry",
        ],
    ):
        with pytest.raises(
            ValueError, match="requires both --validation-regime and --validation-config"
        ):
            train_main()
