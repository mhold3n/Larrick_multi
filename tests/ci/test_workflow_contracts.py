"""Tests for workflow contract helpers."""

from __future__ import annotations

import json

import pytest

from larrak2.architecture.workflow_contracts import load_simulation_dataset_bundle


def test_load_simulation_dataset_bundle_round_trips(tmp_path) -> None:
    bundle_path = tmp_path / "simulation_dataset_bundle.json"
    bundle = {
        "simulation_api_version": "sim-api-v2",
        "artifacts": [{"name": "simulation-dataset-bundle", "path": "outputs/bundle.json"}],
    }
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    assert load_simulation_dataset_bundle(bundle_path) == bundle


def test_load_simulation_dataset_bundle_requires_version_field(tmp_path) -> None:
    bundle_path = tmp_path / "simulation_dataset_bundle.json"
    bundle_path.write_text(json.dumps({"artifacts": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="simulation_api_version"):
        load_simulation_dataset_bundle(bundle_path)
