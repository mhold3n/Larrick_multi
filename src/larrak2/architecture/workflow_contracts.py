"""Workflow contract helpers for shared simulation dataset bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_simulation_dataset_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("simulation dataset bundle must decode to a JSON object")

    version = payload.get("simulation_api_version") or payload.get("bundle_schema_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            "simulation dataset bundle must declare 'simulation_api_version' or 'bundle_schema_version'"
        )

    artifacts = payload.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, list):
        raise ValueError("simulation dataset bundle 'artifacts' must be a list when present")

    return payload
