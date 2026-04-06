"""Surrogate Inference Engine.

Orchestrates loading and prediction for multiple surrogate models.
Enforces strict schema validation.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from larrak_runtime.core.artifact_paths import (
    DEFAULT_SURROGATE_V1_DIR,
    assert_not_legacy_models_path,
)
from larrak_runtime.surrogate.features import (
    extract_gear_features_v1,
    extract_scavenge_features_v1,
    get_gear_schema_v1,
    get_scavenge_schema_v1,
)


def _model_roots() -> list[Path]:
    roots: list[Path] = [DEFAULT_SURROGATE_V1_DIR]
    env_root = str(os.environ.get("LARRAK2_SURROGATE_V1_ROOT", "")).strip()
    if env_root:
        roots.insert(
            0,
            Path(
                assert_not_legacy_models_path(
                    env_root,
                    purpose="Surrogate v1 model root",
                )
            ),
        )

    dedup: list[Path] = []
    seen: set[str] = set()
    for p in roots:
        key = str(p)
        if key not in seen:
            dedup.append(p)
            seen.add(key)
    return dedup


def _get_model_path(key: str) -> Path | None:
    """Resolve model path by convention."""
    names = [f"model_{key}.pkl", f"{key}.pkl", f"{key}/model.pkl", f"{key}/best_model.pt"]
    for root in _model_roots():
        for name in names:
            p = root / name
            if p.exists():
                return p
    return None


class SurrogateEngine:
    """Manages lifecycle and prediction of surrogate models."""

    def __init__(self):
        self.models: dict[str, Any] = {}
        self.schemas = {
            "gear": get_gear_schema_v1(),
            "scavenge": get_scavenge_schema_v1(),
        }
        self.extractors = {
            "gear": extract_gear_features_v1,
            "scavenge": extract_scavenge_features_v1,
        }
        # Attempt to load known models
        self._load_model("gear")
        self._load_model("scavenge")

    def _load_model(self, key: str):
        """Load a specific model if available and valid."""
        path = _get_model_path(key)
        if not path:
            return

        try:
            with open(path, "rb") as f:
                artifact = pickle.load(f)

            model = None
            loaded_hash = ""

            # Case 1: Legacy Dict Artifact
            if isinstance(artifact, dict):
                if "model" in artifact and "schema_hash" in artifact:
                    model = artifact["model"]
                    loaded_hash = artifact["schema_hash"]
                else:
                    warnings.warn(f"Invalid dict artifact format for {key} at {path}")
                    return

            # Case 2: EnsembleRegressor Object
            elif hasattr(artifact, "schema_hash") and hasattr(artifact, "predict"):
                model = artifact
                loaded_hash = getattr(artifact, "schema_hash")  # type: ignore

            else:
                warnings.warn(f"Invalid artifact type for {key} at {path}: {type(artifact)}")
                return

            # Verify Schema
            expected_schema = self.schemas[key]

            if not expected_schema.validate(loaded_hash):
                warnings.warn(
                    f"Schema Mismatch for {key}: "
                    f"Expected {expected_schema._hash}, Got {loaded_hash}. "
                    "Ignoring model."
                )
                return

            self.models[key] = model
            print(f"Loaded surrogate: {key} (v{expected_schema.version})")

        except Exception as e:
            warnings.warn(f"Failed to load surrogate {key}: {e}")

    def predict_corrections(self, x: np.ndarray) -> tuple[float, float, dict]:
        """Predict corrections for efficiency and loss.

        Returns:
            (delta_eff, delta_loss, meta)
        """
        delta_eff = 0.0
        delta_loss = 0.0
        meta: dict[str, Any] = {"surrogates_active": [], "uncertainty": {}}

        # 1. Gear (Delta Loss)
        if "gear" in self.models:
            feats = self.extractors["gear"](x)
            feats_in = feats.reshape(1, -1)

            # Predict
            res = self.models["gear"].predict(feats_in)

            # Handle Ensemble vs Standard
            if isinstance(res, tuple) and len(res) == 2:
                pred, std = res
                d_loss = float(pred[0])
                metrics = {"val": d_loss, "std": float(std[0])}
            else:
                d_loss = float(res[0])
                metrics = {"val": d_loss, "std": 0.0}

            delta_loss += d_loss
            meta["surrogates_active"].append("gear")
            meta["delta_loss_gear"] = d_loss
            meta["uncertainty"]["gear"] = metrics["std"]

        # 2. Scavenge (Delta Efficiency)
        if "scavenge" in self.models:
            feats = self.extractors["scavenge"](x)
            feats_in = feats.reshape(1, -1)

            res = self.models["scavenge"].predict(feats_in)

            if isinstance(res, tuple) and len(res) == 2:
                pred, std = res
                d_eff = float(pred[0])
                metrics = {"val": d_eff, "std": float(std[0])}
            else:
                d_eff = float(res[0])
                metrics = {"val": d_eff, "std": 0.0}

            delta_eff += d_eff
            meta["surrogates_active"].append("scavenge")
            meta["delta_eff_scavenge"] = d_eff
            meta["uncertainty"]["scavenge"] = metrics["std"]

        return delta_eff, delta_loss, meta


# Singleton
_ENGINE = None


def get_surrogate_engine() -> SurrogateEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SurrogateEngine()
    return _ENGINE
