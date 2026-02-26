"""Surrogate adapter for initialization heuristics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

log = logging.getLogger(__name__)


class VoxelSurrogateAdapter:
    """Load and evaluate fast geometric surrogate models."""

    MODEL_FILES = {
        "scavenge_area": "scavenge_area.joblib",
        "exhaust_area": "exhaust_area.joblib",
        "geometric_cr": "geometric_cr.joblib",
        "stroke_check": "stroke_check.joblib",
    }

    def __init__(self, model_dir: str | Path = "data/models/surrogate") -> None:
        self.model_dir = Path(model_dir)
        self.models: dict[str, Any] = {}
        self._load_models()

    def _load_models(self) -> None:
        if not self.model_dir.exists():
            log.warning("Surrogate model directory not found: %s", self.model_dir)
            return

        for key, filename in self.MODEL_FILES.items():
            path = self.model_dir / filename
            if not path.exists():
                continue
            try:
                self.models[key] = joblib.load(path)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Failed to load model %s from %s: %s", key, path, exc)

    @staticmethod
    def _as_feature_array(
        bore: float,
        stroke: float,
        con_rod_length: float,
        cycle_time: float,
    ) -> np.ndarray:
        return np.array([[bore, stroke, con_rod_length, cycle_time]], dtype=np.float64)

    def _predict_or_default(self, key: str, features: np.ndarray, default: float) -> float:
        model = self.models.get(key)
        if model is None:
            return float(default)

        # Try direct ndarray prediction first. Fall back to dict-record style when required.
        try:
            value = model.predict(features)[0]
            return float(value)
        except Exception:
            feature_dicts = [
                {
                    "bore": float(features[0, 0]),
                    "stroke": float(features[0, 1]),
                    "con_rod_length": float(features[0, 2]),
                    "cycle_time": float(features[0, 3]),
                }
            ]
            value = model.predict(feature_dicts)[0]
            return float(value)

    def predict(
        self,
        bore: float,
        stroke: float,
        con_rod_length: float,
        cycle_time: float,
    ) -> dict[str, float]:
        """Predict surrogate metrics for one design point."""
        features = self._as_feature_array(bore, stroke, con_rod_length, cycle_time)

        return {
            "mean_scavenge_area": self._predict_or_default("scavenge_area", features, 0.0),
            "mean_exhaust_area": self._predict_or_default("exhaust_area", features, 0.0),
            "geometric_cr": self._predict_or_default("geometric_cr", features, 10.5),
        }
