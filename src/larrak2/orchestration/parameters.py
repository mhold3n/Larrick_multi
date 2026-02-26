"""Parameter extraction helpers for orchestration adapters."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


class ParameterMapper:
    """Extracts numeric features from nested candidate dictionaries."""

    @staticmethod
    def extract_features(candidate: dict[str, Any], keys: list[str]) -> list[float]:
        features: list[float] = []
        for key in keys:
            value = ParameterMapper._get_value(candidate, key)
            try:
                features.append(float(value))
            except (TypeError, ValueError):
                LOGGER.warning("Feature '%s' is non-numeric (%r); using 0.0", key, value)
                features.append(0.0)
        return features

    @staticmethod
    def _get_value(candidate: dict[str, Any], key: str) -> Any:
        if key in candidate:
            return candidate[key]

        for parent_key in ("geometry", "operating_point", "thermodynamics"):
            parent = candidate.get(parent_key)
            if isinstance(parent, dict) and key in parent:
                return parent[key]

        if key == "load_fraction":
            return ParameterMapper._derive_load_fraction(candidate)

        return 0.0

    @staticmethod
    def _derive_load_fraction(candidate: dict[str, Any]) -> float:
        if "p_intake_bar" in candidate:
            return float(candidate["p_intake_bar"]) / 4.0

        operating_point = candidate.get("operating_point")
        if isinstance(operating_point, dict) and "p_int" in operating_point:
            return float(operating_point["p_int"]) / 400000.0

        return 0.5


__all__ = ["ParameterMapper"]

