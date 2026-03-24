"""Validation dataset registry — source-aware and regime-aware metric storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ComparisonMode, SourceType, ValidationDatasetManifest, ValidationMetricSpec


class DatasetRegistryError(ValueError):
    """Raised when dataset registry validation fails."""


class DatasetRegistry:
    """Source-aware, regime-aware registry for validation datasets.

    Rules enforced:
    - Every synthetic target must record measured_anchor_ids and governing_basis.
    - Synthetic extension is allowed only after at least one measured dataset
      exists for that regime.
    """

    def __init__(self) -> None:
        self._datasets: dict[str, ValidationDatasetManifest] = {}

    @property
    def datasets(self) -> dict[str, ValidationDatasetManifest]:
        return dict(self._datasets)

    def register(self, dataset: ValidationDatasetManifest) -> None:
        """Register a dataset, enforcing provenance rules."""
        # Provenance check
        errors = dataset.validate_provenance()
        if errors:
            raise DatasetRegistryError(f"Provenance validation failed: {'; '.join(errors)}")

        # Synthetic extension requires existing measured dataset in same regime
        if dataset.source_type == SourceType.SYNTHETIC:
            has_measured = any(
                d.regime == dataset.regime and d.source_type == SourceType.MEASURED
                for d in self._datasets.values()
            )
            if not has_measured:
                raise DatasetRegistryError(
                    f"Cannot register synthetic dataset '{dataset.dataset_id}' "
                    f"for regime '{dataset.regime}': no measured dataset exists "
                    f"for this regime yet. Register at least one measured dataset first."
                )

        self._datasets[dataset.dataset_id] = dataset

    def get(self, dataset_id: str) -> ValidationDatasetManifest:
        if dataset_id not in self._datasets:
            raise KeyError(f"Dataset '{dataset_id}' not found in registry")
        return self._datasets[dataset_id]

    def by_regime(self, regime: str) -> list[ValidationDatasetManifest]:
        return [d for d in self._datasets.values() if d.regime == regime]

    def by_fuel_family(self, fuel_family: str) -> list[ValidationDatasetManifest]:
        return [d for d in self._datasets.values() if d.fuel_family == fuel_family]

    def measured_for_regime(self, regime: str) -> list[ValidationDatasetManifest]:
        return [
            d
            for d in self._datasets.values()
            if d.regime == regime and d.source_type == SourceType.MEASURED
        ]

    def has_measured_for_regime(self, regime: str) -> bool:
        return len(self.measured_for_regime(regime)) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "datasets": {
                did: {
                    "dataset_id": d.dataset_id,
                    "regime": d.regime,
                    "fuel_family": d.fuel_family,
                    "source_type": d.source_type.value,
                    "provenance": d.provenance,
                    "operating_bounds": d.operating_bounds,
                    "measured_anchor_ids": d.measured_anchor_ids,
                    "governing_basis": d.governing_basis,
                    "literature_reference": d.literature_reference,
                    "n_metrics": len(d.metrics),
                }
                for did, d in self._datasets.items()
            }
        }

    @classmethod
    def from_json(cls, path: str | Path) -> DatasetRegistry:
        """Load registry from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        registry = cls()
        for entry in data.get("datasets", []):
            metrics = []
            for m in entry.get("metrics", []):
                metrics.append(
                    ValidationMetricSpec(
                        metric_id=str(m["metric_id"]),
                        units=str(m.get("units", "")),
                        comparison_mode=ComparisonMode(m.get("comparison_mode", "absolute")),
                        tolerance_band=float(m.get("tolerance_band", 0.0)),
                        source_type=SourceType(m.get("source_type", "measured")),
                        required=bool(m.get("required", True)),
                        description=str(m.get("description", "")),
                    )
                )
            dataset = ValidationDatasetManifest(
                dataset_id=str(entry["dataset_id"]),
                regime=str(entry["regime"]),
                fuel_family=str(entry.get("fuel_family", "gasoline")),
                source_type=SourceType(entry.get("source_type", "measured")),
                provenance=dict(entry.get("provenance", {})),
                operating_bounds=dict(entry.get("operating_bounds", {})),
                metrics=metrics,
                measured_anchor_ids=list(entry.get("measured_anchor_ids", [])),
                governing_basis=str(entry.get("governing_basis", "")),
                literature_reference=str(entry.get("literature_reference", "")),
                standard_reference=str(entry.get("standard_reference", "")),
            )
            registry.register(dataset)
        return registry
