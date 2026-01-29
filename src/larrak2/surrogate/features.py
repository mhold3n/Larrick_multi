"""Feature extraction and schema validation for surrogates.

Ensures that the input features provided to a model match exactly
what the model was trained on via Strict Schema Hashing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeatureSchema:
    """Defines the inputs expected by a surrogate model."""

    feature_names: list[str]
    target_names: list[str]
    description: str = ""
    version: str = "v1"

    _hash: str = field(init=False, default="")

    def __post_init__(self):
        self._hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute deterministic hash of the schema."""
        # Stabilize ordering
        data = {
            "features": sorted(self.feature_names),  # Sort or keep order? Order matters for arrays!
            # Actually order matters for input vectors. We must NOT sort if order implies index.
            # But "names" usually implies order.
            "features_ordered": self.feature_names,
            "targets": self.target_names,
            "version": self.version,
        }
        s = json.dumps(data, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def validate(self, other_hash: str) -> bool:
        return self._hash == other_hash


# --- Specific Schemas ---


def get_gear_schema_v1() -> FeatureSchema:
    """Schema for Gear Surrogate V1."""
    # Inputs: 8 gear params (base_radius + 7 coeffs)
    feats = [
        "gear_base_radius",
        "gear_c1",
        "gear_c2",
        "gear_c3",
        "gear_c4",
        "gear_c5",
        "gear_c6",
        "gear_c7",
    ]
    targets = ["delta_loss"]
    return FeatureSchema(
        feature_names=feats, target_names=targets, description="Gear Loss Residual"
    )


def get_scavenge_schema_v1() -> FeatureSchema:
    """Schema for Scavenging Surrogate V1."""
    # Inputs: Thermo params (4) + maybe others?
    # Thermo: compression, expansion, hr_center, hr_width
    feats = [
        "thermo_compression_ratio",
        "thermo_expansion_ratio",
        "thermo_hr_center",
        "thermo_hr_width",
    ]
    targets = ["delta_efficiency"]
    return FeatureSchema(
        feature_names=feats, target_names=targets, description="Scavenge Efficiency Residual"
    )


# --- Extractors ---


def extract_gear_features_v1(x: np.ndarray) -> np.ndarray:
    """Extract features for Gear model from full decision vector."""
    # Assuming standard encoding: [Thermo(4), Gear(8)]
    # Gear is indices 4:12 (inclusive 4, exclusive 12)
    # Validate size?
    if len(x) < 12:
        raise ValueError(f"Input vector too small for V1 schema: {len(x)}")
    return x[4:12]


def extract_scavenge_features_v1(x: np.ndarray) -> np.ndarray:
    """Extract features for Scavenge model."""
    # Thermo is 0:4
    return x[0:4]
