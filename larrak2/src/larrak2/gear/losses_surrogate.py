"""Surrogate model for gear losses.

Placeholder for trained surrogate models (neural network, GP, etc.)
that can predict losses faster than physics-based models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SurrogateModel:
    """Placeholder surrogate model container."""

    name: str
    version: str
    _weights: Any = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict loss from input features.

        Args:
            x: Feature array of shape (n_samples, n_features).

        Returns:
            Predicted loss array of shape (n_samples,).
        """
        # Placeholder: return zeros
        n_samples = x.shape[0] if x.ndim > 1 else 1
        return np.zeros(n_samples, dtype=np.float64)

    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._weights is not None


def load_surrogate(path: str) -> SurrogateModel:
    """Load surrogate model from file.

    Args:
        path: Path to saved model.

    Returns:
        SurrogateModel instance.

    Raises:
        NotImplementedError: Surrogate loading not yet implemented.
    """
    raise NotImplementedError("Surrogate loading not yet implemented")


def create_placeholder_surrogate() -> SurrogateModel:
    """Create placeholder surrogate for testing.

    Returns:
        Untrained SurrogateModel.
    """
    return SurrogateModel(name="placeholder", version="0.0.1")
