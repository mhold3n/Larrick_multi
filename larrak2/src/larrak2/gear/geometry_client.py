"""Geometry client placeholder.

Client for external geometry evaluation services (e.g., CAD kernel, FEA).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeometryResult:
    """Result from geometry evaluation."""

    volume: float  # mm³
    surface_area: float  # mm²
    inertia_tensor: np.ndarray  # 3x3 (kg·mm²)
    center_of_mass: np.ndarray  # 3D position (mm)


class GeometryClient:
    """Client for geometry evaluation services."""

    def __init__(self, endpoint: str | None = None) -> None:
        """Initialize geometry client.

        Args:
            endpoint: Service endpoint URL (None for local stub).
        """
        self.endpoint = endpoint
        self._connected = False

    def connect(self) -> bool:
        """Connect to geometry service.

        Returns:
            True if connection successful.
        """
        if self.endpoint is None:
            # Local stub mode
            self._connected = True
            return True

        # TODO: Implement actual connection
        raise NotImplementedError("Remote geometry service not yet implemented")

    def evaluate(self, mesh_data: bytes) -> GeometryResult:
        """Evaluate geometry from mesh data.

        Args:
            mesh_data: Serialized mesh (e.g., STL bytes).

        Returns:
            GeometryResult with computed properties.
        """
        if not self._connected:
            self.connect()

        # Placeholder response
        return GeometryResult(
            volume=1000.0,
            surface_area=600.0,
            inertia_tensor=np.eye(3) * 100,
            center_of_mass=np.zeros(3),
        )

    def close(self) -> None:
        """Close connection."""
        self._connected = False
