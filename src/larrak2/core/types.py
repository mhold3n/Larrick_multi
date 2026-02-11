"""Core types for evaluation context and results.

This module defines the canonical types that form the interface
between physics models and optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BreathingConfig:
    """Operating-point breathing/BC/timing inputs for OpenFOAM NN inference.

    These are used only when the OpenFOAM NN surrogate is active (fidelity >= 2).
    """

    bore_mm: float = 80.0
    stroke_mm: float = 90.0
    intake_port_area_m2: float = 4.0e-4
    exhaust_port_area_m2: float = 4.0e-4
    p_manifold_Pa: float = 101325.0
    p_back_Pa: float = 101325.0
    overlap_deg: float = 0.0
    intake_open_deg: float = 0.0
    intake_close_deg: float = 0.0
    exhaust_open_deg: float = 0.0
    exhaust_close_deg: float = 0.0


@dataclass(frozen=True)
class EvalContext:
    """Context for candidate evaluation.

    Attributes:
        rpm: Engine speed (rev/min).
        torque: Torque demand (Nm).
        fidelity: Model fidelity level.
            0 = cheap/toy physics
            1 = mid fidelity
            2 = expensive/high-fidelity
        seed: Random seed for deterministic evaluation.
    """

    rpm: float
    torque: float
    fidelity: int = 0
    seed: int = 0
    breathing: BreathingConfig | None = None
    # Gear Loss Coefficients (Calibration Data)
    # Keys: "bearing" (4 floats), "churning" (2 floats), "mesh" (1 float mu)
    loss_coeffs: dict[str, tuple[float, ...]] | None = None
    gear_process_params: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.fidelity not in (0, 1, 2):
            raise ValueError(f"fidelity must be 0, 1, or 2, got {self.fidelity}")


@dataclass
class EvalResult:
    """Result from candidate evaluation.

    Attributes:
        F: Objective values (minimize). Shape: (n_obj,)
        G: Constraint values. Convention: G <= 0 is feasible. Shape: (n_constr,)
        diag: Diagnostics dictionary with sub-dicts for thermo/gear and timings.
    """

    F: np.ndarray
    G: np.ndarray
    diag: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce float64
        self.F = np.asarray(self.F, dtype=np.float64)
        self.G = np.asarray(self.G, dtype=np.float64)

    @property
    def is_feasible(self) -> bool:
        """Check if all constraints are satisfied (G <= 0)."""
        return bool(np.all(self.G <= 0))

    @property
    def max_violation(self) -> float:
        """Return maximum constraint violation (0 if feasible)."""
        return float(np.maximum(self.G, 0).max()) if len(self.G) > 0 else 0.0
