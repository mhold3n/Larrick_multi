"""Core types for evaluation context and results.

This module defines the canonical types that form the interface
between physics models and optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
