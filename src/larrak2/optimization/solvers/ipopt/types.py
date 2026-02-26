"""Shared IPOPT dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class IPOPTOptions:
    """Normalized IPOPT option set used by larrak2."""

    max_iter: int = 500
    tol: float = 1e-6
    print_level: int = 0
    linear_solver: str = "mumps"
    warm_start_init_point: str = "yes"
    hessian_approximation: str = "limited-memory"
    jacobian_approximation: str = "finite-difference-values"
    acceptable_tol: float = 1e-4
    acceptable_iter: int = 5
    extra: dict[str, Any] = field(default_factory=dict)

    def as_ipopt_dict(self) -> dict[str, Any]:
        base = {
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "print_level": int(self.print_level),
            "linear_solver": str(self.linear_solver),
            "warm_start_init_point": str(self.warm_start_init_point),
            "hessian_approximation": str(self.hessian_approximation),
            "jacobian_approximation": str(self.jacobian_approximation),
            "acceptable_tol": float(self.acceptable_tol),
            "acceptable_iter": int(self.acceptable_iter),
        }
        base.update(self.extra)
        return base


@dataclass
class IPOPTResult:
    """Normalized IPOPT solve result."""

    x_opt: np.ndarray
    f_opt: float
    g_opt: np.ndarray
    success: bool
    status: str
    iterations: int
    cpu_time_s: float
    stats: dict[str, Any] = field(default_factory=dict)
