"""Problem specification types for CasADi/Ipopt refinement workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OptimizationObjective(Enum):
    """Supported objective categories."""

    MINIMIZE_JERK = "minimize_jerk"
    MAXIMIZE_THERMAL_EFFICIENCY = "maximize_thermal_efficiency"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_ENERGY = "minimize_energy"
    SMOOTHNESS = "smoothness"


class CollocationMethod(Enum):
    """Supported collocation methods."""

    LEGENDRE = "legendre"
    RADAU = "radau"
    TRAPEZOIDAL = "trapezoidal"


@dataclass
class CasADiMotionProblem:
    """Clean problem specification for local CasADi optimization."""

    stroke: float
    cycle_time: float
    upstroke_percent: float
    duration_angle_deg: float

    max_velocity: float | None = None
    max_acceleration: float | None = None
    max_jerk: float | None = None
    compression_ratio_limits: tuple[float, float] = (20.0, 70.0)

    objectives: list[OptimizationObjective] = field(
        default_factory=lambda: [
            OptimizationObjective.MINIMIZE_JERK,
            OptimizationObjective.MAXIMIZE_THERMAL_EFFICIENCY,
        ]
    )
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "jerk": 1.0,
            "thermal_efficiency": 0.1,
            "smoothness": 0.01,
            "time": 0.0,
            "energy": 0.0,
        }
    )

    n_segments: int = 50
    poly_order: int = 3
    collocation_method: CollocationMethod = CollocationMethod.LEGENDRE

    solver_options: dict[str, Any] = field(
        default_factory=lambda: {
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-6,
            "ipopt.print_level": 0,
            "ipopt.warm_start_init_point": "yes",
        }
    )

    thermal_efficiency_target: float = 0.55
    heat_transfer_coeff: float = 0.1
    friction_coeff: float = 0.01

    def __post_init__(self) -> None:
        self._validate_parameters()
        self._normalize_weights()

    def _validate_parameters(self) -> None:
        if self.stroke <= 0:
            raise ValueError("stroke must be positive")
        if self.cycle_time <= 0:
            raise ValueError("cycle_time must be positive")
        if self.duration_angle_deg <= 0:
            raise ValueError("duration_angle_deg must be positive")
        if not 0 < self.upstroke_percent < 100:
            raise ValueError("upstroke_percent must be in (0, 100)")

        if self.max_velocity is not None and self.max_velocity <= 0:
            raise ValueError("max_velocity must be positive")
        if self.max_acceleration is not None and self.max_acceleration <= 0:
            raise ValueError("max_acceleration must be positive")
        if self.max_jerk is not None and self.max_jerk <= 0:
            raise ValueError("max_jerk must be positive")

        lo, hi = self.compression_ratio_limits
        if lo >= hi:
            raise ValueError("compression_ratio_limits must be ordered")

        if self.n_segments <= 0:
            raise ValueError("n_segments must be positive")
        if self.poly_order < 1:
            raise ValueError("poly_order must be >= 1")

        if not 0 < self.thermal_efficiency_target < 1:
            raise ValueError("thermal_efficiency_target must be in (0, 1)")

    def _normalize_weights(self) -> None:
        total = float(sum(self.weights.values()))
        if total > 0:
            self.weights = {k: float(v) / total for k, v in self.weights.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "stroke": self.stroke,
            "cycle_time": self.cycle_time,
            "duration_angle_deg": self.duration_angle_deg,
            "upstroke_percent": self.upstroke_percent,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "max_jerk": self.max_jerk,
            "compression_ratio_limits": self.compression_ratio_limits,
            "objectives": [obj.value for obj in self.objectives],
            "weights": dict(self.weights),
            "n_segments": self.n_segments,
            "poly_order": self.poly_order,
            "collocation_method": self.collocation_method.value,
            "solver_options": dict(self.solver_options),
            "thermal_efficiency_target": self.thermal_efficiency_target,
            "heat_transfer_coeff": self.heat_transfer_coeff,
            "friction_coeff": self.friction_coeff,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CasADiMotionProblem":
        duration_angle_deg = data.get("duration_angle_deg")
        if duration_angle_deg is None:
            raise ValueError("duration_angle_deg is required")

        objectives = [OptimizationObjective(obj) for obj in data.get("objectives", [])]
        collocation_method = CollocationMethod(data.get("collocation_method", "legendre"))

        return cls(
            stroke=float(data["stroke"]),
            cycle_time=float(data["cycle_time"]),
            duration_angle_deg=float(duration_angle_deg),
            upstroke_percent=float(data["upstroke_percent"]),
            max_velocity=data.get("max_velocity"),
            max_acceleration=data.get("max_acceleration"),
            max_jerk=data.get("max_jerk"),
            compression_ratio_limits=tuple(data.get("compression_ratio_limits", (20.0, 70.0))),
            objectives=objectives
            if objectives
            else [
                OptimizationObjective.MINIMIZE_JERK,
                OptimizationObjective.MAXIMIZE_THERMAL_EFFICIENCY,
            ],
            weights=data.get("weights", {}),
            n_segments=int(data.get("n_segments", 50)),
            poly_order=int(data.get("poly_order", 3)),
            collocation_method=collocation_method,
            solver_options=data.get("solver_options", {}),
            thermal_efficiency_target=float(data.get("thermal_efficiency_target", 0.55)),
            heat_transfer_coeff=float(data.get("heat_transfer_coeff", 0.1)),
            friction_coeff=float(data.get("friction_coeff", 0.01)),
        )

    def update_weights(self, **weights: float) -> None:
        self.weights.update({k: float(v) for k, v in weights.items()})
        self._normalize_weights()

    def add_objective(self, objective: OptimizationObjective, weight: float = 1.0) -> None:
        if objective not in self.objectives:
            self.objectives.append(objective)
        self.weights[objective.value] = float(weight)
        self._normalize_weights()

    def remove_objective(self, objective: OptimizationObjective) -> None:
        if objective in self.objectives:
            self.objectives.remove(objective)
        self.weights.pop(objective.value, None)
        self._normalize_weights()
