"""Trust-region controller for surrogate-guided refinement."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class TrustRegionConfig:
    """Trust-region update settings."""

    initial_radius: float = 0.1
    min_radius: float = 0.01
    max_radius: float = 0.5
    expand_factor: float = 1.5
    shrink_factor: float = 0.5
    uncertainty_weight: float = 1.0
    good_agreement_threshold: float = 0.1
    bad_agreement_threshold: float = 0.3


class TrustRegion:
    """Bounds solver steps based on uncertainty and prediction agreement."""

    def __init__(self, config: TrustRegionConfig | None = None, n_vars: int | None = None) -> None:
        self.config = config or TrustRegionConfig()
        self.n_vars = n_vars
        self._radius: float | np.ndarray
        if n_vars is None:
            self._radius = float(self.config.initial_radius)
        else:
            self._radius = np.ones(int(n_vars), dtype=np.float64) * float(self.config.initial_radius)
        self._history: list[dict[str, Any]] = []

    @property
    def radius(self) -> float | np.ndarray:
        return self._radius

    def bound_step(
        self,
        proposed_step: np.ndarray,
        uncertainty: float | np.ndarray,
        variable_scales: np.ndarray | None = None,
    ) -> np.ndarray:
        step = np.asarray(proposed_step, dtype=np.float64)
        if variable_scales is None:
            variable_scales = np.ones_like(step, dtype=np.float64)
        else:
            variable_scales = np.asarray(variable_scales, dtype=np.float64)
            if variable_scales.shape != step.shape:
                raise ValueError(
                    "variable_scales shape mismatch: "
                    f"{variable_scales.shape} vs {step.shape}"
                )

        unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
        if unc.size == 0:
            unc_eff = np.zeros_like(step)
        elif unc.size == 1:
            unc_eff = np.ones_like(step) * float(unc[0])
        elif unc.size == step.size:
            unc_eff = unc.reshape(step.shape)
        else:
            unc_eff = np.ones_like(step) * float(np.mean(unc))

        uncertainty_factor = 1.0 / (1.0 + float(self.config.uncertainty_weight) * unc_eff)
        if isinstance(self._radius, np.ndarray):
            radius_eff = self._radius * uncertainty_factor * variable_scales
        else:
            radius_eff = float(self._radius) * uncertainty_factor * variable_scales

        magnitude = np.abs(step)
        scale = np.minimum(1.0, radius_eff / np.maximum(magnitude, 1e-12))
        bounded = step * scale
        if np.any(scale < 0.5):
            LOGGER.debug("Trust region clipped %d variables by >50%%", int(np.sum(scale < 0.5)))
        return bounded

    def update(
        self,
        predicted_improvement: float,
        actual_improvement: float,
        uncertainty_at_step: float | np.ndarray,
    ) -> None:
        pred = float(predicted_improvement)
        act = float(actual_improvement)

        if abs(pred) <= 1e-12:
            agreement = 0.0 if abs(act) <= 1e-12 else 1.0
        else:
            agreement = abs(act - pred) / abs(pred)

        mean_radius = float(np.mean(self._radius)) if isinstance(self._radius, np.ndarray) else float(self._radius)
        self._history.append(
            {
                "predicted": pred,
                "actual": act,
                "agreement": float(agreement),
                "radius_before": mean_radius,
                "uncertainty": float(np.mean(np.asarray(uncertainty_at_step, dtype=np.float64))),
            }
        )

        if agreement < float(self.config.good_agreement_threshold):
            self._expand()
        elif agreement > float(self.config.bad_agreement_threshold):
            self._shrink()

    def _expand(self) -> None:
        if isinstance(self._radius, np.ndarray):
            self._radius = np.minimum(
                self._radius * float(self.config.expand_factor),
                float(self.config.max_radius),
            )
        else:
            self._radius = min(
                float(self._radius) * float(self.config.expand_factor),
                float(self.config.max_radius),
            )

    def _shrink(self) -> None:
        if isinstance(self._radius, np.ndarray):
            self._radius = np.maximum(
                self._radius * float(self.config.shrink_factor),
                float(self.config.min_radius),
            )
        else:
            self._radius = max(
                float(self._radius) * float(self.config.shrink_factor),
                float(self.config.min_radius),
            )

    def conservative_bounds(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        uncertainty: np.ndarray,
        safety_margin: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(lb, dtype=np.float64)
        upper = np.asarray(ub, dtype=np.float64)
        unc = np.asarray(uncertainty, dtype=np.float64)
        if lower.shape != upper.shape:
            raise ValueError("lb and ub shapes must match")
        if unc.size == 1:
            unc = np.ones_like(lower) * float(unc.reshape(-1)[0])
        elif unc.shape != lower.shape:
            unc = np.ones_like(lower) * float(np.mean(unc))

        margin = float(safety_margin) * (1.0 + unc)
        width = upper - lower

        clb = lower + margin * width
        cub = upper - margin * width
        mid = (lower + upper) * 0.5
        clb = np.minimum(clb, mid - 1e-6)
        cub = np.maximum(cub, mid + 1e-6)
        return clb, cub

    def get_statistics(self) -> dict[str, Any]:
        if not self._history:
            return {
                "n_updates": 0,
                "current_radius": float(np.mean(self._radius))
                if isinstance(self._radius, np.ndarray)
                else float(self._radius),
            }
        agreements = np.array([h["agreement"] for h in self._history], dtype=np.float64)
        return {
            "n_updates": int(len(self._history)),
            "mean_agreement": float(np.mean(agreements)),
            "current_radius": float(np.mean(self._radius))
            if isinstance(self._radius, np.ndarray)
            else float(self._radius),
        }


__all__ = [
    "TrustRegion",
    "TrustRegionConfig",
]

