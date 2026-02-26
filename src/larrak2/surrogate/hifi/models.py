"""HiFi surrogate model definitions."""

from __future__ import annotations

import numpy as np
import torch

from .ensemble import EnsembleSurrogate


class ThermalSurrogate(EnsembleSurrogate):
    """Ensemble surrogate for piston crown temperature."""

    T_MIN = 350.0
    T_MAX = 700.0

    def __init__(
        self,
        n_models: int = 5,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__(
            n_models=n_models,
            input_dim=5,
            output_dim=1,
            hidden_dims=hidden_dims or [64, 64, 64],
            output_bounds={0: ("sigmoid", 0.0, 1.0)},
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_norm, std_norm = super().forward(x)
        span = self.T_MAX - self.T_MIN
        return mean_norm * span + self.T_MIN, std_norm * span

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32)
            mean, std = self.forward(x)
            return mean.numpy(), std.numpy()

    def is_feasible(
        self,
        inputs: np.ndarray,
        T_limit: float = 620.0,
        confidence: float = 2.0,
    ) -> tuple[bool, float, float]:
        mean, std = self.predict(inputs)
        T_pred = float(mean[0, 0])
        T_std = float(std[0, 0])
        return (T_pred + confidence * T_std) < T_limit, T_pred, T_std


class FlowCoefficientSurrogate(EnsembleSurrogate):
    """Ensemble surrogate for discharge coefficient prediction."""

    CD_MIN = 0.3
    CD_MAX = 0.8

    def __init__(
        self,
        n_models: int = 5,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__(
            n_models=n_models,
            input_dim=5,
            output_dim=1,
            hidden_dims=hidden_dims or [64, 64, 64],
            output_bounds={0: ("sigmoid", 0.0, 1.0)},
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_norm, std_norm = super().forward(x)
        span = self.CD_MAX - self.CD_MIN
        return mean_norm * span + self.CD_MIN, std_norm * span

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32)
            mean, std = self.forward(x)
            return mean.numpy(), std.numpy()


class StructuralSurrogate(EnsembleSurrogate):
    """Ensemble surrogate for peak stress prediction."""

    STRESS_MIN = 0.0
    STRESS_MAX = 400.0

    def __init__(
        self,
        n_models: int = 5,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__(
            n_models=n_models,
            input_dim=5,
            output_dim=1,
            hidden_dims=hidden_dims or [64, 64, 64],
            output_bounds={0: ("sigmoid", 0.0, 1.0)},
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_norm, std_norm = super().forward(x)
        span = self.STRESS_MAX - self.STRESS_MIN
        return mean_norm * span + self.STRESS_MIN, std_norm * span

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32)
            mean, std = self.forward(x)
            return mean.numpy(), std.numpy()

    def is_feasible(
        self,
        inputs: np.ndarray,
        yield_strength: float = 280.0,
        safety_factor: float = 1.5,
        confidence: float = 2.0,
    ) -> tuple[bool, float, float]:
        mean, std = self.predict(inputs)
        pred = float(mean[0, 0])
        unc = float(std[0, 0])
        allowable = yield_strength / safety_factor
        return (pred + confidence * unc) < allowable, pred, unc
