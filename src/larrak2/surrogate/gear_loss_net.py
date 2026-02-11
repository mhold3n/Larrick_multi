from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import joblib
except ImportError:
    joblib = None


class GearLossNetwork(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, output_dim: int = 3):
        """
        Inputs:
        1. RPM
        2. Torque
        3. Base Radius
        4-8. Fourier Coeffs (C0..C4)
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),  # Outputs: Mesh, Bearing, Churning (or others)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GearSurrogate:
    """Runtime wrapper for Gear Loss Surrogate."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

        if joblib is None:
            raise ImportError("joblib is required for GearSurrogate")

        # Load Scalers
        self.scaler_X = joblib.load(self.model_dir / "scaler_X.pkl")
        self.scaler_y = joblib.load(self.model_dir / "scaler_y.pkl")

        # Load Model
        self.model = GearLossNetwork(input_dim=8, hidden_dim=64)
        state_dict = torch.load(self.model_dir / "best_model.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(
        self, rpm: float, torque: float, base_radius: float, coeffs: list[float]
    ) -> dict[str, float]:
        """Predict losses for single operating point."""
        # Ensure coeffs is length 5
        c_pad = list(coeffs) + [0.0] * (5 - len(coeffs))
        c_pad = c_pad[:5]

        row = np.array([[rpm, torque, base_radius, *c_pad]], dtype=np.float32)

        # Scale Input
        x_scaled = self.scaler_X.transform(row)

        # Predict
        with torch.no_grad():
            y_scaled = self.model(torch.tensor(x_scaled)).numpy()

        # Scale Output
        y = self.scaler_y.inverse_transform(y_scaled)[0]

        # Return Dict
        return {
            "loss_mesh": float(max(0.0, y[0])),
            "loss_bearing": float(max(0.0, y[1])),
            "loss_churning": float(max(0.0, y[2])),
            "loss_total": float(max(0.0, np.sum(y))),
        }


# Singleton
_GEAR_SURROGATE: GearSurrogate | None = None


def get_gear_surrogate(model_dir: str | Path) -> GearSurrogate:
    """Load and cache Gear Surrogate."""
    global _GEAR_SURROGATE
    if _GEAR_SURROGATE is None:
        _GEAR_SURROGATE = GearSurrogate(model_dir)
    return _GEAR_SURROGATE
