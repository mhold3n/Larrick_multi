from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from larrak2.core.artifact_paths import (
    DEFAULT_MACHINING_NN_ARTIFACT,
    assert_not_legacy_models_path,
)
from larrak2.gear.manufacturability_limits import PROFILE_NAMES


class MachiningSurrogateNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


class MachiningInference:
    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_MACHINING_NN_ARTIFACT,
        mode: str = "nn",
    ) -> None:
        self.model: MachiningSurrogateNet | None = None
        self.mode = str(mode)
        if self.mode not in {"nn", "analytical"}:
            raise ValueError(f"machining mode must be 'nn' or 'analytical', got {self.mode!r}")
        self.model_path = Path(
            assert_not_legacy_models_path(model_path, purpose="Machining NN artifact")
        )
        self.shape_map = {name: i for i, name in enumerate(PROFILE_NAMES)}
        self.input_dim = 2 + len(PROFILE_NAMES)

    def load(self) -> None:
        if self.mode != "nn" or self.model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Machining NN surrogate is required in mode='nn' but missing: "
                f"'{self.model_path}'. Move/train artifact to outputs/artifacts/... or "
                "set mode='analytical' explicitly for a physics-only bypass."
            )

        model = MachiningSurrogateNet(self.input_dim)
        try:
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load machining surrogate from '{self.model_path}'"
            ) from exc
        self.model = model

    def _predict_analytical(
        self,
        duration_deg: float,
        amplitude: float,
        shape_name: str,  # noqa: ARG002
    ) -> tuple[float, float, float, float]:
        dur = float(max(duration_deg, 1.0))
        amp = float(abs(amplitude))
        # Conservative analytical proxy for production fail-hard fallback mode.
        t_min = max(0.25, 0.08 + 0.0011 * dur + 0.06 * amp)
        b_max = max(0.20, 0.10 + 0.0007 * dur + 0.04 * amp)
        hole_d = max(0.30, 0.12 + 0.0009 * dur + 0.03 * amp)
        hole_c = max(0.40, 0.18 + 0.0015 * dur + 0.05 * amp)
        return float(t_min), float(b_max), float(hole_d), float(hole_c)

    def predict(self, duration_deg: float, amplitude: float, shape_name: str):
        if self.mode == "analytical":
            return self._predict_analytical(duration_deg, amplitude, shape_name)

        self.load()
        assert self.model is not None

        norm_dur = float(duration_deg) / 360.0
        norm_amp = (float(amplitude) + 1.5) / 5.5

        shape_idx = int(self.shape_map.get(shape_name, 0))
        one_hot = [0.0] * len(PROFILE_NAMES)
        if 0 <= shape_idx < len(one_hot):
            one_hot[shape_idx] = 1.0

        x = torch.tensor([[norm_dur, norm_amp] + one_hot], dtype=torch.float32)
        with torch.no_grad():
            y = self.model(x).numpy()[0]

        return float(y[0]), float(y[1]), float(y[2]), float(y[3])


_ENGINE_CACHE: dict[tuple[str, str], MachiningInference] = {}


def get_machining_engine(
    *,
    mode: str = "nn",
    model_path: str | Path | None = None,
) -> MachiningInference:
    resolved_model_path = (
        str(DEFAULT_MACHINING_NN_ARTIFACT) if model_path is None else str(model_path)
    )
    key = (str(mode), resolved_model_path)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = MachiningInference(
            mode=str(mode),
            model_path=resolved_model_path,
        )
    return _ENGINE_CACHE[key]
