"""Ensemble neural surrogate primitives with uncertainty estimation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class BoundedMLP(nn.Module):
    """MLP with optional per-output transforms."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        output_bounds: dict[int, tuple[str, float, float]] | None = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64, 64]

        self.output_bounds = output_bounds or {}

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        if not self.output_bounds:
            return raw

        out_cols: list[torch.Tensor] = []
        for i in range(raw.shape[-1]):
            col = raw[..., i : i + 1]
            cfg = self.output_bounds.get(i)
            if cfg is None:
                out_cols.append(col)
                continue

            transform, lo, hi = cfg
            if transform == "sigmoid":
                col = torch.sigmoid(col) * (hi - lo) + lo
            elif transform == "softplus":
                col = nn.functional.softplus(col) + lo
            out_cols.append(col)

        return torch.cat(out_cols, dim=-1)


class EnsembleSurrogate(nn.Module):
    """Bagged ensemble of bounded MLPs."""

    def __init__(
        self,
        n_models: int,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        output_bounds: dict[int, tuple[str, float, float]] | None = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.n_models = int(n_models)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = hidden_dims or [64, 64, 64]
        self.output_bounds = output_bounds
        self.dropout_rate = float(dropout_rate)

        self.models = nn.ModuleList(
            [
                BoundedMLP(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    hidden_dims=self.hidden_dims,
                    output_bounds=self.output_bounds,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(self.n_models)
            ]
        )

        self._uncertainty_threshold: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preds = torch.stack([m(x) for m in self.models], dim=0)
        return preds.mean(dim=0), preds.std(dim=0)

    def get_model(self, idx: int) -> BoundedMLP:
        return self.models[int(idx)]

    def predict_with_samples(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.models], dim=0)

    def reject_threshold(
        self, uncertainty: torch.Tensor, threshold: torch.Tensor | None = None
    ) -> torch.Tensor:
        thresh = threshold if threshold is not None else self._uncertainty_threshold
        if thresh is None:
            thresh = uncertainty.mean() * 2.0
        return (uncertainty > thresh).any(dim=-1)

    def calibrate(self, val_loader: Any, coverage: float = 0.95) -> None:
        self.eval()
        all_std: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                _, std = self(x)
                all_std.append(std)
        if not all_std:
            self._uncertainty_threshold = torch.tensor(0.0)
            return
        std_cat = torch.cat(all_std, dim=0)
        self._uncertainty_threshold = torch.quantile(std_cat.max(dim=-1).values, coverage)

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "n_models": self.n_models,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "output_bounds": self.output_bounds,
            "dropout_rate": self.dropout_rate,
            "threshold": self._uncertainty_threshold.item()
            if self._uncertainty_threshold is not None
            else None,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> EnsembleSurrogate:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        for key in [
            "n_models",
            "input_dim",
            "output_dim",
            "hidden_dims",
            "output_bounds",
            "dropout_rate",
        ]:
            if key in ckpt:
                kwargs.setdefault(key, ckpt[key])

        model = cls(**kwargs)
        model.load_state_dict(ckpt["state_dict"])
        if ckpt.get("threshold") is not None:
            model._uncertainty_threshold = torch.tensor(float(ckpt["threshold"]))
        return model
