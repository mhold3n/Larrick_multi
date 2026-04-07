"""Stack surrogate: submodule runtime re-exports plus integration training entrypoints."""

from __future__ import annotations

import importlib

_m = importlib.import_module("larrak_runtime.surrogate.stack")
for _k, _v in vars(_m).items():
    if _k.startswith("__"):
        continue
    globals()[_k] = _v

from .train import (  # noqa: E402
    Normalization,
    StackMLP,
    export_torch_mlp_artifact,
    train_stack_surrogate,
)

__all__ = list(getattr(_m, "__all__", [])) + [
    "Normalization",
    "StackMLP",
    "export_torch_mlp_artifact",
    "train_stack_surrogate",
]
del importlib, _m, _k, _v
