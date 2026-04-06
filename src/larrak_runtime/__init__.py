"""Shared runtime package for optimization-facing engine evaluation."""

from importlib import import_module
from typing import Any

__all__ = [
    "Candidate",
    "EvalContext",
    "EvalResult",
    "GearParams",
    "RealWorldParams",
    "ThermoParams",
    "VariableMetadata",
    "bounds",
    "decode_candidate",
    "encode_candidate",
    "evaluate_candidate",
    "group_indices",
    "variable_manifest",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        core = import_module("larrak_runtime.core")
        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
