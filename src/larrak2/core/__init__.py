from __future__ import annotations

"""Core module — types, encoding, evaluator, utilities."""

from .encoding import (
    Candidate,
    GearParams,
    RealWorldParams,
    ThermoParams,
    VariableMetadata,
    bounds,
    decode_candidate,
    encode_candidate,
    group_indices,
    variable_manifest,
)
from .evaluator import evaluate_candidate
from .types import EvalContext, EvalResult

__all__ = [
    "EvalContext",
    "EvalResult",
    "Candidate",
    "ThermoParams",
    "GearParams",
    "RealWorldParams",
    "VariableMetadata",
    "decode_candidate",
    "encode_candidate",
    "bounds",
    "variable_manifest",
    "group_indices",
    "evaluate_candidate",
]
