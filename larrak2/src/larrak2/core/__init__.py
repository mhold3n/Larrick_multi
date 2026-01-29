"""Core module — types, encoding, evaluator, utilities."""

from .encoding import (
    Candidate,
    GearParams,
    ThermoParams,
    bounds,
    decode_candidate,
    encode_candidate,
)
from .evaluator import evaluate_candidate
from .types import EvalContext, EvalResult

__all__ = [
    "EvalContext",
    "EvalResult",
    "Candidate",
    "ThermoParams",
    "GearParams",
    "decode_candidate",
    "encode_candidate",
    "bounds",
    "evaluate_candidate",
]
