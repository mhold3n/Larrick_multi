from __future__ import annotations

"""Thermo module — equation-first thermodynamic physics models."""

from .constants import ThermoConstants, load_anchor_manifest, load_thermo_constants
from .validation import ThermoValidationReport

__all__ = [
    "ThermoConstants",
    "ThermoResult",
    "ThermoValidationReport",
    "TwoZoneThermoResult",
    "eval_thermo",
    "evaluate_two_zone_thermo",
    "load_anchor_manifest",
    "load_thermo_constants",
]


def __getattr__(name: str):
    if name in {"ThermoResult", "eval_thermo"}:
        from .motionlaw import ThermoResult, eval_thermo

        exports = {
            "ThermoResult": ThermoResult,
            "eval_thermo": eval_thermo,
        }
        return exports[name]
    if name in {"TwoZoneThermoResult", "evaluate_two_zone_thermo"}:
        from .two_zone import TwoZoneThermoResult, evaluate_two_zone_thermo

        exports = {
            "TwoZoneThermoResult": TwoZoneThermoResult,
            "evaluate_two_zone_thermo": evaluate_two_zone_thermo,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
