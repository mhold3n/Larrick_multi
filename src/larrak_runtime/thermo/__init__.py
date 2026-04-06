"""Thermo module — equation-first thermodynamic physics models."""

from .constants import ThermoConstants, load_anchor_manifest, load_thermo_constants
from .validation import ThermoValidationReport

__all__ = [
    "ThermoConstants",
    "ThermoResult",
    "ThermoSymbolicArtifact",
    "ThermoValidationReport",
    "TwoZoneThermoResult",
    "apply_thermo_symbolic_overlay",
    "assemble_thermo_symbolic_feature_vector",
    "eval_thermo",
    "evaluate_two_zone_thermo",
    "load_anchor_manifest",
    "load_thermo_constants",
    "load_thermo_symbolic_artifact",
    "numeric_thermo_forward",
    "save_thermo_symbolic_artifact",
    "symbolic_thermo_forward",
    "train_thermo_symbolic_affine",
]


def __getattr__(name: str):
    if name in {
        "ThermoSymbolicArtifact",
        "load_thermo_symbolic_artifact",
        "save_thermo_symbolic_artifact",
        "train_thermo_symbolic_affine",
    }:
        from .symbolic_artifact import (
            ThermoSymbolicArtifact,
            load_thermo_symbolic_artifact,
            save_thermo_symbolic_artifact,
            train_thermo_symbolic_affine,
        )

        exports = {
            "ThermoSymbolicArtifact": ThermoSymbolicArtifact,
            "load_thermo_symbolic_artifact": load_thermo_symbolic_artifact,
            "save_thermo_symbolic_artifact": save_thermo_symbolic_artifact,
            "train_thermo_symbolic_affine": train_thermo_symbolic_affine,
        }
        return exports[name]
    if name in {
        "apply_thermo_symbolic_overlay",
        "assemble_thermo_symbolic_feature_vector",
        "numeric_thermo_forward",
        "symbolic_thermo_forward",
    }:
        from .symbolic_bridge import (
            apply_thermo_symbolic_overlay,
            assemble_thermo_symbolic_feature_vector,
            numeric_thermo_forward,
            symbolic_thermo_forward,
        )

        exports = {
            "apply_thermo_symbolic_overlay": apply_thermo_symbolic_overlay,
            "assemble_thermo_symbolic_feature_vector": assemble_thermo_symbolic_feature_vector,
            "numeric_thermo_forward": numeric_thermo_forward,
            "symbolic_thermo_forward": symbolic_thermo_forward,
        }
        return exports[name]
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
