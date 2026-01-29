"""Thermo module — thermodynamic physics models."""

from .motionlaw import ThermoResult, eval_thermo

__all__ = ["eval_thermo", "ThermoResult"]
