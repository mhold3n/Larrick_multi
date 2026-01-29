"""Gear module — gear synthesis and losses."""

from .litvin_core import GearResult, eval_gear
from .pitchcurve import PitchCurve, fourier_pitch_curve

__all__ = ["eval_gear", "GearResult", "PitchCurve", "fourier_pitch_curve"]
