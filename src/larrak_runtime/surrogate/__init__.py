"""Surrogate modeling components."""

from .calculix_nn import CalculixSurrogate
from .features import extract_scavenge_features_v1, get_scavenge_schema_v1
from .gear_loss_net import GearLossNetwork
from .hifi import (
    FlowCoefficientSurrogate,
    StructuralSurrogate,
    ThermalSurrogate,
)
from .models import EnsembleRegressor
from .openfoam_nn import OpenFoamSurrogate

__all__ = [
    "CalculixSurrogate",
    "extract_scavenge_features_v1",
    "get_scavenge_schema_v1",
    "GearLossNetwork",
    "EnsembleRegressor",
    "OpenFoamSurrogate",
    "ThermalSurrogate",
    "StructuralSurrogate",
    "FlowCoefficientSurrogate",
]
