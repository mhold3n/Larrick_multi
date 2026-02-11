"""Surrogate modeling components."""

from larrak2.surrogate.features import extract_scavenge_features_v1, get_scavenge_schema_v1
from larrak2.surrogate.gear_loss_net import GearLossNetwork
from larrak2.surrogate.models import EnsembleRegressor
from larrak2.surrogate.openfoam_nn import OpenFoamSurrogate

__all__ = [
    "extract_scavenge_features_v1",
    "get_scavenge_schema_v1",
    "GearLossNetwork",
    "EnsembleRegressor",
    "OpenFoamSurrogate",
]
