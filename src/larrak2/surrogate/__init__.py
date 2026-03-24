from __future__ import annotations

"""Surrogate modeling components."""

from larrak2.surrogate.calculix_nn import CalculixSurrogate  # noqa: E402
from larrak2.surrogate.features import extract_scavenge_features_v1, get_scavenge_schema_v1  # noqa: E402
from larrak2.surrogate.gear_loss_net import GearLossNetwork  # noqa: E402
from larrak2.surrogate.hifi import (  # noqa: E402
    FlowCoefficientSurrogate,
    StructuralSurrogate,
    ThermalSurrogate,
)
from larrak2.surrogate.models import EnsembleRegressor  # noqa: E402
from larrak2.surrogate.openfoam_nn import OpenFoamSurrogate  # noqa: E402

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
