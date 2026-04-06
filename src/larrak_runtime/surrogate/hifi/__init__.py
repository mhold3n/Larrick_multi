"""HiFi surrogate models and training primitives."""

from .ensemble import BoundedMLP, EnsembleSurrogate
from .models import FlowCoefficientSurrogate, StructuralSurrogate, ThermalSurrogate

__all__ = [
    "BoundedMLP",
    "EnsembleSurrogate",
    "FlowCoefficientSurrogate",
    "StructuralSurrogate",
    "ThermalSurrogate",
]
