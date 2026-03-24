from __future__ import annotations

"""Computational Engineering Model (CEM) domain modules.

Authoritative physics for material selection, tribology, surface finish,
lubrication, and post-processing.  Used for post-optimization validation
(no surrogates) and as the ground-truth backing the optimization-loop
surrogate system.

Dataset ingestion architecture is set up with placeholder tables derived
from NASA, ISO 6336, and manufacturer datasheets.  Real experimental data
slots in via the registry without code changes.
"""

from larrak2.cem.lubrication import LubricationMode, LubricationParams  # noqa: E402
from larrak2.cem.material_db import MaterialClass, MaterialProperties, get_material  # noqa: E402
from larrak2.cem.post_processing import CoatingType, HeatTreatment  # noqa: E402
from larrak2.cem.registry import DatasetRegistry, get_registry  # noqa: E402
from larrak2.cem.surface_finish import (  # noqa: E402
    SurfaceFinishTier,
    effective_composite_roughness,
)
from larrak2.cem.tribology import (  # noqa: E402
    LubeRegime,
    TribologyParams,
    classify_regime,
    compute_lambda,
    compute_micropitting_safety,
    compute_scuff_margin,
    compute_scuff_margins,
    evaluate_tribology,
)

__all__ = [
    "MaterialClass",
    "MaterialProperties",
    "get_material",
    "LubeRegime",
    "TribologyParams",
    "classify_regime",
    "compute_lambda",
    "compute_micropitting_safety",
    "compute_scuff_margin",
    "compute_scuff_margins",
    "evaluate_tribology",
    "SurfaceFinishTier",
    "effective_composite_roughness",
    "LubricationMode",
    "LubricationParams",
    "CoatingType",
    "HeatTreatment",
    "DatasetRegistry",
    "get_registry",
]
