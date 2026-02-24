"""Identity-less material state screening and validation.

This module provides the soft selection logic for the optimizer's
4D material state continuous space without creating 'virtual alloys'.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from larrak2.cem.registry import get_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locked Normalization Bounds (Do not derive dynamically!)
# ---------------------------------------------------------------------------
# Axes must explicitly match route_metadata.csv and be static to ensure
# reproducible DOE distances.
NORM_BOUNDS = {
    "case_hardness_hrc_nom": (50.0, 70.0),
    "core_toughness_KIC_MPa_m05": (30.0, 160.0),
    "max_service_temp_C": (100.0, 500.0),
    "cleanliness_grade_proxy": (0.0, 1.0),
}
_AXES = list(NORM_BOUNDS.keys())

# Cache: list of dict representing the verified route_metadata table.
_ROUTE_CLOUD_CACHE: list[dict[str, Any]] | None = None


def _load_and_validate_route_cloud() -> list[dict[str, Any]]:
    """Load route_metadata.csv and validate against NORM_BOUNDS."""
    global _ROUTE_CLOUD_CACHE
    if _ROUTE_CLOUD_CACHE is not None:
        return _ROUTE_CLOUD_CACHE

    reg = get_registry()
    table = reg.load_table("route_metadata")

    n_rows = len(table["route_id"])
    routes = []
    for i in range(n_rows):
        route = {
            "route_id": table["route_id"][i],
            "process_family": table["process_family"][i],
            "quality_grade": table.get("quality_grade", [""] * n_rows)[i],
            "provenance": table.get("provenance", [""] * n_rows)[i],
        }

        # Parse and validate the 4 axes
        for axis in _AXES:
            val = float(table[axis][i])
            vmin, vmax = NORM_BOUNDS[axis]
            if not (vmin <= val <= vmax):
                raise ValueError(
                    f"Route {route['route_id']} axis {axis}={val} is outside "
                    f"locked normalization bounds [{vmin}, {vmax}]."
                )
            route[axis] = val

            # Store pre-normalized active coordinate for fast distance calc
            route[f"{axis}_norm"] = (val - vmin) / (vmax - vmin)

        routes.append(route)

    _ROUTE_CLOUD_CACHE = routes
    return _ROUTE_CLOUD_CACHE


def invalidate_snapping_cache() -> None:
    """Clear module-level metadata cache."""
    global _ROUTE_CLOUD_CACHE
    _ROUTE_CLOUD_CACHE = None


def get_soft_selected_routes(
    state_norm: np.ndarray,
    gear_bulk_temp_C: float,
    k: int = 3,
    tau: float = 0.1,
) -> tuple[float, list[tuple[str, float]]]:
    """Retrieve top-k valid discrete routes and their selection probabilities.

    Filters routes strictly by their max_service_temp_C limit.
    Computes Euclidean distance across the 4 normalized axes.
    Probability mixing weights alpha_i proportional to exp(-d_i / tau).

    Args:
        state_norm: Length-4 normalized state array [0-1].
        gear_bulk_temp_C: Environmental condition filter.
        k: Top nearest neighbors to return.
        tau: Temperature scaling parameter for softmax blending.

    Returns:
        (min_snap_distance, [(route_id_1, alpha_1), ...])

    Raises:
        ValueError: If no valid routes survive the temperature filter.
    """
    if len(state_norm) != len(_AXES):
        raise ValueError(f"State vector must have length {len(_AXES)}.")

    routes = _load_and_validate_route_cloud()

    valid_routes = []
    for r in routes:
        if gear_bulk_temp_C > r["max_service_temp_C"]:
            continue
        valid_routes.append(r)

    if not valid_routes:
        raise ValueError(
            f"No routes available with max_service_temp_C >= {gear_bulk_temp_C} °C"
        )

    # Compute distances
    scored = []
    for r in valid_routes:
        r_coords = np.array([r[f"{a}_norm"] for a in _AXES])
        # Unweighted Euclidean
        d = float(np.linalg.norm(state_norm - r_coords))
        scored.append((d, r))

    # Sort and take top k
    scored.sort(key=lambda x: x[0])
    top_k = scored[:k]

    min_dist = top_k[0][0]

    # Compute soft attention probabilities
    weights = []
    for d, _ in top_k:
        weights.append(np.exp(-d / tau))

    sum_w = sum(weights)
    alphas = [w / sum_w for w in weights]

    result_list = [(r["route_id"], alpha) for (_, r), alpha in zip(top_k, alphas)]

    return min_dist, result_list
