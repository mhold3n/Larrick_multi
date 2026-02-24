"""Strict temperature-dependent material property accessor.

Provides ``get_property_at_temp()`` which interpolates temperature-dependent
properties from the ``temperature_curves`` dataset.

Policies:
    - **No extrapolation**: raises ``ValueError`` if T is outside data range.
    - **No duplicate temps**: raises ``ValueError`` on repeated T for same
      (route_id, property) pair.
    - **Loud failure**: raises ``ValueError`` for unknown (route_id, property).
    - **Derived properties**: ``diffusivity_m2_s`` is computed from
      ``k_W_mK``, ``rho_kg_m3``, ``cp_J_kgK`` (never stored directly).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Module-scope cache: {(route_id, property): [(T, value), ...]}
_CURVE_CACHE: dict[tuple[str, str], list[tuple[float, float]]] | None = None


def _load_curves() -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Load and cache temperature curves from the registry."""
    global _CURVE_CACHE
    if _CURVE_CACHE is not None:
        return _CURVE_CACHE

    from larrak2.cem.registry import get_registry

    reg = get_registry()
    table = reg.load_table("temperature_curves")

    curves: dict[tuple[str, str], list[tuple[float, float]]] = {}

    if table.get("route_id") and len(table["route_id"]) > 0:
        for i in range(len(table["route_id"])):
            rid = str(table["route_id"][i]).strip()
            prop = str(table["property"][i]).strip()
            temp = float(table["temp_c"][i])
            val = float(table["value"][i])
            key = (rid, prop)
            if key not in curves:
                curves[key] = []
            curves[key].append((temp, val))

    # Validate: sorted temps, no duplicates
    for key, points in curves.items():
        temps = [t for t, _ in points]
        if len(temps) != len(set(temps)):
            raise ValueError(
                f"Duplicate temperatures for {key}: {temps}. "
                f"Each (route_id, property) must have unique temperature points."
            )
        # Sort by temperature
        curves[key] = sorted(points, key=lambda x: x[0])

    _CURVE_CACHE = curves
    return _CURVE_CACHE


def get_property_at_temp(
    route_id: str,
    property_name: str,
    T_C: float,
) -> float:
    """Retrieve a material property at a given temperature.

    For ``diffusivity_m2_s``, computes α = k / (ρ · cp) from constituent
    properties at the same temperature.

    Args:
        route_id: Material route identifier (e.g. ``"AISI_9310"``).
        property_name: Property key (e.g. ``"youngs_modulus_GPa"``).
        T_C: Temperature in °C.

    Returns:
        Interpolated property value.

    Raises:
        ValueError: If extrapolation would be required, or data is missing.
    """
    # Derived property: diffusivity
    if property_name == "diffusivity_m2_s":
        k = get_property_at_temp(route_id, "k_W_mK", T_C)
        rho = get_property_at_temp(route_id, "rho_kg_m3", T_C)
        cp = get_property_at_temp(route_id, "cp_J_kgK", T_C)
        return k / (rho * cp)

    curves = _load_curves()
    key = (route_id, property_name)

    if key not in curves:
        raise ValueError(
            f"No temperature curve found for route_id='{route_id}', "
            f"property='{property_name}'. "
            f"Available keys: {sorted(curves.keys())}"
        )

    points = curves[key]
    temps = np.array([t for t, _ in points])
    vals = np.array([v for _, v in points])

    # Strict: no extrapolation
    if T_C < temps[0] or T_C > temps[-1]:
        raise ValueError(
            f"Temperature {T_C}°C is outside the data range "
            f"[{temps[0]}, {temps[-1]}]°C for route_id='{route_id}', "
            f"property='{property_name}'. No extrapolation allowed."
        )

    # Exact match
    if len(temps) == 1:
        if T_C == temps[0]:
            return float(vals[0])
        raise ValueError(
            f"Only one data point at {temps[0]}°C for {key}, "
            f"cannot interpolate to {T_C}°C."
        )

    # Linear interpolation
    return float(np.interp(T_C, temps, vals))


def invalidate_cache() -> None:
    """Clear the module-scope curve cache (useful for testing)."""
    global _CURVE_CACHE
    _CURVE_CACHE = None
