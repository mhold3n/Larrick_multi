"""Real-world constraint functions for the optimization loop.

Converts surrogate evaluation results into G-vector constraint values
following the project convention: G ≤ 0 → feasible.

Constraints:
    rw_lambda_min:       λ_min ≥ 1.0 (target full EHL)
    rw_scuff_margin:     Scuff temperature margin ≥ 0 °C
    rw_micropitting_sf:  Micropitting safety factor S_λ ≥ 1.0
    rw_material_temp:    Material service temp − operating temp ≥ 0 °C
    rw_cost_index:       Soft penalty if cost exceeds threshold
    rw_life_damage_10k:  log10(D_total) ≤ 0.0 (10,000 h service life)
"""

from __future__ import annotations

import numpy as np

from larrak2.realworld.surrogates import RealWorldSurrogateResult

# Constraint names (must match core/constraints.py registry)
REALWORLD_CONSTRAINT_NAMES: list[str] = [
    "rw_lambda_min",
    "rw_scuff_margin",
    "rw_micropitting_sf",
    "rw_material_temp",
    "rw_cost_index",
    "rw_life_damage_10k",
    "rw_material_snap_dist",
]

# Thresholds
_LAMBDA_TARGET = 1.0  # Full EHL target
_SCUFF_MARGIN_MIN = 0.0  # Minimum scuff margin (°C)
_MICROPITTING_SF_MIN = 1.0  # Minimum micropitting safety factor
_COST_THRESHOLD = 8.0  # Soft penalty above this cost index
_SNAP_DIST_MAX = 0.4  # Max allowed normalized material property distance


def compute_realworld_constraints(
    result: RealWorldSurrogateResult,
    operating_temp_C: float = 200.0,
    life_damage_total: float = 0.0,
    min_snap_distance: float | None = None,
) -> tuple[list[float], list[str]]:
    """Convert surrogate results into G-vector constraints.

    Convention: G ≤ 0 → feasible.

    Args:
        result: Output from evaluate_realworld_surrogates().
        operating_temp_C: Bulk gear temperature (for diagnostics).
        life_damage_total: Accumulated Miner damage D_total from life_damage module.
        min_snap_distance: Distance to nearest feasible material route (0.0 if exact match).

    Returns:
        (G_values, constraint_names): Lists of constraint values and names.
    """
    G: list[float] = []

    # 1. λ_min ≥ 1.0 → G = 1.0 − λ_min (negative when λ > 1)
    G.append(_LAMBDA_TARGET - result.lambda_min)

    # 2. Scuff margin ≥ 0 → G = −scuff_margin (negative when margin > 0)
    G.append(-result.scuff_margin_C)

    # 3. Micropitting S_λ ≥ 1.0 → G = 1.0 − S_λ
    G.append(_MICROPITTING_SF_MIN - result.micropitting_safety)

    # 4. Material temp margin ≥ 0 → G = −temp_margin
    G.append(-result.material_temp_margin_C)

    # 5. Cost index soft penalty → G = cost − threshold (0 if below threshold)
    cost_violation = max(0.0, result.total_cost_index - _COST_THRESHOLD)
    G.append(cost_violation)

    # 6. Life damage D_total ≤ 1.0.
    # Use log compression so large positive violations do not numerically
    # dominate the rest of the optimization while preserving feasibility:
    #   D_total <= 1  <=>  log10(D_total) <= 0
    D_total = max(float(life_damage_total), 1e-12)
    G.append(float(np.log10(D_total)))

    # 7. Material Snapping Distance penalty → G = snap_dist - d_max
    dist = (
        float(min_snap_distance)
        if min_snap_distance is not None
        else getattr(result, "min_snap_distance", 0.0)
    )
    G.append(dist - _SNAP_DIST_MAX)

    return G, list(REALWORLD_CONSTRAINT_NAMES)
