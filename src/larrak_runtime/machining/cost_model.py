from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Standard End Mill Diameters (mm)
# Prefer common sizes: 2, 3, 4, 6, 8, 10, 12, 16, 20
STANDARD_FISHPOD_TOOLS = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0])

# Base Machining Cost (arbitrary units, maybe scaled to $)
BASE_SETUP_COST = 100.0
COST_PER_HOUR = 80.0


@dataclass
class MachiningMetrics:
    tooling_cost: float
    tolerance_penalty: float
    is_standard_tool: bool
    required_tolerance_mm: float


def calculate_tooling_cost(
    min_feature_size_outer_mm: float, min_feature_size_inner_mm: float, depth_mm: float = 14.0
) -> tuple[float, bool]:
    """Calculate relative tooling cost based on geometric constraints.

    Args:
        min_feature_size_outer_mm: 2 * BMaxSurvivable (Max tool diameter for outer profile)
        min_feature_size_inner_mm: MinHoleDiameter (Max tool/boring bar for holes)

    Returns:
        (cost_index, is_standard)
    """
    # 1. Outer Profile (End Mill)
    # Tool diameter must be <= min_feature_size_outer_mm
    # Ideally slightly smaller to allow movement?
    # PicoGK BMaxSurvivable is the radius of the largest circle.
    # So max tool diameter = 2 * BMaxSurvivable.
    # Actually, BMaxSurvivable includes kerf? No, it's the inset.
    # If BMaxSurvivable = 2mm, then a R=2mm (D=4mm) tool fits exactly.

    max_d_outer = min_feature_size_outer_mm

    # Filter standard tools that fit
    # Allow 5% margin for clearance? Or assume exact fit is non-optimal?
    # Standard practice: Tool D <= 0.9 * Feature Minimum? No, 1.0 is acceptable for contouring if open.
    # But for concave corners (fillets), Tool Radius <= Feature Radius.

    # BMaxSurvivable essentially measures the tightest concave radius (or gap check).

    valid_tools = STANDARD_FISHPOD_TOOLS[STANDARD_FISHPOD_TOOLS <= (max_d_outer + 1e-3)]

    if len(valid_tools) == 0:
        # Requires custom micro-tool
        # Cost scales inversely with diameter (very small tools break, slow feed)
        effective_d = max(0.1, max_d_outer)
        cost = 5.0 + (10.0 / effective_d)
        return cost, False

    best_tool_d = valid_tools[-1]  # Largest standard tool

    # 2. Inner Profile (Holes)
    # If there are holes, we need a tool that fits inside.
    # If min_feature_size_inner_mm > 0 (holes exist)
    if min_feature_size_inner_mm > 0.0:
        if min_feature_size_inner_mm < best_tool_d:
            # We need a smaller tool for the holes?
            # Or assume tool change?
            # Start simple: Bottleneck is the smallest feature anywhere.
            # Identify valid tools for inner too.
            valid_inner = STANDARD_FISHPOD_TOOLS[
                STANDARD_FISHPOD_TOOLS <= min_feature_size_inner_mm
            ]
            if len(valid_inner) == 0:
                effective_d = max(0.1, min_feature_size_inner_mm)
                return 5.0 + (10.0 / effective_d), False

            # Use smaller of the two best tools?
            # Actually cost is driven by the smallest tool required for the job.
            best_tool_d = min(best_tool_d, valid_inner[-1])

    # Cost model: Larger tools are faster (MRR ~ D^2)
    # Cost ~ 1 / D
    # Normalize: D=10mm -> Cost 1.0
    cost = 10.0 / best_tool_d
    return cost, True


def calculate_tolerance_budget(
    min_ligament_mm: float,
    min_curvature_mm: float,
    aspect_ratio: float = 1.0,
    torque_nm: float = 100.0,
    budget_mm: float = 0.5,
    mode: str = "capability_floor",
    penalty_weight: float = 10.0,
) -> tuple[float, float]:
    """Calculate required tolerance and penalty.

    Modes:
        - capability_floor: violate when total_req < budget_mm
        - stack_budget_max: violate when total_req > budget_mm

    Returns:
        (total_required_tolerance, penalty)
    """
    # 1. Profile Tolerance (Driven by ligament and curvature)
    # Thin ligaments need tight tolerance to exist.
    # Empirical model:
    # If t > 2.0, tol = 0.1
    # If t < 0.5, tol = 0.01

    # Sigmoid or interpolation?
    # Linear interp:
    # 0.2mm -> 0.005mm
    # 2.0mm -> 0.1mm

    if min_ligament_mm < 2.0:
        t_prof_lig = 0.005 + (0.1 - 0.005) * (max(0, min_ligament_mm - 0.2) / 1.8)
    else:
        t_prof_lig = 0.1

    # Curvature-driven profile tolerance term.
    # Tighter curvature (smaller radius) generally demands tighter tolerance.
    curv = float(max(min_curvature_mm, 0.1))
    curv_norm = min(max((curv - 0.1) / 1.9, 0.0), 1.0)  # 0.1 mm -> 0, 2.0 mm -> 1
    t_prof_curv = 0.005 + (0.1 - 0.005) * curv_norm

    # Required profile tolerance is controlled by the tightest feature.
    t_prof = min(t_prof_lig, t_prof_curv)

    # 2. Thickness Tolerance
    # Aspect ratio > 10 -> warp -> tight flatness.
    if aspect_ratio > 10.0:
        t_thick = 0.05
    else:
        t_thick = 0.2

    # 3. Bore Tolerance
    # High torque -> tight fit.
    # Linear scale with torque?
    # Simplified: 0.02 (H7) for precision, 0.05 for loose.
    # If torque > 500, needs 0.01.
    if torque_nm > 500:
        t_bore = 0.01
    elif torque_nm > 100:
        t_bore = 0.025
    else:
        t_bore = 0.05

    total_req = t_prof + t_thick + t_bore

    # Penalty
    if mode == "capability_floor":
        penalty = max(0.0, budget_mm - total_req) * float(penalty_weight)
    elif mode == "stack_budget_max":
        penalty = max(0.0, total_req - budget_mm) * float(penalty_weight)
    else:
        raise ValueError(
            f"Unsupported tolerance budget mode '{mode}'. "
            "Use 'capability_floor' or 'stack_budget_max'."
        )

    return total_req, penalty
