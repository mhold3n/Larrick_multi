"""Export Litvin-derived gear profiles to JSON for PicoGK manufacturability oracle.

Converts polar radius arrays from litvin_synthesize into closed 2D polylines
(Cartesian coordinates) and writes them as JSON for the C# CLI oracle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def polar_to_cartesian(
    theta: np.ndarray,
    r: np.ndarray,
) -> list[list[float]]:
    """Convert polar (θ, r) arrays to a closed Cartesian polyline [[x, y], ...]."""
    theta = np.asarray(theta, dtype=float)
    r = np.asarray(r, dtype=float)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = [[float(xi), float(yi)] for xi, yi in zip(x, y)]
    # Close the polyline by appending the first point
    if len(points) > 1 and points[0] != points[-1]:
        points.append(points[0])
    return points


def export_profile_json(
    theta: np.ndarray,
    r_planet: np.ndarray,
    process_params: dict[str, float],
    output_path: Path | str,
    *,
    gear_id: str = "",
    sample_id: str = "",
    R_psi: np.ndarray | None = None,
    psi: np.ndarray | None = None,
) -> Path:
    """Write a gear profile to JSON for the PicoGK oracle.

    Args:
        theta: Cam angle grid (radians), length N, [0, 2π).
        r_planet: Planet polar radius r(θ), length N.
        process_params: WEDM/laser process parameters dict with keys:
            wire_d_mm, overcut_mm, corner_margin_mm, min_ligament_mm.
        output_path: Destination JSON file path.
        gear_id: Optional identifier for the gear design.
        sample_id: Optional identifier for this sample/candidate.
        R_psi: Optional ring profile R(ψ) for dual-profile export.
        psi: Optional ring angle grid ψ (required if R_psi given).

    Returns:
        Path to the written JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    outer = polar_to_cartesian(theta, r_planet)

    payload: dict[str, Any] = {
        "units": "mm",
        "outer": outer,
        "holes": [],
        "metadata": {
            "gear_id": gear_id,
            "sample_id": sample_id,
            "n_points": len(theta),
        },
        "process": process_params,
    }

    # Optionally include ring profile
    if R_psi is not None and psi is not None:
        payload["ring_profile"] = polar_to_cartesian(psi, R_psi)

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.debug("Exported gear profile to %s (%d points)", output_path, len(theta))
    return output_path


def process_params_to_dict(
    wire_d_mm: float = 0.2,
    overcut_mm: float = 0.05,
    corner_margin_mm: float = 0.0,
    min_ligament_mm: float = 0.35,
) -> dict[str, float]:
    """Build a process parameters dict for the oracle JSON contract."""
    return {
        "wire_d_mm": wire_d_mm,
        "overcut_mm": overcut_mm,
        "corner_margin_mm": corner_margin_mm,
        "min_ligament_mm": min_ligament_mm,
    }
