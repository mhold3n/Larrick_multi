import logging

import numpy as np
from larrak_engines.gear import picogk_adapter
from larrak_engines.gear.picogk_adapter import evaluate_manufacturability

# Enable debug logging
logging.basicConfig(level=logging.INFO)
picogk_adapter.logger.setLevel(logging.DEBUG)


def test_holes():
    print("Testing Hole Metrics and Subtraction...")

    # 1. Create a simple circular gear (Ring)
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    r_planet = np.ones_like(theta) * 50.0  # 50mm radius

    # 2. Create a hole (Square 10x10 at center? No, center is void)
    # The gear is a ring.
    # Actually, PicoGK logic:
    # Outer is the boundary.
    # Extrusion fills the inside.
    # If I want a hole, I subtract from the filled shape.

    # Let's define a hole: A small square at (10, 0)
    # Square side 10mm.
    # Points: (5, -5), (15, -5), (15, 5), (5, 5)
    # Area = 100.
    # Equivalent Diameter = 2 * sqrt(100/pi) = 2 * 5.64 = 11.28mm.
    # Min Curvature: 0 (Polygon corners). But with osuculating fit?
    # 3-point fit on square corners -> 90 deg -> Radius?
    # If I define it as a circle approximation

    # Let's use a circle hole for clean metrics.
    # Circle radius 5mm at (20,0).
    h_theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    h_r = 5.0
    h_x = 20.0 + h_r * np.cos(h_theta)
    h_y = 0.0 + h_r * np.sin(h_theta)
    hole_poly = np.column_stack((h_x, h_y)).tolist()

    # Pass as list of holes
    holes = [hole_poly]

    result = evaluate_manufacturability(
        theta,
        r_planet,
        wire_d_mm=0.2,
        min_ligament_mm=0.25,
        holes=holes,
        voxel_size_mm=0.1,  # Coarser for speed
        timeout_s=60,
    )

    print("\nResult:")
    print(f"Passed: {result['passed']}")
    print(f"Min Hole Diameter: {result.get('min_hole_diameter_mm')} mm")
    print(f"Min Hole Curvature: {result.get('min_hole_curvature_radius_mm')} mm")

    # Expected:
    # Area ~ pi*5^2 = 78.5
    # EqDiam ~ 10.0
    # Curvature ~ 5.0

    if result.get("min_hole_diameter_mm", 0) > 9.0:
        print("PASS: Hole diameter metric detected.")
    else:
        print("FAIL: Hole diameter incorrect.")


if __name__ == "__main__":
    test_holes()
