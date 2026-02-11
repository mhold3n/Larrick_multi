"""Generate opposed-piston rotary valve geometry STLs for OpenFOAM overset mesh.

This script generates the 3D geometry for:
- Cylinder body
- Rotating intake valve (drum with window cutout)
- Rotating exhaust valves (left and right)
- Intake and exhaust manifold volumes

The geometry is parameterized by bore, stroke, and port areas.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def create_cylinder_stl(
    bore_m: float,
    length_m: float,
    n_segments: int = 64,
) -> np.ndarray:
    """Create a hollow cylinder (tube) as triangles.

    Returns array of shape (n_triangles, 3, 3) for vertices.
    """
    radius = bore_m / 2
    angles = np.linspace(0, 2 * np.pi, n_segments + 1)[:-1]

    triangles = []

    # Outer surface
    for i in range(n_segments):
        a1, a2 = angles[i], angles[(i + 1) % n_segments]

        # Bottom triangle
        v0 = [radius * np.cos(a1), radius * np.sin(a1), 0]
        v1 = [radius * np.cos(a2), radius * np.sin(a2), 0]
        v2 = [radius * np.cos(a1), radius * np.sin(a1), length_m]
        triangles.append([v0, v1, v2])

        # Top triangle
        v3 = [radius * np.cos(a2), radius * np.sin(a2), length_m]
        triangles.append([v1, v3, v2])

    return np.array(triangles, dtype=np.float32)


def create_rotary_valve_stl(
    outer_radius_m: float,
    inner_radius_m: float,
    length_m: float,
    window_arc_deg: float,
    window_length_frac: float = 0.8,
    n_segments: int = 64,
) -> np.ndarray:
    """Create a water-tight rotary valve drum with a window cutout.

    Window is centered at PI (180 deg) to avoid wrapping issues.
    Generates:
    - Outer Surface (with cutout)
    - Inner Surface (full cylinder)
    - End Caps (annular rings)
    - Window Frame (Top, Bottom, Left, Right walls connecting outer/inner)
    """
    triangles = []

    angles = np.linspace(0, 2 * np.pi, n_segments + 1)[:-1]

    # Center window at PI
    window_center = np.pi
    half_arc = math.radians(window_arc_deg / 2)
    window_start = window_center - half_arc
    window_end = window_center + half_arc

    window_z_start = length_m * (1 - window_length_frac) / 2
    window_z_end = length_m - window_z_start

    # Helper to check if an angle is inside the window arc
    def is_in_window(a):
        return a >= window_start and a <= window_end

    # Pre-calculate segment status (Handle wrapping for safety, though center is PI)
    seg_is_window = []
    for i in range(n_segments):
        a1 = angles[i]
        # Segment is window if both vertices are in window
        seg_is_window.append(is_in_window(a1) and is_in_window(angles[(i + 1) % n_segments]))

    for i in range(n_segments):
        a1 = angles[i]
        a2 = angles[(i + 1) % n_segments]

        is_win = seg_is_window[i]

        # Vertices at this segment
        # 0,1: Inner Bot/Top
        # 2,3: Outer Bot/Top
        # 4,5: Outer Win Bot/Top
        # 6,7: Inner Win Bot/Top (Projected)

        def p(r, z, a):
            return [r * np.cos(a), r * np.sin(a), z]

        p_in_bot_1 = p(inner_radius_m, 0, a1)
        p_in_top_1 = p(inner_radius_m, length_m, a1)
        p_in_bot_2 = p(inner_radius_m, 0, a2)
        p_in_top_2 = p(inner_radius_m, length_m, a2)

        p_out_bot_1 = p(outer_radius_m, 0, a1)
        p_out_top_1 = p(outer_radius_m, length_m, a1)
        p_out_bot_2 = p(outer_radius_m, 0, a2)
        p_out_top_2 = p(outer_radius_m, length_m, a2)

        # 1. Inner Cylinder (Always full). Normal Inward.
        triangles.append([p_in_bot_1, p_in_top_1, p_in_bot_2])
        triangles.append([p_in_bot_2, p_in_top_1, p_in_top_2])

        # 3. End Caps (Always exist).
        # Bottom (z=0). Normal Down. Outer->Inner.
        triangles.append([p_out_bot_1, p_in_bot_1, p_out_bot_2])
        triangles.append([p_out_bot_2, p_in_bot_1, p_in_bot_2])
        # Top (z=L). Normal Up. Outer->Inner.
        triangles.append([p_out_top_1, p_out_top_2, p_in_top_1])
        triangles.append([p_out_top_2, p_in_top_2, p_in_top_1])

        if is_win:
            # WINDOW SEGMENT
            # 2. Outer Surface (Split)
            p_out_wbot_1 = p(outer_radius_m, window_z_start, a1)
            p_out_wbot_2 = p(outer_radius_m, window_z_start, a2)
            p_out_wtop_1 = p(outer_radius_m, window_z_end, a1)
            p_out_wtop_2 = p(outer_radius_m, window_z_end, a2)

            p_in_wbot_1 = p(inner_radius_m, window_z_start, a1)
            p_in_wbot_2 = p(inner_radius_m, window_z_start, a2)
            p_in_wtop_1 = p(inner_radius_m, window_z_end, a1)
            p_in_wtop_2 = p(inner_radius_m, window_z_end, a2)

            # Bottom Strip (0 to z_ws) - Normal Out
            triangles.append([p_out_bot_1, p_out_bot_2, p_out_wbot_1])
            triangles.append([p_out_bot_2, p_out_wbot_2, p_out_wbot_1])

            # Top Strip (z_we to L) - Normal Out
            triangles.append([p_out_wtop_1, p_out_wtop_2, p_out_top_1])
            triangles.append([p_out_wtop_2, p_out_top_2, p_out_top_1])

            # Window Frame Ledges
            # Bottom Ledge (z_ws). Normal Up. Outer->Inner.
            triangles.append([p_out_wbot_1, p_out_wbot_2, p_in_wbot_1])
            triangles.append([p_out_wbot_2, p_in_wbot_2, p_in_wbot_1])

            # Top Ledge (z_we). Normal Down. Outer->Inner.
            triangles.append([p_out_wtop_1, p_in_wtop_1, p_out_wtop_2])
            triangles.append([p_out_wtop_2, p_in_wtop_1, p_in_wtop_2])

            # Side Walls (Check Neighbors)
            prev_win = seg_is_window[(i - 1) % n_segments]
            next_win = seg_is_window[(i + 1) % n_segments]

            if not prev_win:
                # Left Wall at `a1`. Normal points right (into window).
                # Face is in plane `a1`.
                # Vertices: OutBot(a1, z_ws), OutTop(a1, z_we), InBot(a1, z_ws), InTop(a1, z_we)
                # CCW from inside?
                # Normal = Tangent(a1 direction) -> Right.
                # p_out_wbot_1 -> p_in_wbot_1 -> p_out_wtop_1
                triangles.append([p_out_wbot_1, p_in_wbot_1, p_out_wtop_1])
                triangles.append([p_in_wbot_1, p_in_wtop_1, p_out_wtop_1])

            if not next_win:
                # Right Wall at `a2`. Normal points left (into window).
                # Face is in plane `a2`.
                # Normal = -Tangent.
                # p_out_wbot_2 -> p_out_wtop_2 -> p_in_wbot_2
                triangles.append([p_out_wbot_2, p_out_wtop_2, p_in_wbot_2])
                triangles.append([p_in_wbot_2, p_out_wtop_2, p_in_wtop_2])

        else:
            # FULL SEGMENT
            # 2. Outer Surface Full. Normal Out.
            triangles.append([p_out_bot_1, p_out_bot_2, p_out_top_1])
            triangles.append([p_out_bot_2, p_out_top_2, p_out_top_1])

    return np.array(triangles, dtype=np.float32)


def create_manifold_box_stl(
    width_m: float,
    height_m: float,
    depth_m: float,
    center: tuple[float, float, float],
) -> np.ndarray:
    """Create a simple box manifold volume."""
    cx, cy, cz = center
    hw, hh, hd = width_m / 2, height_m / 2, depth_m / 2

    # 8 vertices of box
    vertices = np.array(
        [
            [cx - hw, cy - hh, cz - hd],
            [cx + hw, cy - hh, cz - hd],
            [cx + hw, cy + hh, cz - hd],
            [cx - hw, cy + hh, cz - hd],
            [cx - hw, cy - hh, cz + hd],
            [cx + hw, cy - hh, cz + hd],
            [cx + hw, cy + hh, cz + hd],
            [cx - hw, cy + hh, cz + hd],
        ],
        dtype=np.float32,
    )

    # 12 triangles (2 per face)
    faces = [
        # Bottom
        [0, 1, 2],
        [0, 2, 3],
        # Top
        [4, 6, 5],
        [4, 7, 6],
        # Front
        [0, 4, 5],
        [0, 5, 1],
        # Back
        [2, 6, 7],
        [2, 7, 3],
        # Left
        [0, 3, 7],
        [0, 7, 4],
        # Right
        [1, 5, 6],
        [1, 6, 2],
    ]

    triangles = []
    for f in faces:
        triangles.append([vertices[f[0]], vertices[f[1]], vertices[f[2]]])

    return np.array(triangles, dtype=np.float32)


def save_stl_ascii(filepath: Path, triangles: np.ndarray, name: str = "surface") -> None:
    """Save triangles to ASCII STL format."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"solid {name}\n")
        for tri in triangles:
            # Compute normal
            v0, v1, v2 = tri
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
            else:
                n = np.array([0, 0, 1])

            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


def port_area_to_window_arc(
    port_area_m2: float, valve_radius_m: float, valve_length_m: float
) -> float:
    """Convert port area to window arc angle."""
    arc_rad = port_area_m2 / (valve_radius_m * valve_length_m * 0.8)  # 0.8 for window_length_frac
    arc_deg = math.degrees(arc_rad)
    return min(max(arc_deg, 10.0), 120.0)  # Clamp to reasonable range


def generate_stl_workflow(args: Any) -> int:
    """Generate STL workflow entry point."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bore_m = args.bore_mm / 1000.0
    stroke_m = args.stroke_mm / 1000.0

    # Geometry parameters
    cylinder_length = stroke_m * 1.5  # Total cylinder length
    valve_radius = bore_m * 0.45  # Valve drum radius (fits inside bore 0.5)
    valve_inner_radius = bore_m * 0.3  # Inner cavity
    valve_length = bore_m * 0.5  # Axial length of valve

    # Compute window arcs from port areas
    intake_window_arc = port_area_to_window_arc(
        args.intake_port_area_m2, valve_radius, valve_length
    )
    exhaust_window_arc = port_area_to_window_arc(
        args.exhaust_port_area_m2, valve_radius, valve_length
    )

    print(f"Generating geometry: bore={bore_m * 1000:.1f}mm, stroke={stroke_m * 1000:.1f}mm")
    print(f"Intake window arc: {intake_window_arc:.1f}°, Exhaust: {exhaust_window_arc:.1f}°")

    # Generate cylinder
    cylinder_tris = create_cylinder_stl(bore_m, cylinder_length)
    save_stl_ascii(outdir / "cylinder.stl", cylinder_tris, "cylinder")

    # Generate intake valve (at center)
    intake_valve_tris = create_rotary_valve_stl(
        valve_radius, valve_inner_radius, valve_length, intake_window_arc
    )
    # Translate to center of cylinder
    intake_z_offset = cylinder_length / 2 - valve_length / 2
    intake_valve_tris[:, :, 2] += intake_z_offset
    save_stl_ascii(outdir / "intakeValve.stl", intake_valve_tris, "intakeValve")

    # Generate exhaust valves (at ends)
    exhaust_left_tris = create_rotary_valve_stl(
        valve_radius, valve_inner_radius, valve_length, exhaust_window_arc
    )
    # Left exhaust at z=0 end
    save_stl_ascii(outdir / "exhaustValveLeft.stl", exhaust_left_tris, "exhaustValveLeft")

    exhaust_right_tris = create_rotary_valve_stl(
        valve_radius, valve_inner_radius, valve_length, exhaust_window_arc
    )
    # Right exhaust at z=cylinder_length end
    exhaust_right_tris[:, :, 2] += cylinder_length - valve_length
    save_stl_ascii(outdir / "exhaustValveRight.stl", exhaust_right_tris, "exhaustValveRight")

    # Generate manifold boxes
    manifold_width = bore_m * 1.5
    manifold_height = bore_m
    manifold_depth = bore_m * 0.5

    # Intake manifold (above cylinder at center)
    intake_manifold = create_manifold_box_stl(
        manifold_width, manifold_height, manifold_depth, (0, bore_m, cylinder_length / 2)
    )
    save_stl_ascii(outdir / "intakeManifold.stl", intake_manifold, "intakeManifold")

    # Exhaust manifolds (below cylinder at ends)
    exhaust_left_manifold = create_manifold_box_stl(
        manifold_width, manifold_height, manifold_depth, (0, -bore_m, valve_length / 2)
    )
    save_stl_ascii(outdir / "exhaustManifoldLeft.stl", exhaust_left_manifold, "exhaustManifoldLeft")

    exhaust_right_manifold = create_manifold_box_stl(
        manifold_width,
        manifold_height,
        manifold_depth,
        (0, -bore_m, cylinder_length - valve_length / 2),
    )
    save_stl_ascii(
        outdir / "exhaustManifoldRight.stl", exhaust_right_manifold, "exhaustManifoldRight"
    )

    # Generate valve interface cylinders (for Sliding Mesh / AMI)
    gap_mm = 1.0
    gap_m = gap_mm / 1000.0
    interface_radius = valve_radius + (gap_m / 2.0)

    # Helper to create localized cylinder interface
    def create_interface(z_length):
        tris = create_cylinder_stl(interface_radius * 2.0, z_length)
        # Add end caps to the cylinder mesh
        n_seg = 64
        angles = np.linspace(0, 2 * np.pi, n_seg + 1)[:-1]
        caps = []
        for z_cap, flip_normal in [(0, True), (z_length, False)]:
            c = [0, 0, z_cap]
            for i in range(n_seg):
                a1, a2 = angles[i], angles[(i + 1) % n_seg]
                p1 = [interface_radius * np.cos(a1), interface_radius * np.sin(a1), z_cap]
                p2 = [interface_radius * np.cos(a2), interface_radius * np.sin(a2), z_cap]
                if flip_normal:
                    caps.append([c, p2, p1])  # Normal down
                else:
                    caps.append([c, p1, p2])  # Normal up
        caps = np.array(caps, dtype=np.float32)
        return np.concatenate([tris, caps])

    intake_interface = create_interface(valve_length)
    intake_interface[:, :, 2] += intake_z_offset
    save_stl_ascii(outdir / "intakeValveInterface.stl", intake_interface, "intakeValveInterface")

    exhaust_left_interface = create_interface(valve_length)
    save_stl_ascii(
        outdir / "exhaustValveLeftInterface.stl",
        exhaust_left_interface,
        "exhaustValveLeftInterface",
    )

    exhaust_right_interface = create_interface(valve_length)
    exhaust_right_interface[:, :, 2] += cylinder_length - valve_length
    save_stl_ascii(
        outdir / "exhaustValveRightInterface.stl",
        exhaust_right_interface,
        "exhaustValveRightInterface",
    )

    print(f"Generated STLs in {outdir}")
    return 0
