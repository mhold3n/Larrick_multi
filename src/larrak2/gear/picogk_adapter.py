"""Python adapter for the PicoGK manufacturability oracle CLI.

Wraps the C# CLI tool (tools/picogk_manufact) to evaluate WEDM/laser
manufacturability of gear profiles using voxel SDF offset operations.

Usage:
    from larrak2.gear.picogk_adapter import evaluate_manufacturability
    result = evaluate_manufacturability(theta, r_planet, process_params)
    if result["passed"]:
        ...
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .profile_export import export_profile_json, process_params_to_dict

logger = logging.getLogger(__name__)

# Cache: hash(profile + params) -> result dict
_ORACLE_CACHE: dict[str, dict[str, Any]] = {}

# Project root (derived from this file's location)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent.parent  # src/larrak2/gear -> repo root
_ORACLE_PROJECT = _REPO_ROOT / "tools" / "picogk_manufact"


def _cache_key(
    theta: np.ndarray,
    r_planet: np.ndarray,
    wire_d_mm: float,
    overcut_mm: float,
    corner_margin_mm: float,
    min_ligament_mm: float,
    voxel_size_mm: float,
) -> str:
    """Hash profile + params for memoization."""
    h = hashlib.sha256()
    h.update(np.asarray(theta, dtype=np.float64).tobytes())
    h.update(np.asarray(r_planet, dtype=np.float64).tobytes())
    h.update(
        f"{wire_d_mm},{overcut_mm},{corner_margin_mm},{min_ligament_mm},{voxel_size_mm}".encode()
    )
    return h.hexdigest()


def evaluate_manufacturability(
    theta: np.ndarray,
    r_planet: np.ndarray,
    wire_d_mm: float = 0.2,
    overcut_mm: float = 0.05,
    corner_margin_mm: float = 0.0,
    min_ligament_mm: float = 0.35,
    voxel_size_mm: float = 0.001,
    slab_thickness_mm: float = 14.0,
    timeout_s: float = 120.0,
    *,
    R_psi: np.ndarray | None = None,
    psi: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate manufacturability of a gear profile via PicoGK oracle.

    Args:
        theta: Cam angle grid (radians), length N, [0, 2π).
        r_planet: Planet polar radius r(θ), length N.
        wire_d_mm: Wire diameter (WEDM) in mm.
        overcut_mm: Spark gap / overcut in mm.
        corner_margin_mm: Corner radius margin in mm.
        min_ligament_mm: Minimum ligament thickness in mm.
        voxel_size_mm: Voxel resolution in mm (0.001 = 1 µm).
        slab_thickness_mm: Extrusion height in mm.
        timeout_s: CLI timeout in seconds.
        R_psi: Optional ring profile R(ψ).
        psi: Optional ring angle grid.

    Returns:
        Dict with keys: passed, kerf_buffer_mm, t_min_proxy_mm,
        b_max_survivable_mm, area_original_mm2, area_after_inset_mm2,
        component_count_after_inset, voxel_resolution_mm, notes.
    """
    key = _cache_key(
        theta, r_planet, wire_d_mm, overcut_mm, corner_margin_mm, min_ligament_mm, voxel_size_mm
    )
    if key in _ORACLE_CACHE:
        logger.debug("PicoGK oracle cache hit")
        return _ORACLE_CACHE[key]

    process_params = process_params_to_dict(
        wire_d_mm, overcut_mm, corner_margin_mm, min_ligament_mm
    )

    with tempfile.TemporaryDirectory(prefix="picogk_") as tmpdir:
        input_path = Path(tmpdir) / "profile.json"
        export_profile_json(
            theta,
            r_planet,
            process_params,
            input_path,
            R_psi=R_psi,
            psi=psi,
        )

        cmd = [
            "dotnet",
            "run",
            "--project",
            str(_ORACLE_PROJECT),
            "--",
            "--input",
            str(input_path),
            "--voxel-size",
            str(voxel_size_mm),
            "--slab-thickness",
            str(slab_thickness_mm),
        ]

        logger.debug("Running PicoGK oracle: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(_ORACLE_PROJECT),
            )
        except subprocess.TimeoutExpired:
            logger.warning("PicoGK oracle timed out after %.0fs", timeout_s)
            return _fail_result(voxel_size_mm, notes=["Timeout"])
        except FileNotFoundError:
            logger.error("dotnet CLI not found — PicoGK oracle unavailable")
            return _fail_result(voxel_size_mm, notes=["dotnet not found"])

        if proc.returncode != 0:
            logger.warning("PicoGK oracle failed (rc=%d): %s", proc.returncode, proc.stderr[:500])
            # Try to parse stdout anyway (fail-closed result may be in stdout)
            try:
                result = json.loads(proc.stdout)
                _ORACLE_CACHE[key] = result
                return result
            except (json.JSONDecodeError, ValueError):
                return _fail_result(voxel_size_mm, notes=[f"CLI error: {proc.stderr[:200]}"])

        try:
            result = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse oracle output: %s", e)
            return _fail_result(voxel_size_mm, notes=[f"JSON parse error: {e}"])

    _ORACLE_CACHE[key] = result
    return result


def _fail_result(voxel_size_mm: float, notes: list[str] | None = None) -> dict[str, Any]:
    """Construct a fail-closed result dict."""
    return {
        "passed": False,
        "kerf_buffer_mm": 0.0,
        "t_min_proxy_mm": 0.0,
        "b_max_survivable_mm": 0.0,
        "area_original_mm2": 0.0,
        "area_after_inset_mm2": 0.0,
        "component_count_after_inset": 0,
        "voxel_resolution_mm": voxel_size_mm,
        "notes": notes or [],
    }


def clear_cache() -> None:
    """Clear the oracle result cache."""
    _ORACLE_CACHE.clear()
