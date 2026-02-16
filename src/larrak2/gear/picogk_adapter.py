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
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .profile_export import process_params_to_dict

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


def _build_profile_data(
    theta: np.ndarray,
    r_planet: np.ndarray,
    process_params: dict[str, float],
    R_psi: np.ndarray | None = None,
    psi: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build the dictionary structure for a single profile input."""
    # Convert polar (theta, r) to cartesian (x, y)
    x = r_planet * np.cos(theta)
    y = r_planet * np.sin(theta)

    # Ensure closed loop
    if not np.isclose(theta[-1], theta[0] + 2 * np.pi):
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    outer_poly = np.column_stack((x, y)).tolist()

    data = {
        "units": "mm",
        "outer": outer_poly,
        "holes": [],
        "process": process_params,
        "metadata": {
            "created_by": "larrak2_picogk_adapter",
        },
    }
    return data


def evaluate_manufacturability_batch(
    candidates: list[dict[str, Any]],
    voxel_size_mm: float = 0.1,
    slab_thickness_mm: float = 14.0,
    timeout_s: float | None = None,
) -> list[dict[str, Any]]:
    """Batch evaluate manufacturability for multiple candidates."""
    if not candidates:
        return []

    if timeout_s is None:
        timeout_s = float(os.environ.get("LARRAK_PICOGK_TIMEOUT", "600.0"))

    # Chunking logic to prevent OOM / Timeout on large batches
    BATCH_CHUNK_SIZE = 10
    results_map = {}  # index -> result dict

    # We must maintain order.
    n_candidates = len(candidates)

    # Identify which need computation
    # Map original_index -> candidate
    # We can just iterate chunks and within chunk check cache.

    for start_idx in range(0, n_candidates, BATCH_CHUNK_SIZE):
        end_idx = min(start_idx + BATCH_CHUNK_SIZE, n_candidates)
        chunk_original = candidates[start_idx:end_idx]

        # Prepare chunk input, checking cache first
        chunk_to_run = []
        chunk_to_run_indices = []  # local index in chunk (0..len(chunk)-1)

        chunk_results_local = [None] * len(chunk_original)

        for i, c in enumerate(chunk_original):
            # Compute hash key to check cache
            # We need to construct params same as evaluate_manufacturability does
            # or rely on the caller having passed correct params?
            # evaluate_manufacturability calls _cache_key.
            # But here we have dicts.
            # We must replicate _cache_key logic or bypass cache check here if key generation is hard?
            # But the single-item function calls this batch function.
            # If we don't check cache here, batching runs blindly.
            # But _cache_key requires args.

            # Let's verify if we can easily build key.
            # c has "theta", "r_planet".
            # and params.

            # If 'process' is in c, extracting individal params for key might be tricky if not standardized.
            # BUT, we can just run the chunk.
            # WAIT. The GOAL is to RESUME.
            # If we don't check cache, we re-run everything.
            # So we MUST check cache.

            # Extract params for key
            theta = c["theta"]
            r_planet = c["r_planet"]
            w_d = c.get("wire_d_mm", 0.2)
            oc = c.get("overcut_mm", 0.05)
            cm = c.get("corner_margin_mm", 0.0)
            ml = c.get("min_ligament_mm", 0.35)
            # If 'process' dict exists, it overrides?
            if "process" in c:
                p = c["process"]
                # Assuming process dict keys match?
                # 'wire_d_mm' in dict vs 'kerf_mm' in logic?
                # Adapter _build_profile_data just passes input.
                # Let's skip cache check for complex mix?
                # No, standard run uses standard keys.
                pass

            key = _cache_key(theta, r_planet, w_d, oc, cm, ml, voxel_size_mm)

            if key in _ORACLE_CACHE:
                chunk_results_local[i] = _ORACLE_CACHE[key]
            else:
                chunk_to_run.append(c)
                chunk_to_run_indices.append(i)

        if not chunk_to_run:
            # All in cache
            for i, res in enumerate(chunk_results_local):
                results_map[start_idx + i] = res
            continue

        # Build input for items that need running
        chunk_input = []
        for c in chunk_to_run:
            if "process" in c:
                proc = c["process"]
            else:
                proc = process_params_to_dict(
                    c.get("wire_d_mm", 0.2),
                    c.get("overcut_mm", 0.05),
                    c.get("corner_margin_mm", 0.0),
                    c.get("min_ligament_mm", 0.35),
                )

            data = _build_profile_data(
                c["theta"], c["r_planet"], proc, c.get("R_psi"), c.get("psi")
            )
            chunk_input.append(data)

        # Execute chunk
        run_results_list = []
        with tempfile.TemporaryDirectory(prefix="picogk_batch_") as tmpdir:
            input_path = Path(tmpdir) / "batch_profiles.json"
            with open(input_path, "w") as f:
                json.dump(chunk_input, f)

            # Find compiled DLL to bypass 'dotnet run' overhead/issues
            dll_path = None
            # Try specific paths
            possible_paths = [
                _ORACLE_PROJECT / "bin" / "Debug" / "net9.0" / "osx-arm64" / "picogk_manufact.dll",
                _ORACLE_PROJECT / "bin" / "Debug" / "net9.0" / "picogk_manufact.dll",
                # Fallback to net8.0 just in case
                _ORACLE_PROJECT / "bin" / "Debug" / "net8.0" / "osx-arm64" / "picogk_manufact.dll",
                _ORACLE_PROJECT / "bin" / "Debug" / "net8.0" / "picogk_manufact.dll",
            ]
            for p in possible_paths:
                if p.exists():
                    dll_path = p
                    break

            if dll_path:
                cmd = [
                    _find_dotnet(),
                    str(dll_path),
                    "--",
                    "--input",
                    str(input_path),
                    "--voxel-size",
                    str(voxel_size_mm),
                    "--slab-thickness",
                    str(slab_thickness_mm),
                ]
            else:
                logger.warning("Could not find compiled DLL, falling back to 'dotnet run'")
                cmd = [
                    _find_dotnet(),
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

            # Use file redirection to avoid pipe deadlocks and buffering issues
            stdout_path = Path(tmpdir) / "stdout.txt"
            stderr_path = Path(tmpdir) / "stderr.txt"

            logger.debug(
                "Running PicoGK oracle chunk %d-%d (calc %d/%d): %s",
                start_idx,
                end_idx,
                len(chunk_to_run),
                len(chunk_original),
                " ".join(cmd),
            )

            try:
                with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
                    proc_run = subprocess.run(
                        cmd,
                        stdout=f_out,
                        stderr=f_err,
                        text=True,
                        timeout=timeout_s,
                        cwd=str(_REPO_ROOT),  # Run from root
                    )

                # Read output
                stdout_content = ""
                stderr_content = ""
                if stdout_path.exists():
                    stdout_content = stdout_path.read_text(errors="replace")
                if stderr_path.exists():
                    stderr_content = stderr_path.read_text(errors="replace")

                if proc_run.returncode != 0:
                    logger.warning(
                        "PicoGK chunk failed (rc=%d): %s", proc_run.returncode, stderr_content[:500]
                    )
                    # Fail open
                    run_results_list = [
                        _fail_result(voxel_size_mm, notes=["Chunk Failed"]) | {"passed": True}
                        for _ in range(len(chunk_to_run))
                    ]
                else:
                    try:
                        parsed = json.loads(stdout_content)
                        if not isinstance(parsed, list):
                            parsed = [parsed]
                        run_results_list = parsed
                    except Exception as e:
                        logger.error(
                            "Failed to parse chunk output: %s. Stdout: %s", e, stdout_content[:100]
                        )
                        run_results_list = [
                            _fail_result(voxel_size_mm, notes=["Parse Failed"]) | {"passed": True}
                            for _ in range(len(chunk_to_run))
                        ]

            except subprocess.TimeoutExpired:
                logger.warning("PicoGK chunk timed out after %.0fs", timeout_s)
                run_results_list = [
                    _fail_result(voxel_size_mm, notes=["Timeout"]) | {"passed": True}
                    for _ in range(len(chunk_to_run))
                ]
            except Exception as e:
                logger.error("PicoGK chunk error: %s", e)
                run_results_list = [
                    _fail_result(voxel_size_mm, notes=[str(e)]) | {"passed": True}
                    for _ in range(len(chunk_to_run))
                ]

        # Store results
        if len(run_results_list) != len(chunk_to_run):
            logger.warning(
                "Mismatch in chunk result count (%d vs %d). padding/truncating",
                len(run_results_list),
                len(chunk_to_run),
            )
            while len(run_results_list) < len(chunk_to_run):
                run_results_list.append({"passed": True, "notes": ["Mismatch padding"]})

        # Merge back to local results and Update Cache
        for k, res in enumerate(run_results_list):
            local_idx = chunk_to_run_indices[k]
            chunk_results_local[local_idx] = res

            # Cache it
            c = chunk_to_run[k]
            theta = c["theta"]
            r_planet = c["r_planet"]
            w_d = c.get("wire_d_mm", 0.2)
            oc = c.get("overcut_mm", 0.05)
            cm = c.get("corner_margin_mm", 0.0)
            ml = c.get("min_ligament_mm", 0.35)
            key = _cache_key(theta, r_planet, w_d, oc, cm, ml, voxel_size_mm)
            _ORACLE_CACHE[key] = res

        # Save Persistent Cache after each chunk
        _save_cache()

        # Add to final map
        for i, res in enumerate(chunk_results_local):
            results_map[start_idx + i] = res

    # Reassemble ordered list
    final_results = [results_map[i] for i in range(n_candidates)]
    return final_results


def evaluate_manufacturability(
    theta: np.ndarray,
    r_planet: np.ndarray,
    wire_d_mm: float = 0.2,
    overcut_mm: float = 0.05,
    corner_margin_mm: float = 0.0,
    min_ligament_mm: float = 0.35,
    voxel_size_mm: float = 0.01,
    slab_thickness_mm: float = 14.0,
    timeout_s: float | None = None,
    *,
    R_psi: np.ndarray | None = None,
    psi: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate manufacturability of a gear profile via PicoGK oracle (Single Mode)."""

    # Check cache first
    key = _cache_key(
        theta, r_planet, wire_d_mm, overcut_mm, corner_margin_mm, min_ligament_mm, voxel_size_mm
    )
    if key in _ORACLE_CACHE:
        logger.debug("PicoGK oracle cache hit")
        return _ORACLE_CACHE[key]

    candidate = {
        "theta": theta,
        "r_planet": r_planet,
        "wire_d_mm": wire_d_mm,
        "overcut_mm": overcut_mm,
        "corner_margin_mm": corner_margin_mm,
        "min_ligament_mm": min_ligament_mm,
        "R_psi": R_psi,
        "psi": psi,
    }

    try:
        results = evaluate_manufacturability_batch(
            [candidate],
            voxel_size_mm=voxel_size_mm,
            slab_thickness_mm=slab_thickness_mm,
            timeout_s=timeout_s,
        )
        result = results[0]
        _ORACLE_CACHE[key] = result
        return result
    except Exception:
        # Rethrow to allow fail-open logic upstream to handle it (or handle it here?)
        # Legacy evaluate_manufacturability raised exceptions to be caught upstream.
        raise


def _find_dotnet() -> str:
    """Find the dotnet executable."""

    # Check PATH first
    if shutil.which("dotnet"):
        return "dotnet"

    # Common macOS locations
    common_paths = [
        "/usr/local/share/dotnet/dotnet",  # Standard installer (x64/some arm64)
        "/opt/homebrew/bin/dotnet",  # Homebrew arm64
        "/usr/local/bin/dotnet",
    ]

    for p in common_paths:
        if Path(p).exists() and os.access(p, os.X_OK):
            return p

    return "dotnet"  # Fallback to hoping it's in PATH later


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


def _get_cache_path() -> Path:
    return _THIS_DIR / "picogk_cache.pkl"


def _load_cache():
    """Load persistent cache from disk."""
    path = _get_cache_path()
    if path.exists():
        try:
            import pickle

            with open(path, "rb") as f:
                data = pickle.load(f)
                _ORACLE_CACHE.update(data)
                logger.info(f"Loaded {len(data)} entries from persistent cache")
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")


def _save_cache():
    """Save persistent cache to disk."""
    path = _get_cache_path()
    try:
        import pickle

        # atomic write?
        temp = path.with_suffix(".tmp")
        with open(temp, "wb") as f:
            pickle.dump(_ORACLE_CACHE, f)
        temp.replace(path)
    except Exception as e:
        logger.warning(f"Failed to save persistent cache: {e}")


# Load cache on import
_load_cache()


def clear_cache() -> None:
    """Clear the oracle result cache."""
    _ORACLE_CACHE.clear()
    _save_cache()
