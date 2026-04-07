import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
from larrak_engines.gear.manufacturability_limits import (
    ManufacturingProcessParams,
    _build_all_candidates,
    _surrogate_check,
)
from larrak_runtime.core.encoding import GearParams

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyze_partial")


def _cache_key(
    theta: np.ndarray,
    r_planet: np.ndarray,
    wire_d_mm: float,
    overcut_mm: float,
    corner_margin_mm: float,
    min_ligament_mm: float,
    voxel_size_mm: float,
) -> str:
    """Hash profile + params for memoization (Copied from picogk_adapter.py)."""
    h = hashlib.sha256()
    h.update(np.asarray(theta, dtype=np.float64).tobytes())
    h.update(np.asarray(r_planet, dtype=np.float64).tobytes())
    h.update(
        f"{wire_d_mm},{overcut_mm},{corner_margin_mm},{min_ligament_mm},{voxel_size_mm}".encode()
    )
    return h.hexdigest()


def analyze_partial():
    # 1. Load Cache
    cache_path = Path("src/larrak2/gear/picogk_cache.pkl")
    if not cache_path.exists():
        print("No cache found!")
        return

    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    print(f"Loaded {len(cache_data)} cached results.")

    # 2. Reconstruct Candidates
    _gear = GearParams(base_radius=15.0, pitch_coeffs=[0.0])
    process = ManufacturingProcessParams(
        kerf_mm=0.2,
        overcut_mm=0.05,
        min_ligament_mm=0.25,  # UPDATED
        min_feature_radius_mm=0.2,
        max_pressure_angle_deg=35.0,
    )

    # Durations and Amplitudes
    from larrak_engines.gear.manufacturability_limits import DEFAULT_DURATION_GRID_DEG

    durations = DEFAULT_DURATION_GRID_DEG.astype(float)
    amps = np.linspace(-1.5, 4.0, 61)

    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)

    feasible_delta = np.zeros_like(durations, dtype=float)

    evaluated_count = 0
    passed_count = 0

    print("\n--- Analysis by Duration ---")

    # We must iterate in the same order to match expectations, but here we just check for existence.

    for i, duration_deg in enumerate(durations):
        duration_evaluated = 0
        duration_passed = 0
        max_amp_found = 0.0

        # Check if we have ANY data for this duration
        # If not, we can skip printing detailed stats if we want, but let's see.

        for j, amp in enumerate(amps):
            shape_candidates = _build_all_candidates(theta, float(duration_deg), float(amp))

            # Phase 1 Surrogate Check
            valid_shapes = []
            for shape_name, ratio_profile in shape_candidates:
                if _surrogate_check(theta, ratio_profile, process, strict=False):
                    valid_shapes.append((shape_name, ratio_profile))

            if not valid_shapes:
                continue

            any_shape_passed = False

            for shape_name, ratio_profile in valid_shapes:
                ratio_safe = np.maximum(ratio_profile, 1e-6)
                r_planet = 80.0 / ratio_safe

                key = _cache_key(
                    theta,
                    r_planet,
                    process.kerf_mm,
                    process.overcut_mm,
                    0.0,  # corner_margin_mm (default 0.0 in adapter)
                    process.min_ligament_mm,
                    0.1,  # voxel_size_mm (default 0.1 in adapter)
                )

                if key in cache_data:
                    duration_evaluated += 1
                    evaluated_count += 1
                    res = cache_data[key]
                    if res.get("passed", False):
                        duration_passed += 1
                        passed_count += 1
                        any_shape_passed = True

            # If we evaluated this amp (cache hit) and it passed, update max
            if any_shape_passed:
                if abs(amp) > max_amp_found:
                    max_amp_found = abs(amp)
            elif duration_evaluated == 0:
                # If we haven't found any evaluated items for this duration implies
                # we haven't reached this part of the grid in the run?
                pass

        feasible_delta[i] = max_amp_found
        print(
            f"Duration {duration_deg:5.1f} deg: Evaluated {duration_evaluated} shapes. Passed {duration_passed}. Max Delta Ratio: {max_amp_found:.3f}"
        )

    print("\n--- Summary ---")
    print(f"Total Evaluated Shapes: {evaluated_count}")
    print(f"Total Passed Shapes:    {passed_count}")

    # Save partial results
    results = {
        "durations": durations,
        "max_delta_ratio": feasible_delta,
        "process": process,
        "metadata": {"partial": True, "evaluated_count": evaluated_count},
    }
    with open("partial_doe_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved 'partial_doe_results.pkl'.")


if __name__ == "__main__":
    analyze_partial()
