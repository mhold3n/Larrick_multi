import logging
import os
import time

import numpy as np
import pytest

from larrak2.core.encoding import GearParams
from larrak2.gear.manufacturability_limits import (
    ManufacturingProcessParams,
    compute_manufacturable_ratio_rate_limits,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.skipif(
    os.environ.get("LARRAK_PICOGK_ORACLE") != "1", reason="Requires LARRAK_PICOGK_ORACLE=1"
)
def test_picogk_single_performance():
    """Test performance of a single scan point (fast check)."""
    logger.info("Starting PicoGK performance test")

    gear = GearParams(base_radius=15.0, pitch_coeffs=np.zeros(7), face_width_mm=10.0)
    proc = ManufacturingProcessParams(kerf_mm=0.2, overcut_mm=0.05, min_ligament_mm=0.3)

    durations = np.array([45.0])
    amps = np.linspace(0.0, 0.5, 1)  # Single point

    start_time = time.time()

    env = compute_manufacturable_ratio_rate_limits(
        gear, process=proc, durations_deg=durations, amplitude_scan=amps
    )

    duration = time.time() - start_time
    logger.info(f"Test completed in {duration:.2f}s")

    pico_passes = env.metadata.get("picogk_pass_counts")
    assert pico_passes is not None
    assert sum(pico_passes) > 0


if __name__ == "__main__":
    test_picogk_single_performance()
