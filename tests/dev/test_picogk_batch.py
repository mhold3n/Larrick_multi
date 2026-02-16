import numpy as np
import logging
import os
import sys
import pytest
from larrak2.gear.manufacturability_limits import compute_manufacturable_ratio_rate_limits, ManufacturingProcessParams
from larrak2.core.encoding import GearParams

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("larrak2")

@pytest.mark.skipif(os.environ.get("LARRAK_PICOGK_ORACLE") != "1", reason="Requires LARRAK_PICOGK_ORACLE=1")
def test_picogk_batch_performance():
    """Performance test for PicoGK batch processing."""
    print(f"Testing Batch Architecture with ORACLE={os.environ.get('LARRAK_PICOGK_ORACLE', '0')}")
    
    gear = GearParams(base_radius=15.0, pitch_coeffs=np.zeros(7))
    proc = ManufacturingProcessParams(
        kerf_mm=0.2, 
        overcut_mm=0.05, 
        min_ligament_mm=0.3
    )
    
    durations = np.array([10.0, 20.0, 30.0, 45.0, 60.0])
    amps = np.linspace(0.0, 1.0, 6)
    
    logger.info("Calling compute_manufacturable_ratio_rate_limits...")
    
    env = compute_manufacturable_ratio_rate_limits(
        gear, 
        process=proc, 
        durations_deg=durations, 
        amplitude_scan=amps
    )
    
    pico_passes = env.metadata.get('picogk_pass_counts')
    surr_passes = env.metadata.get('surrogate_pass_counts')
    
    logger.info(f"Surrogate pass counts: {surr_passes}")
    logger.info(f"PicoGK pass counts: {pico_passes}")
    
    assert pico_passes is not None
    assert sum(pico_passes) > 0

if __name__ == "__main__":
    test_picogk_batch_performance()
