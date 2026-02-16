import logging
import logging.config
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from larrak2.core.encoding import GearParams
from larrak2.gear.manufacturability_limits import (
    ManufacturingProcessParams,
    compute_manufacturable_ratio_rate_limits,
)

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("doe_run.log")],
)
logger = logging.getLogger("run_full_doe")


def run_doe():
    logger.info("Starting Full Manufacturability DOE")
    logger.info(f"LARRAK_PICOGK_ORACLE: {os.environ.get('LARRAK_PICOGK_ORACLE', 'Not Set')}")
    logger.info(f"LARRAK_PICOGK_TIMEOUT: {os.environ.get('LARRAK_PICOGK_TIMEOUT', 'Default')}")

    # Standard gear for the study
    gear = GearParams(base_radius=15.0, pitch_coeffs=np.zeros(7))

    # Process params
    proc = ManufacturingProcessParams()  # defaults

    logger.info("Computing limits (this may take time)...")
    start_time = time.time()

    try:
        # Uses default grids (13 durations * 61 amps)
        envelope = compute_manufacturable_ratio_rate_limits(gear, process=proc)

        duration = time.time() - start_time
        logger.info(f"DOE completed in {duration:.2f}s")

        # Save results
        output_file = Path("doe_results.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(envelope, f)
        logger.info(f"Results saved to {output_file.absolute()}")

        # Summary
        meta = envelope.metadata
        logger.info("--- DOE Summary ---")
        logger.info(f"Total Grid Points: {meta.get('total_grid_points')}")
        logger.info(f"Surrogate Passes: {sum(meta.get('surrogate_pass_counts', []))}")
        logger.info(f"PicoGK Passes: {sum(meta.get('picogk_pass_counts', []))}")
        logger.info(f"Litvin Calls: {meta.get('total_litvin_calls')}")

    except Exception as e:
        logger.error(f"DOE Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_doe()
