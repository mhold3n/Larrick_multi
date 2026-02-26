from unittest.mock import patch

import numpy as np
import pytest

from larrak2.core.encoding import GearParams
from larrak2.gear.manufacturability_limits import (
    ManufacturingProcessParams,
    compute_manufacturable_ratio_rate_limits,
)


class TestStrictBypass:
    """
    Verifies that enabling the PicoGK oracle bypasses strict surrogate checks.
    """

    @pytest.fixture
    def high_amp_params(self):
        """Parameters designed to fail the strict derivative check (d>10)."""
        gear = GearParams(base_radius=40.0, pitch_coeffs=np.zeros(7), face_width_mm=10.0)
        process = ManufacturingProcessParams()
        durations = np.array([5.0])
        amps = np.array([2.0])
        return gear, process, durations, amps

    @patch("larrak2.gear.manufacturability_limits._PICOGK_ENABLED", False)
    def test_strict_mode_rejects_high_derivatives(self, high_amp_params):
        """When Oracle is OFF (default/strict), high derivatives should be rejected."""
        gear, process, dur, amps = high_amp_params

        env = compute_manufacturable_ratio_rate_limits(
            gear, process=process, durations_deg=dur, amplitude_scan=amps
        )
        assert env.max_delta_ratio[0] == 0.0
        pass_counts = env.metadata.get("surrogate_pass_counts", [0])
        assert pass_counts[0] == 0

    @patch("larrak2.gear.manufacturability_limits._PICOGK_ENABLED", True)
    @patch("larrak2.gear.picogk_adapter.evaluate_manufacturability_batch")
    def test_oracle_mode_bypasses_strict_checks(self, mock_batch, high_amp_params):
        """When Oracle is ON, strict checks are bypassed."""
        # Override params with safer ones that pass monotonicity but fail derivative check
        # Amp=1.0, Dur=1.0 deg => d_ratio ~ 180 > 10. Ratio in [2.0, 3.0] (Safe)
        gear = GearParams(base_radius=40.0, pitch_coeffs=np.zeros(7), face_width_mm=10.0)
        process = ManufacturingProcessParams()
        dur = np.array([1.0])
        amps = np.array([1.0])

        # Verify patch worked
        from larrak2.gear import manufacturability_limits

        print(f"DEBUG: _PICOGK_ENABLED in module = {manufacturability_limits._PICOGK_ENABLED}")

        def side_effect(candidates):
            return [{"passed": True, "notes": "Mock pass"} for _ in candidates]

        mock_batch.side_effect = side_effect

        env = compute_manufacturable_ratio_rate_limits(
            gear, process=process, durations_deg=dur, amplitude_scan=amps
        )

        surr_passes = env.metadata.get("surrogate_pass_counts", [0])
        print(f"DEBUG: Pass counts = {surr_passes}")

        # If this fails, it means even with strict=False, candidates are rejected (e.g. by Monotonicity)
        # OR the patch didn't take effect.
        assert surr_passes[0] > 0
        pico_passes = env.metadata.get("picogk_pass_counts", [0])
        assert pico_passes[0] > 0
        assert env.max_delta_ratio[0] > 0.0
