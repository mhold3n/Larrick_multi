"""Smoke tests for v1 port at fidelity=1.

These tests verify that the v1 port produces valid results:
- Finite F/G values
- Correct shapes
- No crashes on mid-bounds candidate
"""

import numpy as np

from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


class TestV1PortSmoke:
    """Smoke tests for v1 port at fidelity=1."""

    def test_fidelity_1_produces_finite_results(self):
        """Evaluate mid-bounds candidate at fidelity=1, verify finite F/G."""
        x = mid_bounds_candidate()
        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)

        result = evaluate_candidate(x, ctx)

        # F should be finite (2 objectives)
        assert result.F.shape == (2,), f"Expected F.shape=(2,), got {result.F.shape}"
        assert np.all(np.isfinite(result.F)), f"F contains non-finite values: {result.F}"

        # G should be finite (7 constraints)
        assert result.G.shape == (7,), f"Expected G.shape=(7,), got {result.G.shape}"
        assert np.all(np.isfinite(result.G)), f"G contains non-finite values: {result.G}"

    def test_fidelity_1_diag_contains_v1_flag(self):
        """Verify diagnostics indicate v1 port was used."""
        x = mid_bounds_candidate()
        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)

        result = evaluate_candidate(x, ctx)

        # Both thermo and gear diag should have v1_port=True
        assert result.diag.get("thermo", {}).get("v1_port") is True, (
            "Thermo diag should indicate v1_port"
        )
        assert result.diag.get("gear", {}).get("v1_port") is True, (
            "Gear diag should indicate v1_port"
        )

    def test_fidelity_0_still_works(self):
        """Verify fidelity=0 still uses toy physics."""
        x = mid_bounds_candidate()
        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=42)

        result = evaluate_candidate(x, ctx)

        # Should produce valid results
        assert result.F.shape == (2,)
        assert result.G.shape == (7,)
        assert np.all(np.isfinite(result.F))
        assert np.all(np.isfinite(result.G))

        # v1_port flag should be absent or False for fidelity=0
        thermo_v1 = result.diag.get("thermo", {}).get("v1_port", False)
        gear_v1 = result.diag.get("gear", {}).get("v1_port", False)
        assert not thermo_v1, "Fidelity=0 should not use v1 thermo"
        assert not gear_v1, "Fidelity=0 should not use v1 gear"

    def test_multiple_candidates_at_fidelity_1(self):
        """Evaluate several random candidates at fidelity=1."""
        rng = np.random.default_rng(seed=123)
        ctx = EvalContext(rpm=4000.0, torque=150.0, fidelity=1, seed=123)

        from larrak2.core.encoding import bounds

        xl, xu = bounds()

        for i in range(5):
            x = xl + rng.random(len(xl)) * (xu - xl)
            result = evaluate_candidate(x, ctx)

            assert np.all(np.isfinite(result.F)), f"F non-finite for candidate {i}"
            assert np.all(np.isfinite(result.G)), f"G non-finite for candidate {i}"

    def test_low_rpm_torque_does_not_crash(self):
        """Low rpm/torque edge case should not crash."""
        x = mid_bounds_candidate()
        ctx = EvalContext(rpm=100.0, torque=10.0, fidelity=1, seed=42)

        result = evaluate_candidate(x, ctx)

        assert np.all(np.isfinite(result.F))
        assert np.all(np.isfinite(result.G))
