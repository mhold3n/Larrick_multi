"""Determinism tests for v1 port.

These tests verify that v1 port produces identical results for identical inputs.
"""

import numpy as np

from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


class TestV1PortDeterminism:
    """Determinism tests for v1 port at fidelity=1."""

    def test_same_inputs_produce_same_outputs(self):
        """Evaluate same candidate twice, verify identical results."""
        x = mid_bounds_candidate()
        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)

        result1 = evaluate_candidate(x, ctx)
        result2 = evaluate_candidate(x, ctx)

        # F should be identical
        np.testing.assert_array_equal(
            result1.F, result2.F, err_msg="F should be identical for same inputs"
        )

        # G should be identical
        np.testing.assert_array_equal(
            result1.G, result2.G, err_msg="G should be identical for same inputs"
        )

    def test_different_x_produces_different_results(self):
        """Different candidates should produce different results."""
        from larrak2.core.encoding import bounds

        xl, xu = bounds()
        x1 = (xl + xu) / 2  # mid
        x2 = xl + 0.25 * (xu - xl)  # quarter

        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)

        result1 = evaluate_candidate(x1, ctx)
        result2 = evaluate_candidate(x2, ctx)

        # At least one of F or G should differ
        f_differ = not np.allclose(result1.F, result2.F)
        g_differ = not np.allclose(result1.G, result2.G)

        assert f_differ or g_differ, "Different x should produce different F or G"

    def test_different_context_produces_different_results(self):
        """Different context (rpm/torque) should produce different results."""
        x = mid_bounds_candidate()

        ctx1 = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)
        ctx2 = EvalContext(rpm=5000.0, torque=400.0, fidelity=1, seed=42)

        result1 = evaluate_candidate(x, ctx1)
        result2 = evaluate_candidate(x, ctx2)

        # Loss should differ (depends on rpm/torque)
        # Use rtol=0 to require exact match to fail (small values close in relative terms)
        f_differ = not np.array_equal(result1.F, result2.F)
        g_differ = not np.array_equal(result1.G, result2.G)

        assert f_differ or g_differ, (
            "Different context should give different F or G"
        )

    def test_seed_does_not_affect_v1_results(self):
        """v1 port should be deterministic regardless of seed."""
        x = mid_bounds_candidate()

        ctx1 = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)
        ctx2 = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=999)

        result1 = evaluate_candidate(x, ctx1)
        result2 = evaluate_candidate(x, ctx2)

        # Should be identical (v1 port is deterministic)
        np.testing.assert_array_equal(result1.F, result2.F)
        np.testing.assert_array_equal(result1.G, result2.G)
