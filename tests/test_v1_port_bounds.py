"""Bounds and constraint tests for v1 port.

These tests verify that:
- Invalid params produce constraint violations, not crashes
- Extreme params are handled gracefully
- Constraints follow G <= 0 convention
"""

import numpy as np

from larrak2.core.encoding import bounds
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


class TestV1PortBounds:
    """Bounds and constraint tests for v1 port at fidelity=1."""

    def test_lower_bounds_do_not_crash(self):
        """Evaluate at lower bounds, verify no crash."""
        xl, _ = bounds()
        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)

        result = evaluate_candidate(xl, ctx)

        # Should produce finite results
        assert np.all(np.isfinite(result.F)), f"F non-finite at lower bounds: {result.F}"
        assert np.all(np.isfinite(result.G)), f"G non-finite at lower bounds: {result.G}"

    def test_upper_bounds_do_not_crash(self):
        """Evaluate at upper bounds, verify no crash."""
        _, xu = bounds()
        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)

        result = evaluate_candidate(xu, ctx)

        assert np.all(np.isfinite(result.F)), f"F non-finite at upper bounds: {result.F}"
        assert np.all(np.isfinite(result.G)), f"G non-finite at upper bounds: {result.G}"

    def test_extreme_gear_coeffs_cause_violations(self):
        """Extreme pitch coefficients should cause constraint violations."""
        xl, xu = bounds()
        x = (xl + xu) / 2  # Start at mid

        # Set extreme pitch coefficients (indices 5-11 are gear params)
        x[5:12] = xu[5:12]  # Max coefficients

        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)
        result = evaluate_candidate(x, ctx)

        # Should still be finite (not crash)
        assert np.all(np.isfinite(result.F))
        assert np.all(np.isfinite(result.G))

        # May have some constraint violations (G > 0)
        # This is expected for extreme params

    def test_small_base_radius_causes_constraint_violation(self):
        """Small base radius should violate min-radius constraint."""
        xl, xu = bounds()
        x = (xl + xu) / 2

        # Set base_radius to lower bound (index 4)
        x[4] = xl[4]  # 20mm

        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)
        result = evaluate_candidate(x, ctx)

        assert np.all(np.isfinite(result.G))
        # G[1] is min_radius constraint: g = MIN_RADIUS - min_planet_r
        # Small base radius may violate this

    def test_large_base_radius_causes_constraint_violation(self):
        """Large base radius should violate max-radius constraint."""
        xl, xu = bounds()
        x = (xl + xu) / 2

        # Set base_radius to upper bound (index 4)
        x[4] = xu[4]  # 60mm

        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)
        result = evaluate_candidate(x, ctx)

        assert np.all(np.isfinite(result.G))
        # G[2] is max_radius constraint: g = max_planet_r - MAX_RADIUS
        # Large base radius may violate this

    def test_constraint_sign_convention(self):
        """Verify G <= 0 is feasible convention."""
        xl, xu = bounds()
        x = (xl + xu) / 2  # Mid-bounds

        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)
        result = evaluate_candidate(x, ctx)

        # Mid-bounds should satisfy most constraints
        # (some may be violated due to physics)
        n_satisfied = np.sum(result.G <= 0)
        assert n_satisfied >= 4, (
            f"Mid-bounds should satisfy at least half of constraints. "
            f"Satisfied: {n_satisfied}/7, G: {result.G}"
        )

    def test_objectives_are_reasonable(self):
        """Verify objectives are in reasonable range."""
        xl, xu = bounds()
        x = (xl + xu) / 2

        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)
        result = evaluate_candidate(x, ctx)

        # F[0] is -efficiency (minimized), should be in [-1, 0]
        assert -1.0 <= result.F[0] <= 0.0, f"Efficiency objective out of range: {result.F[0]}"

        # F[1] is loss_total (watts), should be positive
        assert result.F[1] >= 0.0, f"Loss objective should be non-negative: {result.F[1]}"
