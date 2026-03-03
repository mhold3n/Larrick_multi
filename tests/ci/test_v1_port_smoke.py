"""Smoke tests for v1 port at fidelity=1.

These tests verify that the v1 port produces valid results:
- Finite F/G values
- Correct shapes
- No crashes on mid-bounds candidate
"""

import numpy as np

from larrak2.core.constraints import get_constraint_names
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

        # F should be finite (multi-objective vector)
        assert result.F.ndim == 1 and result.F.size >= 3, (
            f"Expected 1D objective vector with >=3 axes, got {result.F.shape}"
        )
        assert np.all(np.isfinite(result.F)), f"F contains non-finite values: {result.F}"

        # G should match centralized constraint registry for fidelity=1
        expected_n_constr = len(get_constraint_names(ctx.fidelity))
        assert result.G.shape == (expected_n_constr,), (
            f"Expected G.shape=({expected_n_constr},), got {result.G.shape}"
        )
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
        """Verify fidelity=0 still routes through strict two-zone thermo backend."""
        x = mid_bounds_candidate()
        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=0, seed=42)

        result = evaluate_candidate(x, ctx)

        # Should produce valid results
        assert result.F.ndim == 1 and result.F.size >= 3
        assert result.G.shape == (len(get_constraint_names(ctx.fidelity)),)
        assert np.all(np.isfinite(result.F))
        assert np.all(np.isfinite(result.G))

        # The thermo backend remains strict equation-first; compatibility v1 flag is fidelity-scoped.
        assert result.diag.get("thermo", {}).get("thermo_solver_status") == "ok"
        assert result.diag.get("thermo", {}).get("thermo_model_version") == "two_zone_eq_v1"

        # v1_port compatibility flag remains fidelity-scoped.
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
        ctx = EvalContext(rpm=1000.0, torque=40.0, fidelity=1, seed=42)

        result = evaluate_candidate(x, ctx)

        assert np.all(np.isfinite(result.F))
        assert np.all(np.isfinite(result.G))
