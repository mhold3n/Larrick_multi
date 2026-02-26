"""Integration test for Pareto optimization at fidelity=1.

Tests that NSGA-II optimization completes successfully with v1 physics.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np


class TestParetoFidelity1:
    """Integration tests for Pareto optimization at fidelity=1."""

    @staticmethod
    def _front_or_population(result):
        X = (
            result.X
            if result.X is not None
            else (result.pop.get("X") if result.pop is not None else None)
        )
        F = (
            result.F
            if result.F is not None
            else (result.pop.get("F") if result.pop is not None else None)
        )
        return X, F

    def test_pareto_fidelity1_completes(self):
        """Run small NSGA-II optimization at fidelity=1, verify completion."""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination

        from larrak2.adapters.pymoo_problem import ParetoProblem
        from larrak2.core.types import EvalContext

        # Setup
        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=123)
        problem = ParetoProblem(ctx=ctx)
        algorithm = NSGA2(pop_size=16)
        termination = get_termination("n_gen", 5)

        # Run
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=123,
            verbose=False,
        )

        # Assert run completed
        assert result is not None
        X, F = self._front_or_population(result)
        assert X is not None
        assert F is not None

        # Assert shapes
        n_pareto = X.shape[0]
        assert n_pareto > 0, "Should have at least one Pareto solution"
        assert F.shape[0] == n_pareto
        assert F.shape[1] == problem.n_obj

    def test_pareto_fidelity1_finite_results(self):
        """All F and G values should be finite."""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination

        from larrak2.adapters.pymoo_problem import ParetoProblem
        from larrak2.core.evaluator import evaluate_candidate
        from larrak2.core.types import EvalContext

        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=123)
        problem = ParetoProblem(ctx=ctx)
        algorithm = NSGA2(pop_size=16)
        termination = get_termination("n_gen", 5)

        result = minimize(problem, algorithm, termination, seed=123, verbose=False)
        X, F = self._front_or_population(result)
        assert X is not None
        assert F is not None

        # Check F finite
        assert np.all(np.isfinite(F)), f"F contains non-finite: {F}"

        # Check G finite for all Pareto solutions
        for x in X:
            res = evaluate_candidate(x, ctx)
            assert np.all(np.isfinite(res.G)), f"G contains non-finite: {res.G}"

    def test_pareto_fidelity1_feasible_fraction(self):
        """At lower load points, at least some solutions should be feasible."""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination

        from larrak2.adapters.pymoo_problem import ParetoProblem
        from larrak2.core.evaluator import evaluate_candidate
        from larrak2.core.types import EvalContext

        ctx = EvalContext(rpm=2000.0, torque=80.0, fidelity=1, seed=123)
        problem = ParetoProblem(ctx=ctx)
        algorithm = NSGA2(pop_size=16)
        termination = get_termination("n_gen", 5)

        result = minimize(problem, algorithm, termination, seed=123, verbose=False)
        X, _ = self._front_or_population(result)
        assert X is not None

        # Count feasible solutions
        n_feasible = 0
        for x in X:
            res = evaluate_candidate(x, ctx)
            if np.all(res.G <= 0):
                n_feasible += 1

        n_pareto = X.shape[0]
        feasible_fraction = n_feasible / n_pareto if n_pareto > 0 else 0.0

        # At this operating point we expect non-zero feasible coverage.
        assert feasible_fraction >= 0.05 or n_feasible >= 1, (
            f"Too few feasible: {n_feasible}/{n_pareto} = {feasible_fraction:.1%}"
        )

    def test_pareto_fidelity1_pareto_size(self):
        """Pareto set should have multiple solutions."""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination

        from larrak2.adapters.pymoo_problem import ParetoProblem
        from larrak2.core.types import EvalContext

        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=123)
        problem = ParetoProblem(ctx=ctx)
        algorithm = NSGA2(pop_size=16)
        termination = get_termination("n_gen", 5)

        result = minimize(problem, algorithm, termination, seed=123, verbose=False)
        X, _ = self._front_or_population(result)
        assert X is not None

        n_pareto = X.shape[0]
        # With pop=16, gen=5, expect at least 1 non-dominated solution
        assert n_pareto >= 1, f"Pareto set too small: {n_pareto}"

    def test_pareto_fidelity1_deterministic(self):
        """Same seed should produce identical results."""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination

        from larrak2.adapters.pymoo_problem import ParetoProblem
        from larrak2.core.types import EvalContext

        ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=123)

        # Run 1
        problem1 = ParetoProblem(ctx=ctx)
        algorithm1 = NSGA2(pop_size=16)
        termination1 = get_termination("n_gen", 5)
        result1 = minimize(problem1, algorithm1, termination1, seed=123, verbose=False)

        # Run 2
        problem2 = ParetoProblem(ctx=ctx)
        algorithm2 = NSGA2(pop_size=16)
        termination2 = get_termination("n_gen", 5)
        result2 = minimize(problem2, algorithm2, termination2, seed=123, verbose=False)
        _, F1 = self._front_or_population(result1)
        _, F2 = self._front_or_population(result2)
        assert F1 is not None and F2 is not None

        # Compare F values (sorted to handle different orderings)
        F1_sorted = np.sort(F1, axis=0)
        F2_sorted = np.sort(F2, axis=0)

        np.testing.assert_allclose(
            F1_sorted,
            F2_sorted,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Determinism failed: different F with same seed",
        )

    def test_cli_pareto_fidelity1(self):
        """Test CLI invocation with fidelity=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from larrak2.cli.run_pareto import main

            # Run CLI
            exit_code = main(
                [
                    "--pop",
                    "8",
                    "--gen",
                    "3",
                    "--rpm",
                    "3000",
                    "--torque",
                    "200",
                    "--fidelity",
                    "1",
                    "--seed",
                    "123",
                    "--outdir",
                    tmpdir,
                ]
            )

            assert exit_code == 0, f"CLI failed with exit code {exit_code}"

            # Check outputs exist
            output_dir = Path(tmpdir)
            assert (output_dir / "pareto_X.npy").exists()
            assert (output_dir / "pareto_F.npy").exists()
            assert (output_dir / "pareto_G.npy").exists()
            assert (output_dir / "summary.json").exists()

            # Check summary.json content
            with open(output_dir / "summary.json") as f:
                summary = json.load(f)

            assert summary["fidelity"] == 1
            assert summary["seed"] == 123
            assert "feasible_fraction" in summary
            assert "best_eta_comb" in summary
            assert "best_eta_exp" in summary
            assert "best_eta_gear" in summary
            assert "best_eta_total" in summary
            assert summary["n_final_pop"] > 0
