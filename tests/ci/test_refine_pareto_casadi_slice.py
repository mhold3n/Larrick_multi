"""Refine Pareto candidates with slice metadata checks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from larrak2.adapters.casadi_refine import RefinementResult
from larrak2.cli.refine_pareto import main as refine_main
from larrak2.core.encoding import N_TOTAL, mid_bounds_candidate
from larrak2.core.types import EvalResult
from larrak2.optimization.slicing.slice_problem import SliceSolveResult


def test_refine_pareto_slice_metadata_and_full_dimensionality():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        x0 = mid_bounds_candidate()
        X = np.tile(x0, (1, 1))
        F = np.zeros((1, 3), dtype=np.float64)
        G = np.zeros((1, 10), dtype=np.float64)
        np.save(tmp / "pareto_X.npy", X)
        np.save(tmp / "pareto_F.npy", F)
        np.save(tmp / "pareto_G.npy", G)

        stack_model_path = tmp / "stack_f1_surrogate.npz"
        stack_model_path.write_bytes(b"placeholder")

        def _mock_slice_solve(*args, **kwargs):
            x0_local = np.asarray(args[0], dtype=np.float64)
            return SliceSolveResult(
                x_opt=x0_local,
                success=True,
                message="mock casadi solve ok",
                ipopt_status="Solve_Succeeded",
                iterations=5,
                diagnostics={
                    "nlp_formulation": "global_surrogate_symbolic",
                    "surrogate_stack_version": "testhash001",
                    "validation_attempts": 1,
                    "trust_radius_final": kwargs.get("trust_radius"),
                    "thermo_symbolic_mode": "strict",
                    "thermo_symbolic_used": True,
                    "thermo_symbolic_version": "thermohash123",
                    "thermo_symbolic_path": "outputs/artifacts/surrogates/thermo_symbolic/thermo_symbolic_f1.npz",
                    "thermo_symbolic_overlay_objectives": ["eta_comb_gap"],
                    "thermo_symbolic_overlay_constraints": ["mass_balance"],
                    "thermo_symbolic_error": "",
                },
            )

        def _mock_select_active_set(*_args, **_kwargs):
            from larrak2.optimization.slicing.active_set import SliceSelection

            active = [0, 1, 2, 3, 4, 5]
            frozen = [i for i in range(N_TOTAL) if i not in active]
            return SliceSelection(
                active_indices=active,
                frozen_indices=frozen,
                scores=[1.0 for _ in active],
            )

        def _mock_eval_candidate(x, _ctx):
            x_arr = np.asarray(x, dtype=np.float64)
            return EvalResult(
                F=np.array(
                    [
                        float(np.sum(x_arr)),
                        float(np.mean(x_arr)),
                        0.1,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dtype=np.float64,
                ),
                G=np.zeros(23, dtype=np.float64),
                diag={},
            )

        from unittest.mock import patch

        with (
            patch("larrak2.adapters.casadi_refine.solve_slice_with_ipopt", _mock_slice_solve),
            patch("larrak2.adapters.casadi_refine.select_active_set", _mock_select_active_set),
            patch("larrak2.adapters.casadi_refine.evaluate_candidate", _mock_eval_candidate),
        ):
            code = refine_main(
                [
                    "--input",
                    tmpdir,
                    "--top-k",
                    "1",
                    "--backend",
                    "casadi",
                    "--stack-model-path",
                    str(stack_model_path),
                    "--slice-method",
                    "sensitivity",
                    "--active-k",
                    "6",
                    "--thermo-symbolic-mode",
                    "strict",
                ]
            )
            assert code == 0

        refined_X = np.load(tmp / "refined_X.npy")
        assert refined_X.shape == (1, N_TOTAL)

        summary = json.loads((tmp / "refinement_summary.json").read_text(encoding="utf-8"))
        assert summary["backend"] == "casadi"
        assert summary["slice_method"] == "sensitivity"
        assert summary["n_refined"] == 1

        row = summary["results"][0]
        active = row["active_indices"]
        frozen = row["frozen_indices"]
        assert isinstance(active, list)
        assert isinstance(frozen, list)
        assert sorted(active + frozen) == list(range(N_TOTAL))
        assert row["backend_used"] == "casadi"
        assert row["nlp_formulation"] == "global_surrogate_symbolic"
        assert row["surrogate_stack_version"] == "testhash001"
        assert row["thermo_symbolic_mode"] == "strict"
        assert row["thermo_symbolic_used"] is True
        assert row["thermo_symbolic_version"] == "thermohash123"
        assert row["thermo_symbolic_overlay_objectives"] == ["eta_comb_gap"]
        assert row["thermo_symbolic_overlay_constraints"] == ["mass_balance"]


def test_refine_pareto_defaults_resolve_artifacts_by_fidelity(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        x0 = mid_bounds_candidate()
        np.save(tmp / "pareto_X.npy", np.tile(x0, (1, 1)))
        np.save(tmp / "pareto_F.npy", np.zeros((1, 6), dtype=np.float64))
        np.save(tmp / "pareto_G.npy", np.zeros((1, 10), dtype=np.float64))

        calls: dict[str, tuple[int, bool]] = {}

        def _mock_resolve_stack(*, fidelity, explicit_path=None, must_exist=True):
            _ = explicit_path
            calls["stack"] = (int(fidelity), bool(must_exist))
            return tmp / "outputs/artifacts/surrogates/stack_f2/stack_f2_surrogate.npz"

        def _mock_resolve_thermo(*, fidelity, explicit_path=None, must_exist=True):
            _ = explicit_path
            calls["thermo"] = (int(fidelity), bool(must_exist))
            return (
                tmp
                / "outputs/artifacts/surrogates/thermo_symbolic_f2/thermo_symbolic_f2.npz"
            )

        def _mock_refine_candidate(x0, ctx, **_kwargs):
            eval_res = EvalResult(
                F=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float64),
                G=np.zeros(23, dtype=np.float64),
                diag={},
            )
            return RefinementResult(
                x_refined=np.asarray(x0, dtype=np.float64),
                F_refined=eval_res.F,
                G_refined=eval_res.G,
                diag={
                    "active_indices": [],
                    "frozen_indices": list(range(N_TOTAL)),
                    "thermo_symbolic_mode": str(ctx.thermo_symbolic_mode),
                },
                success=True,
                message="mock refine ok",
                backend_used="casadi",
                ipopt_status="Solve_Succeeded",
            )

        monkeypatch.setattr("larrak2.cli.refine_pareto.resolve_stack_artifact_path", _mock_resolve_stack)
        monkeypatch.setattr(
            "larrak2.cli.refine_pareto.resolve_thermo_symbolic_artifact_path",
            _mock_resolve_thermo,
        )
        monkeypatch.setattr("larrak2.adapters.casadi_refine.refine_candidate", _mock_refine_candidate)

        code = refine_main(
            [
                "--input",
                tmpdir,
                "--top-k",
                "1",
                "--backend",
                "casadi",
                "--fidelity",
                "2",
            ]
        )
        assert code == 0
        assert calls["stack"] == (2, True)
        assert calls["thermo"] == (2, True)

        summary = json.loads((tmp / "refinement_summary.json").read_text(encoding="utf-8"))
        assert summary["stack_model_path"].endswith("stack_f2/stack_f2_surrogate.npz")
        assert summary["thermo_symbolic_artifact_path"].endswith(
            "thermo_symbolic_f2/thermo_symbolic_f2.npz"
        )
