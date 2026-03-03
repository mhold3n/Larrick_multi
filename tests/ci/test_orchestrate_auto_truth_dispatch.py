"""Auto truth-dispatch policy behavior checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.core.encoding import N_TOTAL, bounds
from larrak2.orchestration.orchestrator import OrchestrationConfig, Orchestrator


class _StubCEM:
    def generate_batch(
        self,
        params: dict[str, Any],  # noqa: ARG002
        n: int,  # noqa: ARG002
        *,
        rng: np.random.Generator,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        xl, xu = bounds()
        x0 = (xl + xu) * 0.5
        out: list[dict[str, Any]] = []
        for i in range(4):
            out.append(
                {
                    "x": np.asarray(x0, dtype=np.float64).reshape(N_TOTAL),
                    "id": f"cand-{i}",
                    "feasible": True,
                    "feasibility_score": [0.95, 0.90, 0.75, 0.85][i],
                    "truth_objective": [0.1, 0.2, 0.3, 0.4][i],
                }
            )
        return out

    def check_feasibility(self, candidate: dict[str, Any]) -> tuple[bool, float]:
        return bool(candidate.get("feasible", True)), float(candidate.get("feasibility_score", 1.0))

    def repair(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return candidate

    def update_distribution(self, history: list[dict[str, Any]]) -> None:  # noqa: ARG002
        return None


class _StubSurrogate:
    def predict(self, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        # Ordered to ensure deterministic uncertainty-prioritized auto selection:
        # candidate 1 has highest uncertainty, then candidate 3.
        pred = np.array([0.50, 0.55, 0.40, 0.52], dtype=np.float64)
        unc = np.array([0.20, 0.90, 0.10, 0.60], dtype=np.float64)
        return pred[: len(candidates)], unc[: len(candidates)]

    def update(self, data: list[tuple[dict[str, Any], float]]) -> None:  # noqa: ARG002
        return None


class _StubSolver:
    def refine(
        self,
        candidate: dict[str, Any],
        *,
        context,  # noqa: ARG002
        max_step: np.ndarray,  # noqa: ARG002
    ) -> dict[str, Any]:
        out = dict(candidate)
        out["solver_success"] = True
        out["solver_diag"] = {
            "thermo_symbolic_mode": "strict",
            "thermo_symbolic_used": True,
            "thermo_symbolic_version": "thermover-orch-1",
            "thermo_symbolic_path": "outputs/artifacts/surrogates/thermo_symbolic/thermo_symbolic_f1.npz",
            "thermo_symbolic_overlay_objectives": ["eta_comb_gap"],
            "thermo_symbolic_overlay_constraints": ["mass_balance"],
            "thermo_symbolic_error": "",
        }
        return out


class _StubSimulation:
    def evaluate(self, candidate: dict[str, Any], *, context) -> dict[str, Any]:  # noqa: ARG002
        return {"objective": float(candidate.get("truth_objective", 0.0))}


def test_orchestrate_auto_truth_is_deterministic_and_budget_respecting(tmp_path: Path) -> None:
    outdir = tmp_path / "orch_auto"
    cfg = OrchestrationConfig(
        outdir=outdir,
        max_iterations=1,
        batch_size=4,
        total_sim_budget=2,
        truth_dispatch_mode="auto",
        truth_auto_top_k=3,
        truth_auto_min_uncertainty=0.15,
        truth_auto_min_feasibility=0.8,
        truth_auto_min_pred_quantile=0.5,
        use_provenance=False,
    )
    orch = Orchestrator(
        cem=_StubCEM(),
        surrogate=_StubSurrogate(),
        solver=_StubSolver(),
        simulation=_StubSimulation(),
        config=cfg,
    )
    result = orch.optimize(initial_params={})
    assert result.n_sim_calls == 2

    manifest = json.loads((outdir / "orchestrate_manifest.json").read_text(encoding="utf-8"))
    it0 = manifest["iterations"][0]
    selected = [row["candidate_id"] for row in it0["selected_candidates"]]
    assert selected == ["cand-1", "cand-3"]
    assert it0["selected_candidates"][0]["thermo_symbolic_mode"] == "strict"
    assert it0["selected_candidates"][0]["thermo_symbolic_used"] is True
    assert it0["selected_candidates"][0]["thermo_symbolic_overlay_objectives"] == ["eta_comb_gap"]
    assert it0["n_truth_evaluated"] == 2
    assert "life_damage_input_mode" in it0["truth_results"][0]
    assert "life_damage_status" in it0["truth_results"][0]
    assert "lifetime" in manifest


def test_orchestrate_context_includes_thermo_path_overrides(tmp_path: Path) -> None:
    cfg = OrchestrationConfig(
        outdir=tmp_path / "orch_context",
        max_iterations=1,
        batch_size=1,
        total_sim_budget=0,
        use_provenance=False,
        thermo_constants_path="data/thermo/literature_constants_v1.json",
        thermo_anchor_manifest_path="data/thermo/anchor_manifest_v1.json",
    )
    orch = Orchestrator(
        cem=_StubCEM(),
        surrogate=_StubSurrogate(),
        solver=_StubSolver(),
        simulation=_StubSimulation(),
        config=cfg,
    )
    ctx = orch._build_context()
    assert ctx.thermo_constants_path == "data/thermo/literature_constants_v1.json"
    assert ctx.thermo_anchor_manifest_path == "data/thermo/anchor_manifest_v1.json"
