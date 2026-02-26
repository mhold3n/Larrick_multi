"""Simulation adapter for orchestration truth evaluations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.adapters.calculix import CalculiXRunner
from larrak2.adapters.openfoam import OpenFoamRunner
from larrak2.core.encoding import N_TOTAL, decode_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext

LOGGER = logging.getLogger(__name__)


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


class PhysicsSimulationAdapter:
    """Runs truth evaluations and optionally dispatches OpenFOAM/CalculiX runners."""

    def __init__(
        self,
        *,
        openfoam_runner: OpenFoamRunner | None = None,
        calculix_runner: CalculiXRunner | None = None,
        work_dir: str | Path = "outputs/orchestration/sim_runs",
        run_openfoam: bool = False,
        run_calculix: bool = False,
    ) -> None:
        self.openfoam_runner = openfoam_runner
        self.calculix_runner = calculix_runner
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.run_openfoam = bool(run_openfoam and openfoam_runner is not None)
        self.run_calculix = bool(run_calculix and calculix_runner is not None)

    @staticmethod
    def _objective_from_eval(F: np.ndarray, G: np.ndarray) -> float:
        f = np.asarray(F, dtype=np.float64).reshape(-1)
        g = np.asarray(G, dtype=np.float64).reshape(-1)
        violation = float(np.maximum(g, 0.0).sum())
        return float(-np.sum(f) - 10.0 * violation)

    def _openfoam_params(self, candidate: dict[str, Any], context: EvalContext) -> dict[str, float]:
        x = np.asarray(candidate.get("x", []), dtype=np.float64).reshape(-1)
        decoded = decode_candidate(x)
        lam = float(decoded.thermo.lambda_af)
        p_back = 101_325.0
        p_manifold = max(p_back, p_back * (0.95 + 0.2 * np.clip(lam, 0.6, 1.6)))
        return {
            "rpm": float(context.rpm),
            "torque": float(context.torque),
            "lambda_af": float(lam),
            "bore_mm": float(candidate.get("bore_mm", 80.0)),
            "stroke_mm": float(candidate.get("stroke_mm", 90.0)),
            "intake_port_area_m2": float(candidate.get("intake_port_area_m2", 4.0e-4)),
            "exhaust_port_area_m2": float(candidate.get("exhaust_port_area_m2", 4.0e-4)),
            "p_manifold_Pa": float(p_manifold),
            "p_back_Pa": float(p_back),
            "overlap_deg": float(candidate.get("overlap_deg", 0.0)),
            "intake_open_deg": float(candidate.get("intake_open_deg", 0.0)),
            "intake_close_deg": float(candidate.get("intake_close_deg", 0.0)),
            "exhaust_open_deg": float(candidate.get("exhaust_open_deg", 0.0)),
            "exhaust_close_deg": float(candidate.get("exhaust_close_deg", 0.0)),
        }

    def _calculix_params(self, candidate: dict[str, Any], context: EvalContext) -> dict[str, float]:
        x = np.asarray(candidate.get("x", []), dtype=np.float64).reshape(-1)
        decoded = decode_candidate(x)
        return {
            "rpm": float(context.rpm),
            "torque": float(context.torque),
            "base_radius_mm": float(decoded.gear.base_radius),
            "face_width_mm": float(decoded.gear.face_width_mm),
            "module_mm": float(2.0 + abs(decoded.gear.pitch_coeffs[0])),
            "pressure_angle_deg": float(20.0 + 5.0 * decoded.gear.pitch_coeffs[1]),
            "helix_angle_deg": float(20.0 * decoded.gear.pitch_coeffs[2]),
            "profile_shift": float(decoded.gear.pitch_coeffs[3]),
        }

    def evaluate(
        self,
        candidate: dict[str, Any],
        *,
        context: EvalContext,
    ) -> dict[str, Any]:
        x = np.asarray(candidate.get("x", []), dtype=np.float64).reshape(-1)
        if x.size != N_TOTAL:
            return {"objective": float("-inf"), "error": f"expected x length {N_TOTAL}"}

        eval_result = evaluate_candidate(x, context)
        objective = self._objective_from_eval(eval_result.F, eval_result.G)

        payload: dict[str, Any] = {
            "objective": float(objective),
            "base_objective": float(objective),
            "F": np.asarray(eval_result.F, dtype=np.float64),
            "G": np.asarray(eval_result.G, dtype=np.float64),
            "diag": eval_result.diag,
        }

        candidate_id = str(candidate.get("id", candidate.get("global_index", "cand")))

        if self.run_openfoam and self.openfoam_runner is not None:
            run_dir = self.work_dir / f"openfoam_{candidate_id}"
            try:
                of_metrics = self.openfoam_runner.execute(
                    run_dir=run_dir,
                    params=self._openfoam_params(candidate, context),
                )
                payload["openfoam"] = _json_ready(of_metrics)
                if "scavenging_efficiency" in of_metrics:
                    payload["objective"] = float(payload["objective"]) + float(
                        of_metrics["scavenging_efficiency"]
                    )
            except Exception as exc:
                LOGGER.warning("OpenFOAM dispatch failed for %s: %s", candidate_id, exc)
                payload["openfoam"] = {"error": str(exc)}

        if self.run_calculix and self.calculix_runner is not None:
            run_dir = self.work_dir / f"calculix_{candidate_id}"
            try:
                ccx_metrics = self.calculix_runner.execute(
                    run_dir=run_dir,
                    job_name="gear",
                    params=self._calculix_params(candidate, context),
                )
                payload["calculix"] = _json_ready(ccx_metrics)
            except Exception as exc:
                LOGGER.warning("CalculiX dispatch failed for %s: %s", candidate_id, exc)
                payload["calculix"] = {"error": str(exc)}

        return _json_ready(payload)


__all__ = ["PhysicsSimulationAdapter"]

