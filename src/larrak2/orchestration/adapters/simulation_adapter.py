"""Simulation adapter for orchestration truth evaluations."""

from __future__ import annotations

import logging
import re
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


def _safe_path_token(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "cand"


def candidate_openfoam_params(
    candidate: dict[str, Any],
    context: EvalContext,
    *,
    eval_diag: dict[str, Any] | None = None,
) -> dict[str, float]:
    x = np.asarray(candidate.get("x", []), dtype=np.float64).reshape(-1)
    decoded = decode_candidate(x)
    lam = float(decoded.thermo.lambda_af)
    breathing = getattr(context, "breathing", None)
    thermo_diag = ((eval_diag or {}).get("thermo", {}) or {}) if isinstance(eval_diag, dict) else {}
    valve_timing = (
        dict(thermo_diag.get("valve_timing", {}))
        if isinstance(thermo_diag.get("valve_timing", {}), dict)
        else {}
    )
    p_back = float(getattr(breathing, "p_back_Pa", candidate.get("p_back_Pa", 101_325.0)))
    p_manifold = float(
        getattr(
            breathing,
            "p_manifold_Pa",
            max(p_back, p_back * (0.95 + 0.2 * np.clip(lam, 0.6, 1.6))),
        )
    )
    intake_open = float(
        valve_timing.get("intake_open_deg", decoded.thermo.intake_open_offset_from_bdc)
    )
    intake_close = float(
        valve_timing.get(
            "intake_close_deg",
            decoded.thermo.intake_open_offset_from_bdc + decoded.thermo.intake_duration_deg,
        )
    )
    exhaust_open = float(
        valve_timing.get("exhaust_open_deg", decoded.thermo.exhaust_open_offset_from_expansion_tdc)
    )
    exhaust_close = float(
        valve_timing.get(
            "exhaust_close_deg",
            decoded.thermo.exhaust_open_offset_from_expansion_tdc
            + decoded.thermo.exhaust_duration_deg,
        )
    )
    return {
        "rpm": float(context.rpm),
        "torque": float(context.torque),
        "lambda_af": float(lam),
        "bore_mm": float(getattr(breathing, "bore_mm", candidate.get("bore_mm", 80.0))),
        "stroke_mm": float(getattr(breathing, "stroke_mm", candidate.get("stroke_mm", 90.0))),
        "intake_port_area_m2": float(
            getattr(breathing, "intake_port_area_m2", candidate.get("intake_port_area_m2", 4.0e-4))
        ),
        "exhaust_port_area_m2": float(
            getattr(
                breathing,
                "exhaust_port_area_m2",
                candidate.get("exhaust_port_area_m2", 4.0e-4),
            )
        ),
        "p_manifold_Pa": float(p_manifold),
        "p_back_Pa": float(p_back),
        "overlap_deg": float(valve_timing.get("overlap_deg", candidate.get("overlap_deg", 0.0))),
        "intake_open_deg": intake_open,
        "intake_close_deg": intake_close,
        "exhaust_open_deg": exhaust_open,
        "exhaust_close_deg": exhaust_close,
        "endTime": float(candidate.get("openfoam_endTime", 3.0e-4)),
        "deltaT": float(candidate.get("openfoam_deltaT", 1.0e-4)),
        "writeInterval": int(candidate.get("openfoam_writeInterval", 1)),
        "metricWriteInterval": int(candidate.get("openfoam_metricWriteInterval", 1)),
    }


def candidate_openfoam_geometry_args(
    candidate: dict[str, Any],
    context: EvalContext,
) -> dict[str, float]:
    breathing = getattr(context, "breathing", None)
    return {
        "bore_mm": float(getattr(breathing, "bore_mm", candidate.get("bore_mm", 80.0))),
        "stroke_mm": float(getattr(breathing, "stroke_mm", candidate.get("stroke_mm", 90.0))),
        "intake_port_area_m2": float(
            getattr(breathing, "intake_port_area_m2", candidate.get("intake_port_area_m2", 4.0e-4))
        ),
        "exhaust_port_area_m2": float(
            getattr(
                breathing,
                "exhaust_port_area_m2",
                candidate.get("exhaust_port_area_m2", 4.0e-4),
            )
        ),
    }


def candidate_calculix_params(candidate: dict[str, Any], context: EvalContext) -> dict[str, float]:
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

    def _openfoam_params(
        self,
        candidate: dict[str, Any],
        context: EvalContext,
        *,
        eval_diag: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        return candidate_openfoam_params(candidate, context, eval_diag=eval_diag)

    def _openfoam_geometry_args(
        self,
        candidate: dict[str, Any],
        context: EvalContext,
    ) -> dict[str, float]:
        return candidate_openfoam_geometry_args(candidate, context)

    def _calculix_params(self, candidate: dict[str, Any], context: EvalContext) -> dict[str, float]:
        return candidate_calculix_params(candidate, context)

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
            "operating_point": {
                "rpm": float(context.rpm),
                "torque": float(context.torque),
                "fidelity": int(context.fidelity),
            },
            "dispatch": {
                "run_openfoam": bool(self.run_openfoam),
                "run_calculix": bool(self.run_calculix),
            },
        }

        candidate_id = str(candidate.get("id", candidate.get("global_index", "cand")))
        run_token = _safe_path_token(candidate_id)

        if self.run_openfoam and self.openfoam_runner is not None:
            run_dir = self.work_dir / f"openfoam_{run_token}"
            try:
                openfoam_params = self._openfoam_params(
                    candidate, context, eval_diag=eval_result.diag
                )
                try:
                    of_metrics = self.openfoam_runner.execute(
                        run_dir=run_dir,
                        params=openfoam_params,
                        geometry_args=self._openfoam_geometry_args(candidate, context),
                    )
                except TypeError:
                    of_metrics = self.openfoam_runner.execute(
                        run_dir=run_dir,
                        params=openfoam_params,
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
            run_dir = self.work_dir / f"calculix_{run_token}"
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
