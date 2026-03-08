"""Deterministic reduced-order search and diagnosis for principles regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from larrak2.core.types import EvalContext
from larrak2.pipelines.principles_core import (
    PRINCIPLES_OBJECTIVE_NAMES,
    REDUCED_VARIABLE_NAMES,
    objective_scales_from_profile,
    reduced_bounds,
    reduced_release_stages,
    reduced_seed_states,
    weight_vectors_from_profile,
)
from larrak2.pipelines.principles_evaluator import (
    PrinciplesAlignmentResult,
    evaluate_principles_alignment,
    evaluate_principles_proxy,
)


@dataclass(frozen=True)
class PrinciplesDiagnosis:
    classification: str
    metrics: dict[str, Any]


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


class _CachedPrinciplesScorer:
    def __init__(
        self,
        *,
        profile_payload: dict[str, Any],
        base_ctx: EvalContext,
        alignment_mode: str,
        alignment_fidelity: int,
        alignment_phase: str,
    ) -> None:
        self.profile_payload = profile_payload
        self.base_ctx = base_ctx
        self.alignment_mode = str(alignment_mode)
        self.alignment_fidelity = int(alignment_fidelity)
        self.alignment_phase = str(alignment_phase)
        self.scales = objective_scales_from_profile(profile_payload)
        self.blend_weights = dict(profile_payload.get("blend_weights", {}) or {})
        self.proxy_weight = float(self.blend_weights.get("proxy", 0.7))
        self.canonical_weight = float(self.blend_weights.get("canonical", 0.3))
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _key(self, z: np.ndarray, rpm: float, torque: float) -> tuple[Any, ...]:
        return (
            round(float(rpm), 8),
            round(float(torque), 8),
            tuple(float(np.round(v, 8)) for v in np.asarray(z, dtype=np.float64).reshape(-1)),
        )

    def evaluate(self, z: np.ndarray, *, rpm: float, torque: float) -> dict[str, Any]:
        key = self._key(z, rpm, torque)
        if key in self._cache:
            return self._cache[key]

        proxy = evaluate_principles_proxy(
            np.asarray(z, dtype=np.float64),
            profile_payload=self.profile_payload,
            base_ctx=self.base_ctx,
            rpm=float(rpm),
            torque=float(torque),
        )
        if self.alignment_mode == "proxy_only":
            alignment = PrinciplesAlignmentResult(
                F=np.asarray(proxy.F, dtype=np.float64),
                G=np.asarray(proxy.G, dtype=np.float64),
                diag={},
                objective_names=tuple(proxy.objective_names),
                constraint_names=tuple(proxy.constraint_names),
            )
        else:
            alignment = evaluate_principles_alignment(
                np.asarray(proxy.x_full, dtype=np.float64),
                base_ctx=self.base_ctx,
                rpm=float(rpm),
                torque=float(torque),
                fidelity=self.alignment_fidelity,
                constraint_phase=self.alignment_phase,
            )

        proxy_F = np.asarray(proxy.F, dtype=np.float64)
        align_F = np.asarray(alignment.F, dtype=np.float64)
        if self.alignment_mode == "canonical_only":
            F_blend = align_F.copy()
        elif self.alignment_mode == "proxy_only":
            F_blend = proxy_F.copy()
        else:
            F_blend = self.proxy_weight * (proxy_F / self.scales) + self.canonical_weight * (
                align_F / self.scales
            )

        proxy_G = np.asarray(proxy.G, dtype=np.float64).reshape(-1)
        align_G = np.asarray(alignment.G, dtype=np.float64).reshape(-1)
        max_len = max(proxy_G.size, align_G.size)
        if proxy_G.size < max_len:
            proxy_G = np.pad(proxy_G, (0, max_len - proxy_G.size), constant_values=1.0e3)
        if align_G.size < max_len:
            align_G = np.pad(align_G, (0, max_len - align_G.size), constant_values=1.0e3)
        G_combined = np.maximum(proxy_G, align_G)
        hard_penalty = float(np.sum(np.square(np.maximum(G_combined, 0.0))))

        objective_delta_abs = np.abs(proxy_F - align_F)
        objective_delta_rel = objective_delta_abs / np.maximum(np.abs(align_F), self.scales)
        constraint_delta_abs = np.abs(proxy_G - align_G)
        top_obj = np.argsort(-objective_delta_abs)[:3]
        top_con = np.argsort(-constraint_delta_abs)[:3]
        bundle = {
            "proxy": proxy,
            "alignment": alignment,
            "F_blend": np.asarray(F_blend, dtype=np.float64),
            "G_combined": np.asarray(G_combined, dtype=np.float64),
            "hard_penalty": hard_penalty,
            "proxy_feasible": bool(np.all(proxy_G <= 0.0)),
            "alignment_feasible": bool(np.all(align_G <= 0.0)),
            "combined_feasible": bool(np.all(G_combined <= 0.0)),
            "objective_delta_abs": objective_delta_abs,
            "objective_delta_rel": objective_delta_rel,
            "constraint_delta_abs": constraint_delta_abs,
            "dominant_mismatch_objectives": [
                PRINCIPLES_OBJECTIVE_NAMES[int(i)] for i in top_obj.tolist()
            ],
            "dominant_mismatch_constraints": [
                str((proxy.constraint_names + alignment.constraint_names + ("",) * max_len)[int(i)])
                for i in top_con.tolist()
            ],
        }
        self._cache[key] = bundle
        return bundle


def _weighted_score(
    bundle: dict[str, Any], weights: np.ndarray, z: np.ndarray, z_ref: np.ndarray, span: np.ndarray
) -> float:
    dist_penalty = float(np.mean(np.square((np.asarray(z, dtype=np.float64) - z_ref) / span)))
    return float(
        np.dot(weights, np.asarray(bundle["F_blend"], dtype=np.float64))
        + 100.0 * float(bundle["hard_penalty"])
        + 1.0e-3 * dist_penalty
    )


def _minimize_stage(
    *,
    scorer: _CachedPrinciplesScorer,
    z_current: np.ndarray,
    z_reference: np.ndarray,
    active_indices: list[int],
    rpm: float,
    torque: float,
    weights: np.ndarray,
    max_iter: int,
    stage_name: str,
    xl: np.ndarray,
    xu: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not active_indices:
        bundle = scorer.evaluate(z_current, rpm=rpm, torque=torque)
        return z_current, {
            "stage": stage_name,
            "success": True,
            "skipped": True,
            "score_after": float(
                _weighted_score(bundle, weights, z_current, z_reference, np.maximum(xu - xl, 1e-9))
            ),
        }

    z0 = np.asarray([z_current[i] for i in active_indices], dtype=np.float64)
    lb = np.asarray([xl[i] for i in active_indices], dtype=np.float64)
    ub = np.asarray([xu[i] for i in active_indices], dtype=np.float64)
    span = np.maximum(xu - xl, 1e-9)
    before_bundle = scorer.evaluate(z_current, rpm=rpm, torque=torque)
    before_score = _weighted_score(before_bundle, weights, z_current, z_reference, span)

    def _objective(z_active: np.ndarray) -> float:
        z_trial = np.asarray(z_current, dtype=np.float64).copy()
        z_trial[active_indices] = np.clip(np.asarray(z_active, dtype=np.float64), lb, ub)
        bundle = scorer.evaluate(z_trial, rpm=rpm, torque=torque)
        return _weighted_score(bundle, weights, z_trial, z_reference, span)

    try:
        from scipy.optimize import minimize

        result = minimize(
            _objective,
            z0,
            method="SLSQP",
            bounds=list(zip(lb, ub)),
            options={"maxiter": int(max(2, max_iter)), "ftol": 1e-6, "disp": False},
        )
        z_new = np.asarray(z_current, dtype=np.float64).copy()
        z_new[active_indices] = np.clip(np.asarray(result.x, dtype=np.float64), lb, ub)
        after_bundle = scorer.evaluate(z_new, rpm=rpm, torque=torque)
        after_score = _weighted_score(after_bundle, weights, z_new, z_reference, span)
        return z_new, {
            "stage": stage_name,
            "success": bool(result.success),
            "message": str(result.message),
            "nit": int(getattr(result, "nit", 0)),
            "score_before": float(before_score),
            "score_after": float(after_score),
            "hard_penalty_before": float(before_bundle["hard_penalty"]),
            "hard_penalty_after": float(after_bundle["hard_penalty"]),
            "proxy_feasible_after": bool(after_bundle["proxy_feasible"]),
            "alignment_feasible_after": bool(after_bundle["alignment_feasible"]),
            "combined_feasible_after": bool(after_bundle["combined_feasible"]),
        }
    except Exception as exc:
        return z_current, {
            "stage": stage_name,
            "success": False,
            "error": f"stage_solver_failed: {type(exc).__name__}: {exc}",
            "score_before": float(before_score),
            "hard_penalty_before": float(before_bundle["hard_penalty"]),
        }


def search_principles_region(
    *,
    profile_payload: dict[str, Any],
    operating_points: list[dict[str, Any]],
    base_ctx: EvalContext,
    alignment_mode: str,
    alignment_fidelity: int,
    alignment_phase: str,
    stage_max_iter: int,
    region_min_size: int,
) -> dict[str, Any]:
    xl, xu = reduced_bounds()
    seed_states = reduced_seed_states(profile_payload)
    stages = reduced_release_stages(profile_payload)
    weights = weight_vectors_from_profile(profile_payload)
    scorer = _CachedPrinciplesScorer(
        profile_payload=profile_payload,
        base_ctx=base_ctx,
        alignment_mode=alignment_mode,
        alignment_fidelity=alignment_fidelity,
        alignment_phase=alignment_phase,
    )

    records: list[dict[str, Any]] = []
    op_records: list[dict[str, Any]] = []
    for op_idx, op in enumerate(operating_points):
        rpm = float(op["rpm"])
        torque = float(op["torque"])
        op_records.append({"rpm": rpm, "torque": torque, "source": str(op.get("source", ""))})
        for weight_rec in weights:
            w = np.asarray(weight_rec["weights"], dtype=np.float64)
            for seed_name, seed_state in seed_states.items():
                z_curr = np.clip(np.asarray(seed_state, dtype=np.float64), xl, xu)
                stage_records: list[dict[str, Any]] = []
                for stage in stages:
                    active_indices = [
                        int(REDUCED_VARIABLE_NAMES.index(name)) for name in stage["variable_names"]
                    ]
                    z_curr, stage_diag = _minimize_stage(
                        scorer=scorer,
                        z_current=z_curr,
                        z_reference=np.asarray(seed_state, dtype=np.float64),
                        active_indices=active_indices,
                        rpm=rpm,
                        torque=torque,
                        weights=w,
                        max_iter=int(stage_max_iter),
                        stage_name=str(stage["name"]),
                        xl=xl,
                        xu=xu,
                    )
                    stage_records.append(stage_diag)
                bundle = scorer.evaluate(z_curr, rpm=rpm, torque=torque)
                score = _weighted_score(
                    bundle,
                    w,
                    z_curr,
                    np.asarray(seed_state, dtype=np.float64),
                    np.maximum(xu - xl, 1e-9),
                )
                records.append(
                    {
                        "operating_point_index": int(op_idx),
                        "operating_point": dict(op_records[-1]),
                        "weight_name": str(weight_rec["name"]),
                        "weight_vector": np.asarray(w, dtype=np.float64),
                        "seed_state_name": str(seed_name),
                        "z_reduced": np.asarray(z_curr, dtype=np.float64),
                        "x_full": np.asarray(bundle["proxy"].x_full, dtype=np.float64),
                        "proxy": bundle["proxy"],
                        "alignment": bundle["alignment"],
                        "F_blend": np.asarray(bundle["F_blend"], dtype=np.float64),
                        "G_combined": np.asarray(bundle["G_combined"], dtype=np.float64),
                        "hard_penalty": float(bundle["hard_penalty"]),
                        "proxy_feasible": bool(bundle["proxy_feasible"]),
                        "alignment_feasible": bool(bundle["alignment_feasible"]),
                        "combined_feasible": bool(bundle["combined_feasible"]),
                        "objective_delta_abs": np.asarray(
                            bundle["objective_delta_abs"], dtype=np.float64
                        ),
                        "objective_delta_rel": np.asarray(
                            bundle["objective_delta_rel"], dtype=np.float64
                        ),
                        "constraint_delta_abs": np.asarray(
                            bundle["constraint_delta_abs"], dtype=np.float64
                        ),
                        "dominant_mismatch_objectives": list(
                            bundle["dominant_mismatch_objectives"]
                        ),
                        "dominant_mismatch_constraints": list(
                            bundle["dominant_mismatch_constraints"]
                        ),
                        "stage_records": stage_records,
                        "weighted_score": float(score),
                    }
                )

    if not records:
        raise RuntimeError("Principles search produced no candidate records")

    def _dedupe_indices(indices: list[int]) -> list[int]:
        seen: set[tuple[float, ...]] = set()
        out: list[int] = []
        for idx in indices:
            key = tuple(
                float(np.round(v, 6)) for v in np.asarray(records[idx]["x_full"], dtype=np.float64)
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(int(idx))
        return out

    combined_feasible = [i for i, rec in enumerate(records) if bool(rec["combined_feasible"])]
    combined_feasible.sort(key=lambda i: (float(records[i]["weighted_score"]), int(i)))
    region_indices = _dedupe_indices(combined_feasible)
    if len(region_indices) < int(region_min_size):
        borderline = [i for i, rec in enumerate(records) if i not in region_indices]
        borderline.sort(
            key=lambda i: (
                float(records[i]["hard_penalty"]),
                float(records[i]["weighted_score"]),
                int(i),
            )
        )
        region_indices = _dedupe_indices(
            region_indices + borderline[: max(0, int(region_min_size) - len(region_indices))]
        )
    region_indices = region_indices[: max(1, int(region_min_size))] if region_indices else []

    point_rpms = {round(float(op["rpm"]), 6) for op in op_records}
    point_torques = {round(float(op["torque"]), 6) for op in op_records}
    region_rpms = {round(float(records[i]["operating_point"]["rpm"]), 6) for i in region_indices}
    region_torques = {
        round(float(records[i]["operating_point"]["torque"]), 6) for i in region_indices
    }
    coverage_rpm_fraction = float(len(region_rpms & point_rpms) / max(1, len(point_rpms)))
    coverage_torque_fraction = float(
        len(region_torques & point_torques) / max(1, len(point_torques))
    )

    proxy_feasible_indices = [i for i, rec in enumerate(records) if bool(rec["proxy_feasible"])]
    alignment_feasible_indices = [
        i for i, rec in enumerate(records) if bool(rec["alignment_feasible"])
    ]
    expansion_mapping_failures = [
        i for i in proxy_feasible_indices if not bool(records[i]["alignment_feasible"])
    ]
    mapping_failure_fraction = float(
        len(expansion_mapping_failures) / max(1, len(proxy_feasible_indices))
    )

    finite_objective_deltas = [
        np.asarray(rec["objective_delta_rel"], dtype=np.float64)
        for rec in records
        if not rec["proxy"].error_signature and not rec["alignment"].error_signature
    ]
    if finite_objective_deltas:
        delta_mat = np.vstack(finite_objective_deltas)
        objective_delta_median = np.median(delta_mat, axis=0)
    else:
        objective_delta_median = np.full(len(PRINCIPLES_OBJECTIVE_NAMES), np.inf, dtype=np.float64)

    shared_fail_rows = [
        rec
        for rec in records
        if (not bool(rec["proxy_feasible"])) and (not bool(rec["alignment_feasible"]))
    ]
    shared_reason_matches = 0
    for rec in shared_fail_rows:
        proxy_constraints = [
            str(row.get("name", ""))
            for row in (
                rec["proxy"].diag.get("constraints", [])
                if isinstance(rec["proxy"].diag, dict)
                else []
            )
            if isinstance(row, dict)
            and float(row.get("scaled_raw", row.get("scaled", 0.0))) > 0.0
            and str(row.get("kind", "hard")) == "hard"
        ]
        align_constraints = [
            str(row.get("name", ""))
            for row in (
                rec["alignment"].diag.get("constraints", [])
                if isinstance(rec["alignment"].diag, dict)
                else []
            )
            if isinstance(row, dict)
            and float(row.get("scaled_raw", row.get("scaled", 0.0))) > 0.0
            and str(row.get("kind", "hard")) == "hard"
        ]
        if proxy_constraints and align_constraints and proxy_constraints[0] == align_constraints[0]:
            shared_reason_matches += 1
    shared_failure_fraction = float(shared_reason_matches / max(1, len(shared_fail_rows)))

    thresholds = dict(profile_payload.get("gate_thresholds", {}) or {})
    misconfig_errors = [
        err
        for rec in records
        for err in (str(rec["proxy"].error_signature), str(rec["alignment"].error_signature))
        if err
        and any(
            token in err.lower()
            for token in (
                "not found",
                "dataset",
                "anchor manifest",
                "quality_report",
                "filenotfounderror",
                "valueerror",
            )
        )
    ]

    if misconfig_errors:
        classification = "misconfiguration_or_data_gap"
    elif len(proxy_feasible_indices) < int(thresholds.get("proxy_feasible_min", 3)):
        classification = "proxy_region_gap"
    elif mapping_failure_fraction >= float(
        thresholds.get("expansion_mapping_failure_fraction", 0.5)
    ):
        classification = "expansion_mapping_gap"
    elif int(
        np.sum(
            objective_delta_median
            > float(thresholds.get("objective_plane_residual_median_max", 0.2))
        )
    ) >= int(thresholds.get("objective_plane_mismatch_objectives_min", 3)):
        classification = "objective_plane_gap"
    elif shared_failure_fraction >= float(thresholds.get("shared_failure_fraction", 0.6)):
        classification = "shared_manifold_gap"
    elif len(region_indices) >= int(region_min_size):
        classification = "region_ready"
    else:
        classification = "proxy_region_gap"

    diagnosis = PrinciplesDiagnosis(
        classification=classification,
        metrics={
            "proxy_feasible_count": int(len(proxy_feasible_indices)),
            "alignment_feasible_count": int(len(alignment_feasible_indices)),
            "mapping_failure_fraction": float(mapping_failure_fraction),
            "objective_delta_median": objective_delta_median.tolist(),
            "shared_failure_fraction": float(shared_failure_fraction),
            "misconfiguration_errors": misconfig_errors,
        },
    )

    summary = {
        "n_operating_points": int(len(op_records)),
        "n_weight_vectors": int(len(weights)),
        "n_stage_solves": int(len(records) * len(stages)),
        "n_proxy_feasible": int(len(proxy_feasible_indices)),
        "n_alignment_feasible": int(len(alignment_feasible_indices)),
        "n_region_candidates": int(len(region_indices)),
        "envelope_coverage_rpm_fraction": float(coverage_rpm_fraction),
        "envelope_coverage_torque_fraction": float(coverage_torque_fraction),
        "region_min_size_required": int(region_min_size),
    }

    proxy_vs_canonical = {
        "objective_delta_median": objective_delta_median.tolist(),
        "objective_delta_max": (
            np.max(np.vstack(finite_objective_deltas), axis=0).tolist()
            if finite_objective_deltas
            else [float("inf")] * len(PRINCIPLES_OBJECTIVE_NAMES)
        ),
        "dominant_failure_constraints": sorted(
            {
                str(reason)
                for rec in records
                for reason in list(rec["dominant_mismatch_constraints"])
                if str(reason).strip()
            }
        ),
        "representative_candidate_agreement": [
            {
                "candidate_index": int(idx),
                "operating_point": dict(records[idx]["operating_point"]),
                "weight_name": str(records[idx]["weight_name"]),
                "proxy_feasible": bool(records[idx]["proxy_feasible"]),
                "alignment_feasible": bool(records[idx]["alignment_feasible"]),
                "combined_feasible": bool(records[idx]["combined_feasible"]),
                "objective_delta_abs": np.asarray(
                    records[idx]["objective_delta_abs"], dtype=np.float64
                ).tolist(),
                "objective_delta_rel": np.asarray(
                    records[idx]["objective_delta_rel"], dtype=np.float64
                ).tolist(),
            }
            for idx in region_indices
        ],
    }

    return {
        "records": records,
        "region_indices": region_indices,
        "region_summary": summary,
        "proxy_vs_canonical": proxy_vs_canonical,
        "diagnosis": diagnosis,
    }


__all__ = ["PrinciplesDiagnosis", "search_principles_region"]
