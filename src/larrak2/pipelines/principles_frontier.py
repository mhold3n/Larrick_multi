"""Principles-first frontier synthesis for explore-exploit workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.core.archive_io import save_archive
from larrak2.core.encoding import N_TOTAL, bounds, group_indices, mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext
from larrak2.optimization.candidate_store import CandidateStore


@dataclass(frozen=True)
class OperatingPoint:
    rpm: float
    torque: float
    source: str


@dataclass(frozen=True)
class PrinciplesFrontierResult:
    store: CandidateStore
    pareto_source: Path
    profile_name: str
    profile_path: Path
    profile_payload: dict[str, Any]
    gate: dict[str, Any]
    artifacts: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_profile_path(profile_name: str, repo_root: Path) -> Path:
    if str(profile_name).strip().lower() == "iso_litvin_v1":
        return repo_root / "data" / "optimization" / "principles_frontier_profile_v1.json"
    raw = Path(str(profile_name))
    if raw.is_absolute():
        return raw
    return (repo_root / raw).resolve()


def load_principles_profile(
    profile_name: str, *, repo_root: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    root = _repo_root() if repo_root is None else Path(repo_root)
    path = _resolve_profile_path(profile_name, root)
    if not path.exists():
        raise FileNotFoundError(f"Principles profile not found: {path}")
    payload = _read_json(path)

    required = ("profile_id", "anchor_manifest", "envelope", "gate_thresholds", "source_references")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Principles profile missing required keys: {missing}")

    envelope = payload.get("envelope", {})
    for key in ("rpm_min", "rpm_max", "torque_min", "torque_max"):
        if key not in envelope:
            raise ValueError(f"Principles profile envelope missing '{key}'")
        _ = float(envelope[key])

    thresholds = payload.get("gate_thresholds", {})
    for key in ("coverage_rpm_fraction_min", "coverage_torque_fraction_min"):
        if key not in thresholds:
            raise ValueError(f"Principles profile gate_thresholds missing '{key}'")
        _ = float(thresholds[key])

    refs = payload.get("source_references", [])
    if not isinstance(refs, list) or not refs:
        raise ValueError("Principles profile requires non-empty 'source_references'")
    return path, payload


def _load_anchor_manifest(
    profile_payload: dict[str, Any], repo_root: Path
) -> tuple[Path, dict[str, Any]]:
    raw = Path(str(profile_payload.get("anchor_manifest", "")))
    path = raw if raw.is_absolute() else (repo_root / raw)
    if not path.exists():
        raise FileNotFoundError(f"Anchor manifest not found: {path}")
    payload = _read_json(path)
    anchors = payload.get("anchors", [])
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("Anchor manifest must contain non-empty 'anchors'")
    for i, rec in enumerate(anchors):
        rpm = float(rec.get("rpm", 0.0))
        torque = float(rec.get("torque", -1.0))
        if rpm <= 0.0 or torque < 0.0:
            raise ValueError(f"Invalid anchor[{i}] operating point: rpm={rpm}, torque={torque}")
    return path, payload


def _build_operating_points(
    *,
    profile_payload: dict[str, Any],
    anchor_manifest: dict[str, Any],
) -> list[OperatingPoint]:
    env = profile_payload.get("envelope", {})
    rpm_min = float(env.get("rpm_min", 1000.0))
    rpm_max = float(env.get("rpm_max", rpm_min))
    torque_min = float(env.get("torque_min", 40.0))
    torque_max = float(env.get("torque_max", torque_min))
    rpm_mid = 0.5 * (rpm_min + rpm_max)
    torque_mid = 0.5 * (torque_min + torque_max)

    points: list[OperatingPoint] = []
    for rec in anchor_manifest.get("anchors", []):
        points.append(
            OperatingPoint(
                rpm=float(rec.get("rpm", rpm_mid)),
                torque=float(rec.get("torque", torque_mid)),
                source=str(rec.get("label", "anchor")),
            )
        )
    points.extend(
        [
            OperatingPoint(rpm=rpm_min, torque=torque_min, source="envelope_corner"),
            OperatingPoint(rpm=rpm_min, torque=torque_max, source="envelope_corner"),
            OperatingPoint(rpm=rpm_max, torque=torque_min, source="envelope_corner"),
            OperatingPoint(rpm=rpm_max, torque=torque_max, source="envelope_corner"),
            OperatingPoint(rpm=rpm_mid, torque=torque_mid, source="envelope_center"),
        ]
    )

    dedup: dict[tuple[float, float], OperatingPoint] = {}
    for point in points:
        key = (round(float(point.rpm), 9), round(float(point.torque), 9))
        if key not in dedup:
            dedup[key] = point
    return list(dedup.values())


def _hard_violation_details(diag: dict[str, Any], G: np.ndarray) -> tuple[float, list[str]]:
    constraints = diag.get("constraints", []) if isinstance(diag, dict) else []
    violations: list[float] = []
    reasons: list[str] = []
    if isinstance(constraints, list) and constraints:
        for rec in constraints:
            if not isinstance(rec, dict):
                continue
            if str(rec.get("kind", "hard")) != "hard":
                continue
            scaled = float(rec.get("scaled_raw", rec.get("scaled", 0.0)))
            if scaled > 0.0:
                violations.append(float(scaled))
                reasons.append(str(rec.get("name", "unknown_hard_constraint")))
    else:
        g = np.asarray(G, dtype=np.float64).reshape(-1)
        for i, val in enumerate(g):
            if float(val) > 0.0:
                violations.append(float(val))
                reasons.append(f"hard_constraint_{i}")

    hard_sum = float(np.sum(np.square(violations))) if violations else 0.0
    return hard_sum, reasons


def _evaluate_hard_violation_score(
    x: np.ndarray,
    *,
    ctx: EvalContext,
    seed_x: np.ndarray,
    alpha: float,
    xl: np.ndarray,
    xu: np.ndarray,
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray]:
    try:
        res = evaluate_candidate(x, ctx)
        hard_sum, reasons = _hard_violation_details(res.diag, res.G)
        F = np.asarray(res.F, dtype=np.float64)
        G = np.asarray(res.G, dtype=np.float64)
        eval_error = ""
    except Exception as exc:
        hard_sum = 1.0e12
        reasons = [f"evaluate_candidate_error:{type(exc).__name__}"]
        F = np.full(6, 1.0e6, dtype=np.float64)
        G = np.asarray([1.0], dtype=np.float64)
        eval_error = str(exc)
    span = np.maximum(xu - xl, 1e-9)
    dist_norm = float(np.mean(np.square((x - seed_x) / span)))
    score = float(hard_sum + alpha * dist_norm)
    return (
        score,
        {
            "hard_violation_score": hard_sum,
            "hard_reasons": reasons,
            "dist_norm": dist_norm,
            "eval_error": eval_error,
        },
        F,
        G,
    )


def _minimize_stage(
    *,
    x_current: np.ndarray,
    seed_x: np.ndarray,
    ctx: EvalContext,
    active_indices: list[int],
    stage_name: str,
    max_iter: int,
    alpha: float,
    xl: np.ndarray,
    xu: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not active_indices:
        return x_current, {"stage": stage_name, "success": True, "skipped": True}

    z0 = np.asarray([x_current[i] for i in active_indices], dtype=np.float64)
    stage_lb = np.asarray([xl[i] for i in active_indices], dtype=np.float64)
    stage_ub = np.asarray([xu[i] for i in active_indices], dtype=np.float64)

    def _score(z: np.ndarray) -> float:
        x = np.array(x_current, dtype=np.float64, copy=True)
        x[active_indices] = np.clip(np.asarray(z, dtype=np.float64), stage_lb, stage_ub)
        score, _, _, _ = _evaluate_hard_violation_score(
            x,
            ctx=ctx,
            seed_x=seed_x,
            alpha=alpha,
            xl=xl,
            xu=xu,
        )
        return float(score)

    before_score, before_diag, _, _ = _evaluate_hard_violation_score(
        x_current,
        ctx=ctx,
        seed_x=seed_x,
        alpha=alpha,
        xl=xl,
        xu=xu,
    )

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        return x_current, {
            "stage": stage_name,
            "success": False,
            "error": f"scipy_unavailable: {exc}",
            "score_before": before_score,
            "hard_reasons_before": before_diag["hard_reasons"],
        }

    try:
        result = minimize(
            _score,
            z0,
            method="SLSQP",
            bounds=list(zip(stage_lb, stage_ub)),
            options={"maxiter": int(max(2, max_iter)), "ftol": 1e-6, "disp": False},
        )
        z_opt = np.clip(np.asarray(result.x, dtype=np.float64), stage_lb, stage_ub)
        x_new = np.array(x_current, dtype=np.float64, copy=True)
        x_new[active_indices] = z_opt
        after_score, after_diag, _, _ = _evaluate_hard_violation_score(
            x_new,
            ctx=ctx,
            seed_x=seed_x,
            alpha=alpha,
            xl=xl,
            xu=xu,
        )
        return x_new, {
            "stage": stage_name,
            "success": bool(result.success),
            "message": str(result.message),
            "nit": int(getattr(result, "nit", 0)),
            "score_before": before_score,
            "score_after": after_score,
            "hard_reasons_before": before_diag["hard_reasons"],
            "hard_reasons_after": after_diag["hard_reasons"],
        }
    except Exception as exc:  # pragma: no cover
        return x_current, {
            "stage": stage_name,
            "success": False,
            "error": f"stage_minimize_failed: {exc}",
            "score_before": before_score,
            "hard_reasons_before": before_diag["hard_reasons"],
        }


def _restore_candidate(
    *,
    x_seed: np.ndarray,
    ctx: EvalContext,
    root_max_iter: int,
    xl: np.ndarray,
    xu: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    groups = group_indices()
    stages = [
        ("thermo", [int(i) for i in groups.get("thermo", [])]),
        ("gear", [int(i) for i in groups.get("gear", [])]),
        ("realworld", [int(i) for i in groups.get("realworld", [])]),
    ]
    stage_iter = max(4, int(max(4, root_max_iter) // 3))
    alpha = 0.02

    x_curr = np.asarray(x_seed, dtype=np.float64).copy()
    stage_diags: list[dict[str, Any]] = []
    for stage_name, active in stages:
        x_curr, stage_diag = _minimize_stage(
            x_current=x_curr,
            seed_x=np.asarray(x_seed, dtype=np.float64),
            ctx=ctx,
            active_indices=active,
            stage_name=stage_name,
            max_iter=stage_iter,
            alpha=alpha,
            xl=xl,
            xu=xu,
        )
        stage_diags.append(stage_diag)

    score_diag: dict[str, Any] = {}
    try:
        res = evaluate_candidate(x_curr, ctx)
        hard_sum, hard_reasons = _hard_violation_details(res.diag, res.G)
        F = np.asarray(res.F, dtype=np.float64)
        G = np.asarray(res.G, dtype=np.float64)
    except Exception as exc:
        hard_sum = 1.0e12
        hard_reasons = [f"evaluate_candidate_error:{type(exc).__name__}"]
        score_diag["eval_error"] = str(exc)
        F = np.full(6, 1.0e6, dtype=np.float64)
        G = np.asarray([1.0], dtype=np.float64)
    payload = {
        "root_success": bool(hard_sum <= 1e-12),
        "hard_violation_score": hard_sum,
        "hard_reasons": list(hard_reasons),
        "stages": stage_diags,
        **score_diag,
    }
    return x_curr, payload, F, G


def _halton_samples(n: int, d: int, seed: int) -> np.ndarray:
    n_eff = int(max(1, n))
    d_eff = int(max(1, d))
    try:
        from scipy.stats.qmc import Halton

        sampler = Halton(d=d_eff, scramble=True, seed=int(seed))
        return np.asarray(sampler.random(n_eff), dtype=np.float64)
    except Exception:
        rng = np.random.default_rng(int(seed))
        return rng.random((n_eff, d_eff))


def _generate_seed_vectors(
    *, seed_count: int, seed: int, xl: np.ndarray, xu: np.ndarray
) -> np.ndarray:
    n = int(max(1, seed_count))
    mid = np.asarray(mid_bounds_candidate(), dtype=np.float64)
    samples = _halton_samples(n=n, d=N_TOTAL, seed=seed)
    centered = 2.0 * samples - 1.0

    scales = np.ones(N_TOTAL, dtype=np.float64) * 0.2
    groups = group_indices()
    for idx in groups.get("thermo", []):
        scales[int(idx)] = 0.30
    for idx in groups.get("gear", []):
        scales[int(idx)] = 0.35
    for idx in groups.get("realworld", []):
        scales[int(idx)] = 0.50

    span = np.maximum(np.asarray(xu, dtype=np.float64) - np.asarray(xl, dtype=np.float64), 1e-9)
    seeds = np.clip(mid + centered * span * scales, xl, xu)
    seeds[0] = mid
    return seeds


def _non_dominated_indices(F: np.ndarray) -> np.ndarray:
    F_arr = np.asarray(F, dtype=np.float64)
    n = int(F_arr.shape[0])
    if n <= 1:
        return np.arange(n, dtype=int)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            fi = F_arr[i]
            fj = F_arr[j]
            if np.all(fj <= fi) and np.any(fj < fi):
                dominated[i] = True
                break
    return np.where(~dominated)[0].astype(int)


def _normalize_objectives(F: np.ndarray) -> np.ndarray:
    arr = np.asarray(F, dtype=np.float64)
    if arr.size == 0:
        return arr
    mn = np.min(arr, axis=0)
    mx = np.max(arr, axis=0)
    span = np.maximum(mx - mn, 1e-9)
    return (arr - mn) / span


def _coverage_fraction(seen: set[float], total: set[float]) -> float:
    if not total:
        return 0.0
    return float(len(seen.intersection(total)) / len(total))


def synthesize_principles_frontier(
    *,
    outdir: str | Path,
    ctx: EvalContext,
    profile_name: str = "iso_litvin_v1",
    seed: int = 42,
    seed_count: int = 64,
    min_frontier_size: int = 8,
    root_max_iter: int = 80,
    export_archive_dir: str | Path | None = None,
    contract_version: str = "",
    allow_nonproduction_paths: bool = False,
) -> PrinciplesFrontierResult:
    out_root = Path(outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    repo_root = _repo_root()

    profile_path, profile_payload = load_principles_profile(profile_name, repo_root=repo_root)
    anchor_path, anchor_manifest = _load_anchor_manifest(profile_payload, repo_root)
    operating_points = _build_operating_points(
        profile_payload=profile_payload,
        anchor_manifest=anchor_manifest,
    )
    if not operating_points:
        raise RuntimeError("Principles frontier synthesis requires at least one operating point.")

    xl, xu = bounds()
    seeds = _generate_seed_vectors(
        seed_count=max(int(seed_count), len(operating_points)),
        seed=int(seed),
        xl=xl,
        xu=xu,
    )

    X_all: list[np.ndarray] = []
    F_all: list[np.ndarray] = []
    G_all: list[np.ndarray] = []
    hard_feasible: list[bool] = []
    hard_scores: list[float] = []
    op_indices: list[int] = []
    root_success_count = 0

    root_trace_file = out_root / "principles_rootfinding_trace.jsonl"
    with root_trace_file.open("w", encoding="utf-8") as trace:
        for i in range(int(seeds.shape[0])):
            op_idx = int(i % len(operating_points))
            op = operating_points[op_idx]
            ctx_i = replace(ctx, rpm=float(op.rpm), torque=float(op.torque))
            x_seed = np.asarray(seeds[i], dtype=np.float64)
            x_restored, restore_diag, F_i, G_i = _restore_candidate(
                x_seed=x_seed,
                ctx=ctx_i,
                root_max_iter=int(root_max_iter),
                xl=xl,
                xu=xu,
            )
            success = bool(restore_diag.get("root_success", False))
            hard_score = float(restore_diag.get("hard_violation_score", float("inf")))
            if success:
                root_success_count += 1

            trace_row = {
                "seed_index": int(i),
                "operating_point_index": int(op_idx),
                "operating_point": {"rpm": float(op.rpm), "torque": float(op.torque)},
                "source": str(op.source),
                "root_success": success,
                "hard_violation_score": hard_score,
                "hard_reasons": list(restore_diag.get("hard_reasons", [])),
                "stage_diagnostics": list(restore_diag.get("stages", [])),
            }
            trace.write(json.dumps(_jsonify(trace_row), sort_keys=True) + "\n")

            X_all.append(np.asarray(x_restored, dtype=np.float64))
            F_all.append(np.asarray(F_i, dtype=np.float64))
            G_all.append(np.asarray(G_i, dtype=np.float64))
            hard_feasible.append(success)
            hard_scores.append(hard_score)
            op_indices.append(op_idx)

    X_mat = np.asarray(X_all, dtype=np.float64).reshape(-1, N_TOTAL)
    F_mat = np.asarray(F_all, dtype=np.float64)
    G_mat = np.asarray(G_all, dtype=np.float64)
    hard_idx = np.where(np.asarray(hard_feasible, dtype=bool))[0].astype(int)
    hard_scores_arr = np.asarray(hard_scores, dtype=np.float64).reshape(-1)

    nd_global: np.ndarray
    fallback_idx = np.zeros(0, dtype=int)
    frontier_idx: list[int]
    if hard_idx.size == 0:
        nd_global = np.zeros(0, dtype=int)
        order = np.argsort(hard_scores_arr)
        fallback_n = min(max(1, int(min_frontier_size)), int(order.size))
        fallback_idx = order[:fallback_n].astype(int)
        if fallback_idx.size > 0:
            nd_local = _non_dominated_indices(F_mat[fallback_idx])
            frontier_idx = [int(i) for i in fallback_idx[nd_local].tolist()]
            if not frontier_idx:
                frontier_idx = [int(fallback_idx[0])]
        else:
            frontier_idx = []
    else:
        nd_local = _non_dominated_indices(F_mat[hard_idx])
        nd_global = hard_idx[nd_local]
        frontier_idx = [int(i) for i in nd_global.tolist()]

        if len(frontier_idx) < int(min_frontier_size):
            norm_F = _normalize_objectives(F_mat[hard_idx])
            weights = np.asarray(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.5, 0.5, 0.3, 0.3, 0.3],
                    [0.5, 1.0, 0.5, 0.3, 0.3, 0.3],
                    [0.5, 0.5, 1.0, 0.3, 0.3, 0.3],
                    [0.8, 0.8, 0.8, 0.6, 0.6, 0.6],
                    [0.3, 0.3, 0.3, 1.0, 1.0, 1.0],
                ],
                dtype=np.float64,
            )
            for w in weights:
                if len(frontier_idx) >= int(min_frontier_size):
                    break
                scores = norm_F @ w[: norm_F.shape[1]]
                order = np.argsort(scores)
                for j in order.tolist():
                    g_idx = int(hard_idx[int(j)])
                    if g_idx not in frontier_idx:
                        frontier_idx.append(g_idx)
                        break

    frontier_idx = sorted(set(frontier_idx))
    if frontier_idx:
        f_idx = np.asarray(frontier_idx, dtype=int)
        frontier_X = X_mat[f_idx]
        frontier_F = F_mat[f_idx]
        frontier_G = G_mat[f_idx]
    else:
        frontier_X = np.zeros((0, N_TOTAL), dtype=np.float64)
        n_obj = int(F_mat.shape[1]) if F_mat.ndim == 2 and F_mat.shape[1] > 0 else 6
        n_constr = int(G_mat.shape[1]) if G_mat.ndim == 2 and G_mat.shape[1] > 0 else 0
        frontier_F = np.zeros((0, n_obj), dtype=np.float64)
        frontier_G = np.zeros((0, n_constr), dtype=np.float64)

    point_rpms = {float(p.rpm) for p in operating_points}
    point_torques = {float(p.torque) for p in operating_points}
    if hard_idx.size > 0:
        coverage_idx = hard_idx
    elif int(ctx.fidelity) == 0:
        coverage_idx = np.arange(X_mat.shape[0], dtype=int)
    else:
        coverage_idx = fallback_idx
    seen_rpms = {float(operating_points[op_indices[i]].rpm) for i in coverage_idx.tolist()}
    seen_torques = {float(operating_points[op_indices[i]].torque) for i in coverage_idx.tolist()}

    thresholds = profile_payload.get("gate_thresholds", {})
    coverage_rpm_min = float(thresholds.get("coverage_rpm_fraction_min", 0.6))
    coverage_torque_min = float(thresholds.get("coverage_torque_fraction_min", 0.6))
    profile_min_size = int(thresholds.get("min_frontier_size", 0))
    cli_min_size = int(min_frontier_size)
    min_size_req = int(max(1, cli_min_size if cli_min_size > 0 else profile_min_size))

    coverage_rpm = _coverage_fraction(seen_rpms, point_rpms)
    coverage_torque = _coverage_fraction(seen_torques, point_torques)
    n_hard = int(hard_idx.size)
    n_nondominated = int(nd_global.size)
    gate_basis = "hard_feasible"
    gate_n_feasible = int(n_hard)
    gate_n_nondominated = int(n_nondominated)
    if int(ctx.fidelity) == 0 and n_hard == 0:
        gate_basis = "placeholder_frontier"
        gate_n_feasible = int(frontier_X.shape[0])
        gate_n_nondominated = int(frontier_X.shape[0])
    frontier_gate_pass = bool(
        gate_n_feasible >= int(min_size_req)
        and gate_n_nondominated >= int(min_size_req)
        and coverage_rpm >= coverage_rpm_min
        and coverage_torque >= coverage_torque_min
    )
    placeholder_frontier_disallowed = bool(
        gate_basis == "placeholder_frontier" and not bool(allow_nonproduction_paths)
    )
    if placeholder_frontier_disallowed:
        frontier_gate_pass = False

    gate_payload = {
        "profile_name": str(profile_name),
        "profile_path": str(profile_path),
        "contract_version": str(contract_version or ""),
        "n_operating_points": int(len(operating_points)),
        "n_seeds": int(seeds.shape[0]),
        "n_root_success": int(root_success_count),
        "n_frontier_candidates": int(frontier_X.shape[0]),
        "n_fallback_selected": int(fallback_idx.size),
        "n_hard_feasible_explore": int(n_hard),
        "n_nondominated": int(n_nondominated),
        "n_gate_feasible_explore": int(gate_n_feasible),
        "n_gate_nondominated": int(gate_n_nondominated),
        "gate_basis": str(gate_basis),
        "coverage_rpm_fraction": float(coverage_rpm),
        "coverage_torque_fraction": float(coverage_torque),
        "min_frontier_size_required": int(min_size_req),
        "coverage_rpm_fraction_min": float(coverage_rpm_min),
        "coverage_torque_fraction_min": float(coverage_torque_min),
        "frontier_gate_pass": bool(frontier_gate_pass),
        "placeholder_frontier_disallowed": bool(placeholder_frontier_disallowed),
        "allow_nonproduction_paths": bool(allow_nonproduction_paths),
    }
    gate_path = out_root / "principles_frontier_gate.json"
    gate_path.write_text(json.dumps(_jsonify(gate_payload), indent=2), encoding="utf-8")

    frontier_summary = {
        "profile_name": str(profile_name),
        "profile_path": str(profile_path),
        "anchor_manifest_path": str(anchor_path),
        "n_operating_points": int(len(operating_points)),
        "n_seeds": int(seeds.shape[0]),
        "n_root_success": int(root_success_count),
        "n_hard_feasible_explore": int(n_hard),
        "n_nondominated": int(n_nondominated),
        "n_fallback_selected": int(fallback_idx.size),
        "gate_basis": str(gate_basis),
        "frontier_size": int(frontier_X.shape[0]),
        "frontier_indices": frontier_idx,
        "coverage_rpm_fraction": float(coverage_rpm),
        "coverage_torque_fraction": float(coverage_torque),
        "frontier_gate_pass": bool(frontier_gate_pass),
        "source_references": list(profile_payload.get("source_references", [])),
    }
    summary_path = out_root / "principles_frontier_summary.json"
    summary_path.write_text(json.dumps(_jsonify(frontier_summary), indent=2), encoding="utf-8")

    np.save(out_root / "principles_frontier_X.npy", frontier_X)
    np.save(out_root / "principles_frontier_F.npy", frontier_F)
    np.save(out_root / "principles_frontier_G.npy", frontier_G)

    archive_dir = (
        Path(export_archive_dir)
        if export_archive_dir is not None
        else out_root / "principles_pareto"
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    objective_names = [
        "eta_comb_gap",
        "eta_exp_gap",
        "eta_gear_gap",
        "motion_law_penalty",
        "life_damage_penalty",
        "material_risk_penalty",
    ][: int(frontier_F.shape[1])]
    archive_summary = {
        "n_pareto": int(frontier_X.shape[0]),
        "n_evals": int(seeds.shape[0]),
        "rpm": float(ctx.rpm),
        "torque": float(ctx.torque),
        "fidelity": int(ctx.fidelity),
        "seed": int(seed),
        "objective_names": objective_names,
        "profile_name": str(profile_name),
        "profile_path": str(profile_path),
        "frontier_gate_pass": bool(frontier_gate_pass),
    }
    save_archive(archive_dir, frontier_X, frontier_F, frontier_G, archive_summary)
    store = CandidateStore.from_arrays(
        X=frontier_X,
        F=frontier_F,
        G=frontier_G,
        summary=archive_summary,
        source_dir=archive_dir,
    )

    artifacts = {
        "principles_frontier_X": str(out_root / "principles_frontier_X.npy"),
        "principles_frontier_F": str(out_root / "principles_frontier_F.npy"),
        "principles_frontier_G": str(out_root / "principles_frontier_G.npy"),
        "principles_frontier_summary": str(summary_path),
        "principles_frontier_gate": str(gate_path),
        "principles_rootfinding_trace": str(root_trace_file),
        "principles_export_archive_dir": str(archive_dir),
    }
    return PrinciplesFrontierResult(
        store=store,
        pareto_source=archive_dir,
        profile_name=str(profile_name),
        profile_path=profile_path,
        profile_payload=profile_payload,
        gate=gate_payload,
        artifacts=artifacts,
    )
