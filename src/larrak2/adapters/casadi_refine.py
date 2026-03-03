"""CasADi/Ipopt-first local refinement with SciPy fallback."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from ..core.encoding import N_TOTAL, bounds
from ..core.evaluator import evaluate_candidate
from ..core.types import EvalContext
from ..optimization.slicing import select_active_set, solve_slice_with_ipopt


class RefinementMode(StrEnum):
    """Refinement strategy."""

    WEIGHTED_SUM = "weighted_sum"
    EPS_CONSTRAINT = "eps_constraint"


@dataclass
class RefinementResult:
    """Result from candidate refinement."""

    x_refined: np.ndarray
    F_refined: np.ndarray
    G_refined: np.ndarray
    diag: dict[str, Any]
    success: bool
    message: str
    backend_used: str
    ipopt_status: str = ""


def _scalarized_objective(
    F: np.ndarray,
    G: np.ndarray,
    *,
    mode: RefinementMode,
    weights: np.ndarray | None,
    eps_constraints: np.ndarray | None,
    violation_penalty: float = 10.0,
) -> float:
    F = np.asarray(F, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    if mode == RefinementMode.WEIGHTED_SUM:
        w = np.asarray(weights if weights is not None else np.ones(F.size), dtype=np.float64)
        if w.size != F.size:
            raise ValueError(f"weights length {w.size} != n_obj {F.size}")
        objective = float(np.dot(w, F))
    elif mode == RefinementMode.EPS_CONSTRAINT:
        objective = float(F[0])
        if F.size > 1:
            eps = np.asarray(
                eps_constraints if eps_constraints is not None else F, dtype=np.float64
            )
            if eps.size != F.size:
                raise ValueError(f"eps length {eps.size} != n_obj {F.size}")
            objective += violation_penalty * float(np.maximum(F[1:] - eps[1:], 0.0).sum())
    else:
        raise ValueError(f"Unknown mode: {mode}")

    g_violation = np.maximum(G, 0.0)
    objective += violation_penalty * float(np.dot(g_violation, g_violation))
    return objective


def _normalize_active_indices(active_set: list[int] | None) -> list[int]:
    if active_set is None:
        return []
    dedup = sorted({int(i) for i in active_set if 0 <= int(i) < N_TOTAL})
    return dedup


def _to_freeze_mask(freeze_mask: np.ndarray | None) -> np.ndarray | None:
    if freeze_mask is None:
        return None
    mask = np.asarray(freeze_mask, dtype=bool).reshape(-1)
    if mask.size != N_TOTAL:
        raise ValueError(f"freeze_mask must have length {N_TOTAL}, got {mask.size}")
    return mask


def _reconstruct_full(x_base: np.ndarray, active_indices: list[int], z: np.ndarray) -> np.ndarray:
    x = np.array(x_base, dtype=np.float64, copy=True)
    for j, idx in enumerate(active_indices):
        x[idx] = float(z[j])
    return x


def _scipy_refine_slice(
    x0: np.ndarray,
    ctx: EvalContext,
    active_indices: list[int],
    *,
    mode: RefinementMode,
    weights: np.ndarray | None,
    eps_constraints: np.ndarray | None,
    max_iter: int,
    tol: float,
    trust_radius: float | None,
) -> tuple[np.ndarray, dict[str, Any], bool, str]:
    if not active_indices:
        return x0, {"n_iter": 0}, True, "No active variables selected"

    try:
        from scipy.optimize import minimize
    except Exception:
        return x0, {"error": "scipy not available"}, False, "SciPy is not available"

    xl, xu = bounds()
    z0 = np.array([x0[i] for i in active_indices], dtype=np.float64)
    lb = np.array([xl[i] for i in active_indices], dtype=np.float64)
    ub = np.array([xu[i] for i in active_indices], dtype=np.float64)
    if trust_radius is not None:
        radius = float(trust_radius)
        if radius <= 0:
            return (
                x0,
                {"error": f"trust_radius must be > 0, got {radius}"},
                False,
                "invalid trust_radius",
            )
        lb = np.maximum(lb, z0 - radius)
        ub = np.minimum(ub, z0 + radius)

    def objective(z: np.ndarray) -> float:
        x = _reconstruct_full(x0, active_indices, np.clip(z, lb, ub))
        res = evaluate_candidate(x, ctx)
        return _scalarized_objective(
            res.F,
            res.G,
            mode=mode,
            weights=weights,
            eps_constraints=eps_constraints,
        )

    def constraints_fn(z: np.ndarray) -> np.ndarray:
        x = _reconstruct_full(x0, active_indices, np.clip(z, lb, ub))
        res = evaluate_candidate(x, ctx)
        all_g = [-res.G]
        if mode == RefinementMode.EPS_CONSTRAINT and len(res.F) > 1:
            eps = np.asarray(
                eps_constraints if eps_constraints is not None else res.F, dtype=np.float64
            )
            all_g.append(-(res.F[1:] - eps[1:]))
        return np.concatenate(all_g)

    res = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=list(zip(lb, ub)),
        constraints={"type": "ineq", "fun": constraints_fn},
        options={"maxiter": max_iter, "ftol": tol, "disp": False},
    )

    z_opt = np.clip(np.asarray(res.x, dtype=np.float64), lb, ub)
    x_opt = _reconstruct_full(x0, active_indices, z_opt)
    return (
        x_opt,
        {
            "scipy_result": str(res.message),
            "n_iter": int(getattr(res, "nit", 0)),
            "success": bool(res.success),
            "trust_radius": trust_radius,
        },
        bool(res.success),
        str(res.message),
    )


def refine_candidate(
    x0: np.ndarray,
    ctx: EvalContext,
    mode: RefinementMode = RefinementMode.WEIGHTED_SUM,
    weights: np.ndarray | None = None,
    eps_constraints: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    backend: str = "casadi",
    active_set: list[int] | None = None,
    ipopt_options: dict[str, Any] | None = None,
    freeze_mask: np.ndarray | None = None,
    active_k: int | None = None,
    min_per_group: int = 1,
    slice_method: str = "sensitivity",
    trust_radius: float | None = None,
) -> RefinementResult:
    """Refine candidate in a high-dimensional space using active-variable slices."""
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.size != N_TOTAL:
        raise ValueError(f"Expected x0 length {N_TOTAL}, got {x0.size}")

    mode = RefinementMode(mode)
    freeze_mask_arr = _to_freeze_mask(freeze_mask)

    selection_scores: list[float] = []
    selected_active: list[int]

    explicit_active = _normalize_active_indices(active_set)
    if explicit_active:
        selected_active = explicit_active
    else:
        selection = select_active_set(
            x0,
            ctx,
            active_k=active_k,
            min_per_group=min_per_group,
            method=slice_method,
            mode=str(mode.value),
            weights=weights,
            eps_constraints=eps_constraints,
            freeze_mask=freeze_mask_arr,
        )
        selected_active = selection.active_indices
        selection_scores = selection.scores

    frozen_indices = [i for i in range(N_TOTAL) if i not in selected_active]

    base_eval = evaluate_candidate(x0, ctx)
    x_refined = x0.copy()
    backend_used = "scipy"
    ipopt_status = ""
    success = False
    message = "Refinement failed"
    diag: dict[str, Any] = {
        "active_indices": selected_active,
        "frozen_indices": frozen_indices,
        "slice_scores": selection_scores,
        "trust_radius": trust_radius,
    }

    if backend == "casadi":
        slice_result = solve_slice_with_ipopt(
            x0,
            ctx,
            selected_active,
            mode=mode.value,
            weights=weights,
            eps_constraints=eps_constraints,
            ipopt_options=ipopt_options,
            trust_radius=trust_radius,
        )
        ipopt_status = slice_result.ipopt_status
        diag["ipopt"] = slice_result.diagnostics

        if slice_result.success:
            x_refined = slice_result.x_opt
            backend_used = "casadi"
            success = True
            message = slice_result.message
        else:
            # Controlled fallback keeps the same active/frozen slice partition.
            x_refined, scipy_diag, scipy_success, scipy_msg = _scipy_refine_slice(
                x0,
                ctx,
                selected_active,
                mode=mode,
                weights=weights,
                eps_constraints=eps_constraints,
                max_iter=max_iter,
                tol=tol,
                trust_radius=trust_radius,
            )
            diag["scipy"] = scipy_diag
            backend_used = "scipy_fallback"
            success = scipy_success
            message = f"{slice_result.message}; fallback: {scipy_msg}"
    elif backend == "scipy":
        x_refined, scipy_diag, scipy_success, scipy_msg = _scipy_refine_slice(
            x0,
            ctx,
            selected_active,
            mode=mode,
            weights=weights,
            eps_constraints=eps_constraints,
            max_iter=max_iter,
            tol=tol,
            trust_radius=trust_radius,
        )
        diag["scipy"] = scipy_diag
        backend_used = "scipy"
        success = scipy_success
        message = scipy_msg
    else:
        message = f"Unknown backend: {backend}"

    result_final = evaluate_candidate(x_refined, ctx)

    # Mark success conservatively: solve success + finite outputs.
    finite_ok = np.all(np.isfinite(result_final.F)) and np.all(np.isfinite(result_final.G))
    success = bool(success and finite_ok)

    # If refinement completely failed, preserve base candidate outputs.
    if not success and backend == "casadi" and backend_used == "scipy_fallback":
        if x_refined.shape != x0.shape:
            x_refined = x0.copy()
            result_final = base_eval

    return RefinementResult(
        x_refined=x_refined,
        F_refined=result_final.F,
        G_refined=result_final.G,
        diag={
            **diag,
            "backend_used": backend_used,
            "ipopt_status": ipopt_status,
            "message": message,
        },
        success=success,
        message=message,
        backend_used=backend_used,
        ipopt_status=ipopt_status,
    )
