"""Nonlinear symbolic slice NLP solver backed by the global surrogate stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...core.encoding import N_TOTAL, bounds
from ...core.evaluator import evaluate_candidate
from ...core.types import EvalContext
from ...surrogate.stack import (
    StackSurrogateArtifact,
    assemble_symbolic_feature_vector,
    load_stack_artifact,
    symbolic_objectives_constraints,
)
from ..solvers.ipopt import IPOPTOptions, IPOPTSolver


@dataclass
class SymbolicSliceSolveResult:
    """Result of solving nonlinear surrogate-symbolic slice NLP."""

    x_opt: np.ndarray
    success: bool
    message: str
    ipopt_status: str
    iterations: int
    diagnostics: dict[str, Any]


def _import_casadi():
    try:
        import casadi as ca
    except Exception as exc:  # pragma: no cover - dependency handled by caller
        raise ImportError(f"CasADi unavailable: {exc}") from exc
    return ca


def _reconstruct_full_np(
    x_base: np.ndarray, active_indices: list[int], z: np.ndarray
) -> np.ndarray:
    x = np.array(x_base, dtype=np.float64, copy=True)
    for j, idx in enumerate(active_indices):
        x[idx] = float(z[j])
    return x


def _reconstruct_full_sym(x_base: np.ndarray, active_indices: list[int], z_sym):
    ca = _import_casadi()
    active_lookup = {int(idx): int(j) for j, idx in enumerate(active_indices)}
    elems = []
    for i in range(int(N_TOTAL)):
        j = active_lookup.get(i)
        if j is None:
            elems.append(float(x_base[i]))
        else:
            elems.append(z_sym[j])
    return ca.vertcat(*elems)


def _fit_weights(weights: np.ndarray | None, n_obj: int):
    if weights is None:
        return np.ones(n_obj, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == n_obj:
        return w
    if w.size < n_obj:
        out = np.zeros(n_obj, dtype=np.float64)
        out[: w.size] = w
        return out
    return w[:n_obj]


def _build_nlp(
    *,
    artifact: StackSurrogateArtifact,
    x_base: np.ndarray,
    ctx: EvalContext,
    active_indices: list[int],
    mode: str,
    weights: np.ndarray | None,
    eps_constraints: np.ndarray | None,
    regularization: float,
):
    ca = _import_casadi()
    z0 = np.asarray([x_base[i] for i in active_indices], dtype=np.float64)
    z = ca.MX.sym("z", len(active_indices))
    z0_dm = ca.DM(z0)
    delta = z - z0_dm

    x_full_sym = _reconstruct_full_sym(x_base, active_indices, z)
    feats_sym = assemble_symbolic_feature_vector(
        artifact=artifact,
        x_full_sym=x_full_sym,
        rpm=float(ctx.rpm),
        torque=float(ctx.torque),
    )
    F_hat, G_hat = symbolic_objectives_constraints(artifact, feats_sym)

    constraints = []
    lbg = []
    ubg = []

    for i in range(len(artifact.constraint_names)):
        constraints.append(G_hat[i])
        lbg.append(-np.inf)
        ubg.append(0.0)

    if mode == "eps_constraint":
        objective = F_hat[0]
        if len(artifact.objective_names) > 1:
            eps = np.asarray(
                eps_constraints
                if eps_constraints is not None
                else np.zeros(len(artifact.objective_names)),
                dtype=np.float64,
            )
            if eps.size != len(artifact.objective_names):
                raise ValueError(
                    f"eps_constraints length {eps.size} does not match n_obj {len(artifact.objective_names)}"
                )
            for i in range(1, len(artifact.objective_names)):
                constraints.append(F_hat[i])
                lbg.append(-np.inf)
                ubg.append(float(eps[i]))
    elif mode == "weighted_sum":
        w = _fit_weights(weights, len(artifact.objective_names))
        objective = ca.dot(ca.DM(w), F_hat)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    if float(regularization) > 0.0:
        objective = objective + 0.5 * float(regularization) * ca.dot(delta, delta)

    g_expr = ca.vertcat(*constraints) if constraints else ca.MX([])
    nlp = {"x": z, "f": objective, "g": g_expr}
    return nlp, z0, np.asarray(lbg, dtype=np.float64), np.asarray(ubg, dtype=np.float64)


def _build_bounds(
    *,
    x_base: np.ndarray,
    active_indices: list[int],
    trust_radius: float | None,
):
    xl, xu = bounds()
    z0 = np.asarray([x_base[i] for i in active_indices], dtype=np.float64)
    lb = np.asarray([xl[i] for i in active_indices], dtype=np.float64)
    ub = np.asarray([xu[i] for i in active_indices], dtype=np.float64)
    if trust_radius is not None:
        radius = float(trust_radius)
        if radius <= 0:
            raise ValueError(f"trust_radius must be > 0, got {radius}")
        lb = np.maximum(lb, z0 - radius)
        ub = np.minimum(ub, z0 + radius)
    return lb, ub


def solve_symbolic_slice_with_ipopt(
    x0: np.ndarray,
    ctx: EvalContext,
    active_indices: list[int],
    *,
    surrogate_stack_path: str,
    mode: str = "eps_constraint",
    weights: np.ndarray | None = None,
    eps_constraints: np.ndarray | None = None,
    ipopt_options: dict[str, Any] | None = None,
    regularization: float = 1e-2,
    trust_radius: float | None = None,
    validation_attempts: int = 3,
    validation_tol: float = 1e-8,
    fidelity: int = 1,
) -> SymbolicSliceSolveResult:
    """Solve nonlinear surrogate-symbolic NLP for active slice with post-validation."""
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x0.size != N_TOTAL:
        raise ValueError(f"Expected x0 length {N_TOTAL}, got {x0.size}")
    if not active_indices:
        return SymbolicSliceSolveResult(
            x_opt=x0,
            success=True,
            message="No active variables selected",
            ipopt_status="no_active_variables",
            iterations=0,
            diagnostics={"nlp_formulation": "global_surrogate_symbolic"},
        )

    artifact = load_stack_artifact(
        surrogate_stack_path,
        validation_mode=str(getattr(ctx, "surrogate_validation_mode", "strict")),
    )
    if int(artifact.fidelity) != int(fidelity):
        raise ValueError(
            f"Stack surrogate fidelity mismatch: artifact={artifact.fidelity}, requested={fidelity}"
        )

    ca = _import_casadi()
    ctx_eval = replace(ctx, fidelity=int(fidelity))
    base_eval = evaluate_candidate(x0, ctx_eval)
    base_violation = float(max(0.0, np.max(base_eval.G))) if base_eval.G.size else 0.0

    attempt_radius = float(trust_radius) if trust_radius is not None else 0.25
    attempts = max(1, int(validation_attempts))

    last_status = "not_solved"
    last_msg = "symbolic solve did not run"
    last_diag: dict[str, Any] = {}
    last_x = np.array(x0, copy=True)
    total_iterations = 0

    for attempt in range(attempts):
        radius = attempt_radius if attempt_radius > 0 else 1e-3
        try:
            nlp, z0, lbg, ubg = _build_nlp(
                artifact=artifact,
                x_base=x0,
                ctx=ctx_eval,
                active_indices=active_indices,
                mode=mode,
                weights=weights,
                eps_constraints=eps_constraints,
                regularization=float(regularization),
            )
            lbx, ubx = _build_bounds(
                x_base=x0,
                active_indices=active_indices,
                trust_radius=radius,
            )

            option_map = ipopt_options or {}
            solver = IPOPTSolver(
                options=IPOPTOptions(
                    max_iter=int(option_map.get("max_iter", 200)),
                    tol=float(option_map.get("tol", 1e-6)),
                    print_level=int(option_map.get("print_level", 0)),
                    linear_solver=str(option_map.get("linear_solver", "mumps")),
                    extra={
                        k: v
                        for k, v in option_map.items()
                        if k not in {"max_iter", "tol", "print_level", "linear_solver"}
                    },
                )
            )
            res = solver.solve(
                nlp,
                x0=z0,
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg,
            )
            total_iterations += int(res.iterations)
            last_status = str(res.status)
            last_x = _reconstruct_full_np(
                x0, active_indices, np.asarray(res.x_opt, dtype=np.float64)
            )

            truth_eval = evaluate_candidate(last_x, ctx_eval)
            hard_violation = float(max(0.0, np.max(truth_eval.G))) if truth_eval.G.size else 0.0
            finite_ok = bool(
                np.all(np.isfinite(truth_eval.F)) and np.all(np.isfinite(truth_eval.G))
            )

            last_diag = {
                "f_opt_surrogate": float(res.f_opt),
                "attempt": int(attempt + 1),
                "attempt_radius": float(radius),
                "truth_hard_violation": hard_violation,
                "base_hard_violation": base_violation,
                "surrogate_stack_version": artifact.version_hash,
                "nlp_formulation": "global_surrogate_symbolic",
            }

            violation_cap = max(base_violation + 1e-5, float(validation_tol))
            if bool(res.success) and finite_ok and hard_violation <= violation_cap:
                return SymbolicSliceSolveResult(
                    x_opt=last_x,
                    success=True,
                    message="IPOPT symbolic NLP solve finished",
                    ipopt_status=last_status,
                    iterations=total_iterations,
                    diagnostics={
                        **last_diag,
                        "validation_attempts": int(attempt + 1),
                        "trust_radius_final": float(radius),
                    },
                )

            last_msg = (
                "validation failed after IPOPT solve "
                f"(success={res.success}, finite={finite_ok}, hard_violation={hard_violation:.3e})"
            )
        except Exception as exc:
            last_status = "solve_exception"
            last_msg = f"symbolic slice solve failed: {exc}"
            last_diag = {
                "error": str(exc),
                "attempt": int(attempt + 1),
                "attempt_radius": float(radius),
                "surrogate_stack_version": artifact.version_hash,
                "nlp_formulation": "global_surrogate_symbolic",
            }

        attempt_radius *= 0.5
        if attempt_radius < 1e-4:
            attempt_radius = 1e-4

    return SymbolicSliceSolveResult(
        x_opt=last_x,
        success=False,
        message=last_msg,
        ipopt_status=last_status,
        iterations=total_iterations,
        diagnostics={
            **last_diag,
            "validation_attempts": int(attempts),
            "trust_radius_final": float(attempt_radius),
        },
    )
