"""Local slice NLP builder and CasADi/Ipopt solver bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.encoding import bounds
from ...core.evaluator import evaluate_candidate
from ...core.types import EvalContext
from ..solvers.ipopt import IPOPTOptions, IPOPTSolver
from .active_set import _normalize_weights


@dataclass
class SliceSolveResult:
    """Result of solving a local active-variable slice."""

    x_opt: np.ndarray
    success: bool
    message: str
    ipopt_status: str
    iterations: int
    diagnostics: dict[str, Any]


def _reconstruct_full(x_base: np.ndarray, active_indices: list[int], z: np.ndarray) -> np.ndarray:
    x = np.array(x_base, dtype=np.float64, copy=True)
    for j, idx in enumerate(active_indices):
        x[idx] = float(z[j])
    return x


def _scalarized_value(
    F: np.ndarray,
    G: np.ndarray,
    *,
    mode: str,
    weights: np.ndarray | None,
    eps_constraints: np.ndarray | None,
    violation_penalty: float,
) -> float:
    F = np.asarray(F, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    if mode == "weighted_sum":
        w = _normalize_weights(weights, F.size)
        objective = float(np.dot(w, F))
    elif mode == "eps_constraint":
        objective = float(F[0])
        if F.size > 1:
            eps = np.asarray(eps_constraints if eps_constraints is not None else F, dtype=np.float64)
            objective += violation_penalty * float(np.maximum(F[1:] - eps[1:], 0.0).sum())
    else:
        raise ValueError(f"Unknown mode: {mode}")

    g_violation = np.maximum(G, 0.0)
    objective += violation_penalty * float(np.dot(g_violation, g_violation))
    return objective


def _local_derivatives(
    x0: np.ndarray,
    ctx: EvalContext,
    active_indices: list[int],
    *,
    mode: str,
    weights: np.ndarray | None,
    eps_constraints: np.ndarray | None,
    step_frac: float = 1e-3,
    violation_penalty: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finite-difference local derivatives for objectives/constraints around x0."""
    xl, xu = bounds()
    base = evaluate_candidate(x0, ctx)

    n_active = len(active_indices)
    n_obj = int(base.F.size)
    n_constr = int(base.G.size)

    grad_scalar = np.zeros(n_active, dtype=np.float64)
    grad_F = np.zeros((n_obj, n_active), dtype=np.float64)
    grad_G = np.zeros((n_constr, n_active), dtype=np.float64)

    for col, idx in enumerate(active_indices):
        delta = max(abs(x0[idx]) * step_frac, (xu[idx] - xl[idx]) * step_frac, 1e-8)

        xp = x0.copy()
        xm = x0.copy()
        xp[idx] = min(x0[idx] + delta, xu[idx])
        xm[idx] = max(x0[idx] - delta, xl[idx])

        plus = evaluate_candidate(xp, ctx)
        minus = evaluate_candidate(xm, ctx)

        denom = max(xp[idx] - xm[idx], 1e-12)
        grad_F[:, col] = (plus.F - minus.F) / denom
        grad_G[:, col] = (plus.G - minus.G) / denom

        s_plus = _scalarized_value(
            plus.F,
            plus.G,
            mode=mode,
            weights=weights,
            eps_constraints=eps_constraints,
            violation_penalty=violation_penalty,
        )
        s_minus = _scalarized_value(
            minus.F,
            minus.G,
            mode=mode,
            weights=weights,
            eps_constraints=eps_constraints,
            violation_penalty=violation_penalty,
        )
        grad_scalar[col] = (s_plus - s_minus) / denom

    return base.F, base.G, grad_scalar, grad_F, grad_G, x0[np.array(active_indices, dtype=int)]


def solve_slice_with_ipopt(
    x0: np.ndarray,
    ctx: EvalContext,
    active_indices: list[int],
    *,
    mode: str = "weighted_sum",
    weights: np.ndarray | None = None,
    eps_constraints: np.ndarray | None = None,
    ipopt_options: dict[str, Any] | None = None,
    regularization: float = 1e-2,
) -> SliceSolveResult:
    """Solve a local linearized slice NLP with CasADi/Ipopt."""
    if not active_indices:
        return SliceSolveResult(
            x_opt=np.asarray(x0, dtype=np.float64),
            success=True,
            message="No active variables selected",
            ipopt_status="no_active_variables",
            iterations=0,
            diagnostics={},
        )

    try:
        import casadi as ca
    except Exception as exc:
        return SliceSolveResult(
            x_opt=np.asarray(x0, dtype=np.float64),
            success=False,
            message=f"CasADi unavailable: {exc}",
            ipopt_status="casadi_unavailable",
            iterations=0,
            diagnostics={"error": str(exc)},
        )

    x0 = np.asarray(x0, dtype=np.float64)

    try:
        F0, G0, grad_scalar, grad_F, grad_G, z0 = _local_derivatives(
            x0,
            ctx,
            active_indices,
            mode=mode,
            weights=weights,
            eps_constraints=eps_constraints,
        )

        n_active = len(active_indices)
        z = ca.MX.sym("z", n_active)
        z0_dm = ca.DM(z0)

        delta = z - z0_dm
        objective = float(_scalarized_value(F0, G0, mode=mode, weights=weights, eps_constraints=eps_constraints, violation_penalty=10.0))
        objective += ca.dot(ca.DM(grad_scalar), delta)
        objective += 0.5 * float(regularization) * ca.dot(delta, delta)

        constraints = []
        lbg = []
        ubg = []

        for row in range(G0.size):
            g_lin = float(G0[row]) + ca.dot(ca.DM(grad_G[row]), delta)
            constraints.append(g_lin)
            lbg.append(-np.inf)
            ubg.append(0.0)

        if mode == "eps_constraint" and F0.size > 1:
            eps = np.asarray(eps_constraints if eps_constraints is not None else F0, dtype=np.float64)
            for i in range(1, F0.size):
                f_lin = float(F0[i]) + ca.dot(ca.DM(grad_F[i]), delta)
                constraints.append(f_lin)
                lbg.append(-np.inf)
                ubg.append(float(eps[i]))

        g_expr = ca.vertcat(*constraints) if constraints else ca.MX([])
        nlp = {"x": z, "f": objective, "g": g_expr}

        xl, xu = bounds()
        lbx = np.array([xl[i] for i in active_indices], dtype=np.float64)
        ubx = np.array([xu[i] for i in active_indices], dtype=np.float64)

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
                    if k
                    not in {
                        "max_iter",
                        "tol",
                        "print_level",
                        "linear_solver",
                    }
                },
            )
        )

        res = solver.solve(
            nlp,
            x0=z0,
            lbx=lbx,
            ubx=ubx,
            lbg=np.array(lbg, dtype=np.float64),
            ubg=np.array(ubg, dtype=np.float64),
        )

        x_opt = _reconstruct_full(x0, active_indices, res.x_opt)
        return SliceSolveResult(
            x_opt=x_opt,
            success=bool(res.success),
            message="IPOPT solve finished" if res.success else "IPOPT did not converge",
            ipopt_status=res.status,
            iterations=int(res.iterations),
            diagnostics={
                "f_opt": float(res.f_opt),
                "stats": res.stats,
                "active_dim": n_active,
            },
        )
    except Exception as exc:
        return SliceSolveResult(
            x_opt=np.asarray(x0, dtype=np.float64),
            success=False,
            message=f"Slice solve failed: {exc}",
            ipopt_status="solve_exception",
            iterations=0,
            diagnostics={"error": str(exc)},
        )
