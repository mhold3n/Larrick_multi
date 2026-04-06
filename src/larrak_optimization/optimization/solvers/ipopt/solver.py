"""High-level IPOPT solver wrapper."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .diagnostics import summarize_stats
from .factory import create_ipopt_solver
from .log_parser import is_success_status
from .types import IPOPTOptions, IPOPTResult


class IPOPTSolver:
    """Simple CasADi IPOPT wrapper used by slice refinement."""

    def __init__(self, options: IPOPTOptions | None = None):
        self.options = options or IPOPTOptions()

    def solve(
        self,
        nlp: dict[str, Any],
        *,
        x0: np.ndarray,
        lbx: np.ndarray,
        ubx: np.ndarray,
        lbg: np.ndarray,
        ubg: np.ndarray,
        p: np.ndarray | None = None,
    ) -> IPOPTResult:
        start = time.perf_counter()
        solver = create_ipopt_solver("larrak_ipopt", nlp, options=self.options)
        res = solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
            p=np.array([]) if p is None else p,
        )
        elapsed = time.perf_counter() - start

        stats = solver.stats()
        diag = summarize_stats(stats)
        status = str(diag.get("return_status", ""))
        success = bool(diag.get("success", False)) or is_success_status(status)

        x_opt = np.asarray(res["x"], dtype=np.float64).reshape(-1)
        g_opt = np.asarray(res["g"], dtype=np.float64).reshape(-1)

        return IPOPTResult(
            x_opt=x_opt,
            f_opt=float(res["f"]),
            g_opt=g_opt,
            success=success,
            status=status,
            iterations=int(diag.get("iter_count", 0)),
            cpu_time_s=float(elapsed),
            stats=stats,
        )
