"""Factory methods for IPOPT solver objects."""

from __future__ import annotations

from typing import Any, Mapping

from .options import create_ipopt_options
from .types import IPOPTOptions


def create_ipopt_solver(
    name: str,
    nlp: dict[str, Any],
    *,
    options: IPOPTOptions | None = None,
    overrides: Mapping[str, Any] | None = None,
):
    """Create a CasADi IPOPT solver with consistent defaults."""
    import casadi as ca

    solver_opts = {
        "print_time": False,
        "ipopt": create_ipopt_options(options=options, overrides=overrides),
    }
    return ca.nlpsol(name, "ipopt", nlp, solver_opts)
