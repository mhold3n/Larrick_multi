"""Config adapter for IPOPT option generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import IPOPTOptions


class IPOPTConfigAdapter:
    """Adapter from plain dictionaries to IPOPTOptions."""

    @staticmethod
    def from_mapping(data: Mapping[str, Any] | None = None) -> IPOPTOptions:
        if not data:
            return IPOPTOptions()

        known = {
            "max_iter",
            "tol",
            "print_level",
            "linear_solver",
            "warm_start_init_point",
            "hessian_approximation",
            "jacobian_approximation",
            "acceptable_tol",
            "acceptable_iter",
        }
        kwargs = {k: data[k] for k in known if k in data}
        extra = {k: v for k, v in data.items() if k not in known}
        opt = IPOPTOptions(**kwargs)
        opt.extra.update(extra)
        return opt
