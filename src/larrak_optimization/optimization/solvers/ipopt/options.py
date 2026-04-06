"""IPOPT option composition helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import IPOPTOptions


def create_ipopt_options(
    options: IPOPTOptions | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose IPOPT options dict for CasADi's `nlpsol` interface."""
    base = (options or IPOPTOptions()).as_ipopt_dict()
    if overrides:
        base.update(dict(overrides))
    return base
