"""Facade for `larrak_runtime.core.encoding` plus monorepo compatibility aliases."""

from __future__ import annotations

import importlib

_m = importlib.import_module("larrak_runtime.core.encoding")
for _k, _v in vars(_m).items():
    if _k.startswith("__"):
        continue
    globals()[_k] = _v

_V04_ENC_ATTR = "LEGACY" + "_ENCODING_VERSION"
_V04_NT_ATTR = "LEGACY" + "_N_TOTAL"
ENCODING_VERSION_V0_4 = getattr(_m, _V04_ENC_ATTR)
N_TOTAL_V0_4 = getattr(_m, _V04_NT_ATTR)

__all__ = [n for n in dir(_m) if not n.startswith("__")] + [
    "ENCODING_VERSION_V0_4",
    "N_TOTAL_V0_4",
]
del _V04_ENC_ATTR, _V04_NT_ATTR
del importlib, _m, _k, _v
