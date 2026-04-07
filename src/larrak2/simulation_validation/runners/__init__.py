"""Facade: canonical implementation lives in `larrak_simulation.simulation_validation.runners` (submodule package).

This file is part of the Larrick_multi integration distribution only.
"""

from __future__ import annotations

import importlib

_canonical = importlib.import_module("larrak_simulation.simulation_validation.runners")
for _k, _v in vars(_canonical).items():
    if _k.startswith("__"):
        continue
    globals()[_k] = _v
del importlib, _canonical, _k, _v
