"""Larrak v1 forward-evaluation port.

This package contains pure forward-evaluation logic ported from Larrak v1.
NO optimizer loops, pack/unpack schemes, or global state are included.

Ported with attribution from:
- campro/physics/chem.py (Wiebe functions)
- campro/physics/geometry/litvin.py (LitvinSynthesis)
- campro/physics/geometry/curvature.py (curvature computation)

Usage:
    from larrak2.ports.larrak_v1 import v1_eval_gear_forward, v1_eval_thermo_forward
"""

from .gear_forward import v1_eval_gear_forward
from .thermo_forward import v1_eval_thermo_forward

__all__ = ["v1_eval_gear_forward", "v1_eval_thermo_forward"]
