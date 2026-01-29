# Larrak V1 Forward-Evaluation Port

This package contains **pure forward-evaluation logic** ported from Larrak v1.

## What Was Ported

### Gear Forward (`gear_forward.py`)

| v1 Source | Function | Purpose |
|-----------|----------|---------|
| `campro/physics/geometry/curvature.py` | `compute_curvature` | Polar curve curvature κ(θ) and osculating radius ρ(θ) |
| `campro/physics/geometry/litvin.py` | `LitvinSynthesis.synthesize_from_cam_profile` | Conjugate ring profile synthesis |
| `campro/physics/geometry/litvin.py` | `LitvinGearGeometry.from_synthesis` | Gear geometry metrics |

### Thermo Forward (`thermo_forward.py`)

| v1 Source | Function | Purpose |
|-----------|----------|---------|
| `campro/physics/chem.py` | `wiebe_function` | Cumulative burn fraction |
| `campro/physics/chem.py` | `wiebe_heat_release_rate` | Instantaneous heat release rate |
| `campro/physics/chem.py` | `CombustionParameters` | Parameter dataclass |

## What Was NOT Ported

- CasADi symbolic expressions (`campro/physics/casadi/`)
- NLP solver integration (`campro/optimization/nlp/`)
- Optimizer-coupled adapters (`simple_cycle_adapter.py`)
- Any `run_*` scripts or solver loops

## Constraint Sign Convention

Larrak v1 uses **mixed conventions**. This port normalizes all constraints to:

```
G <= 0  →  feasible (larrak2 convention)
```

Where v1 uses `G >= 0` for feasible, we apply `G *= -1`.

## Usage

```python
from larrak2.core.types import EvalContext
from larrak2.ports.larrak_v1 import v1_eval_gear_forward

ctx = EvalContext(rpm=3000.0, torque=200.0, fidelity=1, seed=42)
result = v1_eval_gear_forward(gear_params, i_req_profile, ctx)
```

## Environment Variable

To enable v1 port in tests when v1 repo is available:

```bash
export LARRAK2_ENABLE_V1=1
```

Tests skip gracefully when this is not set.
