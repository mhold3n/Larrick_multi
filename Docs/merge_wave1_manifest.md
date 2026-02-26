# Wave 1 Merge Manifest

This document tracks legacy-to-`larrak2` migration for Wave 1 (optimizer + surrogate core).

| Legacy Source | New Canonical Target | Status |
| --- | --- | --- |
| `Legacy_Monolithic/Larrak/src/campro/optimization/numerical/casadi_problem_spec.py` | `src/larrak2/optimization/numerical/casadi_problem_spec.py` | Ported + normalized imports |
| `Legacy_Monolithic/Larrak/src/campro/optimization/solvers/ipopt/*` | `src/larrak2/optimization/solvers/ipopt/*` | Ported as streamlined compatible stack |
| `Legacy_Monolithic/Larrak/src/campro/optimization/nlp/scaling/*` | `src/larrak2/optimization/scaling/*` | Selected utilities ported (variable/constraint/evaluation) |
| `Legacy_Monolithic/Larrak/src/campro/optimization/initialization/surrogate_adapter.py` | `src/larrak2/optimization/initialization/surrogate_adapter.py` | Ported + pandas dependency removed |
| `Legacy_Monolithic/Larrak/src/truthmaker/surrogates/models/ensemble.py` | `src/larrak2/surrogate/hifi/ensemble.py` | Ported |
| `Legacy_Monolithic/Larrak/src/truthmaker/surrogates/models/hifi_surrogates.py` | `src/larrak2/surrogate/hifi/models.py` | Ported |
| `Legacy_Monolithic/Larrak/src/Simulations/hifi/training_schema.py` | `src/larrak2/training/hifi_schema.py` | Ported |
| `Legacy_Monolithic/Larrak/src/Simulations/hifi/train_hifi_surrogates.py` | `src/larrak2/training/hifi_train.py` | Ported, removed `sys.path` hacks |
| *(new in larrak2)* | `src/larrak2/optimization/slicing/active_set.py` | Added |
| *(new in larrak2)* | `src/larrak2/optimization/slicing/slice_problem.py` | Added |
| `src/larrak2/adapters/casadi_refine.py` (existing) | `src/larrak2/adapters/casadi_refine.py` | Replaced internals: CasADi/Ipopt-first + SciPy fallback |
| `src/larrak2/cli/refine_pareto.py` (existing) | `src/larrak2/cli/refine_pareto.py` | Extended with backend/slicing/IPOPT flags + metadata |

## Excluded in Wave 1

- Legacy runtime artifacts (`_runs`, `_runs_orchestration`, `__pycache__`, `.DS_Store`)
- Legacy webapp/dashboard/provenance stacks
- Full monolithic tree relocation in a single pass

## Duplicate Resolution Rubric (Applied)

When both legacy and current implementations existed, the Wave 1 canonical choice was scored on:

1. Existing test coverage in this repo.
2. Feature completeness required by the Wave 1 scope.
3. Interface clarity with current `larrak2` APIs.
4. Failure handling and fallback behavior.
5. Dependency hygiene (avoid new heavyweight runtime requirements unless necessary).

Tie-breaker rule: retain the current `larrak2` public interface and graft stronger internals.
