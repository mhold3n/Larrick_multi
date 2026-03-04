# Thermo Symbolic Hardening Runbook

Status: active  
Last updated: 2026-03-04

## 1. Default Runtime Behavior

1. `EvalContext.thermo_symbolic_mode` defaults to `strict`.
2. Symbolic entrypoints default to strict:
   - `larrak2.cli.run explore-exploit`
   - `larrak2.cli.run orchestrate`
   - `larrak2.cli.refine_pareto`
3. In strict mode, missing/invalid thermo symbolic artifacts fail fast with remediation text.

## 2. Thermo Symbolic Quality Profile (Balanced)

The thermo symbolic quality contract uses per-target normalized gates:

1. `normalization_method = p95_p05_range`
2. `val_nrmse <= 0.20` for every target
3. `test_nrmse <= 0.25` for every target
4. `r2 >= 0.40` on both `val` and `test` for every target

Where:

1. `nrmse = rmse / max(p95(y_train) - p05(y_train), 1e-9)`

## 3. Validation Modes

1. `strict`:
   - Raises on artifact load/contract/quality failures.
   - Includes target-specific failing metrics in the error.
2. `warn`:
   - Continues execution.
   - Emits warning logs and `thermo_symbolic_error` degradation diagnostics.
3. `off`:
   - Skips thermo symbolic quality gate enforcement.

## 4. Required Diagnostics Keys

Refinement and orchestration payloads should always carry:

1. `thermo_symbolic_mode`
2. `thermo_symbolic_used`
3. `thermo_symbolic_version`
4. `thermo_symbolic_path`
5. `thermo_symbolic_overlay_objectives`
6. `thermo_symbolic_overlay_constraints`
7. `thermo_symbolic_error` (when fallback/degradation/failure occurs)

## 5. Remediation Commands

Train a fidelity-matching thermo symbolic artifact:

```bash
python -m larrak2.cli.run train-thermo-symbolic \
  --fidelity <fidelity> \
  --rpm 3000 \
  --torque 200
```

Canonical artifact path:

1. `outputs/artifacts/surrogates/thermo_symbolic_f{fidelity}/thermo_symbolic_f{fidelity}.npz`

Legacy compatibility note:

1. `fidelity=1` may auto-resolve legacy path
   `outputs/artifacts/surrogates/thermo_symbolic/thermo_symbolic_f1.npz`
   with a deprecation warning.

Run entrypoint with explicit artifact override:

```bash
python -m larrak2.cli.run explore-exploit \
  --thermo-symbolic-artifact-path outputs/artifacts/surrogates/thermo_symbolic_f<fidelity>/thermo_symbolic_f<fidelity>.npz
```

Equivalent flags are available on `orchestrate` and `larrak2.cli.refine_pareto`.
