# Multi-Objective Production Runbook

Status: active  
Last updated: 2026-03-04

## Scope

This runbook covers production execution of the multi-objective optimization stack
without surrogate retraining. It applies to:

1. `python -m larrak2.cli.run_pareto`
2. `python -m larrak2.cli.run explore-exploit`
3. `python -m larrak2.cli.run orchestrate`

## Strict Defaults

1. Production profile is `strict_prod`.
2. `run_pareto` defaults:
   `fidelity=2`, `constraint_phase=downselect`, `algorithm=nsga3`,
   `partitions=4`, `nsga3_max_ref_dirs=192`.
3. `orchestrate` defaults:
   `fidelity=2`, `constraint_phase=downselect`.
4. Thermo symbolic mode remains strict by default.

## CasADi Artifact Contract

When `backend=casadi`, artifact resolution is fidelity-specific and strict:

1. Stack surrogate canonical path:
   `outputs/artifacts/surrogates/stack_f{fidelity}/stack_f{fidelity}_surrogate.npz`
2. Thermo symbolic canonical path:
   `outputs/artifacts/surrogates/thermo_symbolic_f{fidelity}/thermo_symbolic_f{fidelity}.npz`
3. No cross-fidelity fallback is allowed in strict production paths.
4. Missing artifacts fail fast with remediation commands:
   - `python -m larrak2.cli.run train-stack-surrogate --fidelity <f>`
   - `python -m larrak2.cli.run train-thermo-symbolic --fidelity <f>`

Legacy compatibility note:

1. `fidelity=1` thermo symbolic may temporarily auto-resolve from legacy path
   `outputs/artifacts/surrogates/thermo_symbolic/thermo_symbolic_f1.npz`
   with a deprecation warning.

## Production Gate Contract

All production manifests/summaries emit:

1. `production_profile`
2. `production_gate_pass`
3. `production_gate_failures`
4. `fallback_paths_used`
5. `nonproduction_overrides`
6. `n_eval_errors`
7. `algorithm_used`
8. `fidelity`
9. `constraint_phase`

These keys are emitted both inside `production_gate` and as top-level fields
for uniform downstream parsing across Pareto, explore/exploit, and
orchestration manifests.

Balanced strict thresholds:

1. `n_pareto >= max(8, ceil(0.12 * effective_pop))`
2. `feasible_fraction >= 0.20`
3. `n_eval_errors == 0`
4. Explore/exploit must produce a hard-feasible winner
5. Principles gate must pass and cannot use `placeholder_frontier` basis
6. Orchestration `release_ready` must be true and heuristic fallback must not be used

## Non-Production Override

`--allow-nonproduction-paths` enables legacy fallback paths for diagnostics/dev runs.
When enabled:

1. execution may continue,
2. `nonproduction_overrides` is populated,
3. production gate is forced non-pass, and
4. release-readiness must be treated as non-release.

## Failure Remediation

If strict production gate fails:

1. Inspect `production_gate_failures` in:
   `summary.json`, `explore_exploit_manifest.json`, or `orchestrate_manifest.json`.
2. Resolve the specific failure:
   - low Pareto size/feasibility: increase budget (`--pop`, `--gen`) and verify constraints,
   - eval errors: inspect evaluator exceptions in run logs/signatures,
   - CasADi artifact mismatch/missing: train/publish fidelity-matching stack + thermo symbolic artifacts,
   - frontier gate failures: fix principles profile/anchors and hard-feasibility coverage,
   - release readiness issues: run with `constraint_phase=downselect` and strict data.
3. Re-run without `--allow-nonproduction-paths` for release candidates.
