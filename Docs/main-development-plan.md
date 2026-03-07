# Main Development Plan: Hard-First Compute/Optimize/Solve (Canonical)

Status: active  
Last updated: 2026-03-03  
Scope: backend-only (`src/larrak2`), no GUI/dashboard migration

This is the single source of truth for development priorities.  
Any previous plan doc is superseded by this file.

## 0. Consolidation Record

The following superseded plan documents were intentionally removed and folded into this canonical plan:

1. `Docs/deep-research-report.md`
2. `Docs/deep-research-report 2.md`
3. `Docs/deep-research-report 3.md`
4. `Docs/dress-rehearsal-process.md`
5. `Docs/merge_wave1_manifest.md`
6. `Docs/freepiston/drafts/implementation_outline_missing_features.md`
7. `Docs/freepiston/drafts/op_engine_hybrid_architecture.md`
8. `Docs/freepiston/drafts/wall_function_full_goals_plan.md`
9. `Docs/freepiston/drafts/wall_function_full_goals_summary.md`
10. `Docs/freepiston/drafts/wall_function_gap_analysis.md`
11. `Docs/freepiston/drafts/wall_function_implementation_timeline.md`
12. `Docs/freepiston/drafts/project_status/summary.md`

## 1. Non-Negotiable Program Rules

1. A sprint does not pass if it only works with toy physics (`fidelity=0`) or placeholder constants.
2. Full decision dimensionality must be preserved end-to-end (`N_TOTAL` unchanged). Refinement may optimize slices, but vectors stay full width.
3. All terminal artifacts must be written under `outputs/` and categorized by purpose. No runtime artifact writes to deprecated `models/` or `src/`.
4. Downselect/release gates require explicit material/tribology/lifetime checks, not proxy-only scoring.
5. CI pass is necessary but not sufficient; physical validity gates below must pass.

## 2. Target Pipeline (Explore -> Exploit -> Lifetime)

1. Stage A `Explore` (pymoo):
   Full-vector multi-objective search over motion law + geometry + real-world controls.
2. Stage B `Exploit` (CasADi/Ipopt):
   Candidate-level local refinement on sensitivity-ranked active subsets with frozen baseline variables and trust region.
3. Stage C `Lifetime/Material`:
   Phase-resolved tribology + life damage + material route iteration on refined candidates.
4. Stage D `Truth Loop`:
   OpenFOAM/CalculiX truth runs + surrogate retraining + re-ranking.

## 3. Current Code Snapshot: What Exists vs Hard Gaps

| Area | Present in codebase | Hard gap to close |
| --- | --- | --- |
| Two-stage plumbing | `src/larrak2/pipelines/explore_exploit.py`, `src/larrak2/optimization/candidate_store.py` | Needs strict fidelity gating and stronger high-fidelity objective parity |
| CasADi refinement | `src/larrak2/adapters/casadi_refine.py`, `src/larrak2/optimization/slicing/*` | Nonlinear symbolic slice NLP is integrated with strict thermo symbolic overlay preflight and diagnostics propagation; remaining gap is anchor-governance policy research |
| Thermo core | `src/larrak2/thermo/motionlaw.py`, `src/larrak2/thermo/two_zone.py`, `src/larrak2/thermo/combustion.py`, `src/larrak2/thermo/scavenging.py` | Equation-first two-zone path is integrated; thermo symbolic bridge hardening and balanced per-target quality gates are integrated; remaining gap is anchor governance/provenance policy finalization |
| Tribology/material | `src/larrak2/cem/*`, `src/larrak2/realworld/*` | ISO/FZG data contracts and method-aware scuff evaluation are integrated; remaining gap is lifetime-depth calibration and uncertainty calibration |
| Lifetime model | `src/larrak2/realworld/life_damage.py` | Simplified Miner proxy needs calibrated SN/stress models and route-specific validation |
| Surrogate workflows | `src/larrak2/training/workflows.py`, `src/larrak2/surrogate/*` | Need uncertainty-aware quality gates and stronger dataset contracts per operating regime |
| Orchestration backend | `src/larrak2/orchestration/*`, CLI `orchestrate` | Auto truth dispatch is integrated; remaining gap is confidence-policy tuning and truth-budget calibration |

### 3.1 Thermo Remaining Gaps (Post Two-Zone)

1. Anchor governance contract:
   strict fidelity-2 anchor requirements remain under policy review before lock-in.
2. Anchor reproducibility:
   baseline anchor manifest generation/provenance tooling remains open.
3. Symbolic bridge hardening:
   complete for strict/warn/off runtime modes, preflight compatibility checks, and per-target balanced quality gating.
4. Test isolation:
   thermo-symbolic and cross-stack hardening tests are isolated from unrelated CEM/tribology volatility.

## 4. Hard Missing Components (By Module Class)

### 4.1 Equations and Physics Models

1. Thermodynamics and combustion:
   Replace placeholder combustion/scavenging logic with physically coupled cycle modeling (mass, energy, and gas exchange closure).
2. Gear/contact mechanics:
   Preserve phase-resolved contact stress, sliding, entrainment, and load paths with robust handling of misalignment and thermal distortion.
3. Tribology:
   Replace placeholder EHL/scuff/micropitting constants with dataset-backed correlations over temperature, speed, load, and finish/coating regimes.
4. Lifetime:
   Upgrade from simplified stress-ratio damage proxy to calibrated life model tied to route/material process families and lubrication state.

### 4.2 Data Contracts (Non-Optional)

1. `data/cem/*.csv` must be treated as required scientific inputs, not optional decoration.
2. Each table needs explicit schema, units, provenance, and minimum operating-envelope coverage.
3. Missing/empty critical tables must fail strict modes (CI downselect/release), not silently fallback.
4. OpenFOAM/CalculiX training corpora need condition-space coverage metadata and versioned manifests.

### 4.3 Surrogate Requirements

1. Surrogates must ship with:
   train/val/test metrics by condition slice, OOD diagnostics, and artifact version metadata.
2. Promotion to production mode requires bounded error against truth solvers in target operating windows.
3. Surrogate outputs used in optimization must expose confidence/uncertainty for orchestration decisions.

### 4.4 Optimization/Solver Requirements

1. Keep full-vector representation always; only active subset is decision variable in local solve.
2. CasADi path must solve true slice NLP (not linearized surrogate of the NLP) with explicit constraints and trust region.
3. Refinement metadata must always include active/frozen sets, backend used, solver status, and fallback reason.
4. SciPy fallback remains for robustness, but cannot be the default pass path for high-fidelity milestones.

### 4.5 Artifact and Reproducibility Requirements

1. Runtime artifacts live under `outputs/` only:
   `outputs/artifacts/surrogates/*`, `outputs/explore_exploit/*`, `outputs/orchestration/*`, etc.
2. Any generated cache/model under `src/` is a bug and must be relocated.
3. Every workflow run must emit manifest + config + key metrics for replay.

## 5. Hard-First Execution Plan

### Phase 0 (Now): Documentation and Policy Lock

1. Canonicalize to this doc.
2. Remove stale/completed/superseded plan docs.
3. Point README references to this file.

Exit criteria:

1. No competing roadmap docs remain in active `Docs/` planning surface.
2. Team uses one gate definition for sprint completion.

### Phase 1: Data and Constraint Integrity

1. Expand CEM datasets from minimal placeholders to envelope-complete tables.
2. Enable strict data mode in CI for downselect/release test tracks.
3. Remove silent fallback reads for critical material/tribology terms.

Exit criteria:

1. Missing critical dataset columns/keys fail fast.
2. Material-route and temperature-curve coverage spans intended operating envelope.

### Phase 2: True CasADi Exploit Solve

1. Replace linearized slice NLP approximation with true nonlinear slice refinement.
2. Keep sensitivity-ranked active-set selection with group-floor enforcement.
3. Enforce trust-region bounded updates relative to Pareto seed.

Exit criteria:

1. At least one CasADi/Ipopt-success refinement in CI/dev smoke with full-width vector output.
2. Fallback path is tested and annotated (`backend_used="scipy_fallback"`).

### Phase 3: Physics Hardening for Lifetime Decisions

1. Replace placeholder thermo/combustion/scavenging models used for downselect metrics.
2. Calibrate tribology and lifetime models against route-specific data and truth checks.
3. Promote material/lifetime constraints to hard for downselect.

Exit criteria:

1. No placeholder constants are used in downselect path.
2. Lifetime/material metrics are traceable to concrete datasets and equations.

### Phase 4: Truth-in-the-Loop Surrogate Governance

1. Run controlled OpenFOAM/CalculiX truth campaigns for targeted regions.
2. Retrain surrogates with uncertainty tracking and versioned manifests.
3. Tie orchestration decisions to confidence + budget policies.

Exit criteria:

1. Surrogates meet quality thresholds against truth data.
2. Orchestration records provenance of decisions and truth evaluations.

### Phase 5: Final Candidate Lifetime/Material Iteration

1. Multi-route material/coating iteration on refined candidates.
2. Produce final candidate package:
   geometry/motion settings, lifetime risk envelope, material requirements, and evidence artifacts.

Exit criteria:

1. Final candidate selected using hardened physics and validated data only.
2. No pass conditions rely on toy proxies.

## 6. Gate Framework (Anti-Toy-Physics)

A sprint marked complete must satisfy all relevant gates:

1. Fidelity gate:
   no release/downselect approval from fidelity-0-only runs.
2. Data gate:
   required CEM datasets present, schema-valid, and non-empty for used routes.
3. Solver gate:
   CasADi/Ipopt refinement success demonstrated on target scenarios; fallback path explicit.
4. Physics gate:
   placeholder-only modules not used for final decision metrics.
5. Artifact gate:
   manifests + outputs written to `outputs/` with reproducible run config.

## 7. Integrated Dress-Rehearsal Contract

Dress rehearsal remains the pre-analysis gate and is now governed by this doc:

1. Train surrogates pre-job (`train-surrogates`) using provided or DOE-generated datasets.
2. Verify required surrogate artifacts exist for requested fidelity/modes.
3. Run unit tests unless explicitly skipped for smoke-only runs.
4. Run coarse optimization sweep and require minimum Pareto output.
5. Run CEM validation on promoted candidates and enforce feasibility minimum.
6. Use `dress_rehearsal_manifest.json` `ready_for_quality_analysis` as final gate signal.

## 8. Completed Baseline (Do Not Re-Plan)

The following migration foundation is already integrated and should not be treated as open planning work:

1. Legacy IPOPT/CasADi stack port into `src/larrak2/optimization/*`.
2. Surrogate HiFi core port into `src/larrak2/surrogate/hifi/*` and `src/larrak2/training/*`.
3. Backend orchestration package and CLI `orchestrate` integration.
4. Explore/exploit candidate-store and slice-refinement plumbing.

Future work should extend these modules, not reintroduce parallel duplicate frameworks.

## 9. Readiness Status Snapshot (2026-03-03)

This section links the latest local readiness diagnosis artifacts for
Explore -> Exploit -> Lifetime (A->C) and F2 probe status.

### 9.1 Evidence Artifacts

1. Gap ledger:
   `outputs/readiness/gap_ledger.json`
2. Artifact/data contract audit (pre-remediation):
   `outputs/readiness/artifact_contract_audit_pre.json`
3. Artifact/data contract audit (current):
   `outputs/readiness/artifact_contract_audit.json`
4. Runtime probe matrix:
   `outputs/readiness/runtime_probe_results.json`
5. Single-candidate lifetime extraction:
   `outputs/readiness/single_candidate_lifetime_report.json`
6. F2 blocker register:
   `outputs/readiness/f2_blockers.json`
7. Human-readable readiness summary:
   `outputs/readiness/pipeline_readiness_summary.md`

### 9.2 Snapshot Outcome

1. A->C strict artifact/data prerequisites are green
   (`a_to_c_hard_failed == 0` in `artifact_contract_audit.json`).
2. CasADi is available in the active probe interpreter
   (`/opt/miniconda3/bin/python`, `casadi==3.7.0`); remaining strict A->C/F2
   blockers are tracked as runtime/asset readiness issues in the linked
   readiness artifacts.
3. F2 probe blockers are captured and categorized in
   `outputs/readiness/f2_blockers.json`.

## 10. Architecture Contract-First Status (2026-03-03)

This appendix tracks architecture-level orchestration and interface-contract
readiness for Pareto -> Explore/Exploit -> Single-candidate Lifetime.

### 10.1 Evidence Artifacts

1. Architecture gap ledger:
   `outputs/readiness/architecture/architecture_gap_ledger.json`
2. Edge coverage report:
   `outputs/readiness/architecture/edge_coverage_report.json`
3. Required key parity report (fidelity 0 vs 2):
   `outputs/readiness/architecture/key_parity_report_f0_vs_f2.json`
4. Workflow probe matrix:
   `outputs/readiness/architecture/workflow_probe_results.json`
5. Human-readable architecture readiness summary:
   `outputs/readiness/architecture/pipeline_arch_readiness_summary.md`

### 10.2 Scope Notes

1. These artifacts prioritize orchestration wiring, edge contracts, and fidelity
   routing policy over module calibration details.
2. Fidelity 0 is treated as a full contract observability mode: all required
   edges are expected with placeholder engine mode and required key coverage.
3. Fidelity 2 probe failures remain tracked as blockers but are not treated as
   evidence of contract-shape closure unless key and routing requirements pass.

## 11. Principles-First Explore-Exploit Status (2026-03-03)

This appendix records the principles-first frontier expansion for
`explore-exploit`.

### 11.1 Design Note and Inputs

1. Design and behavior note:
   `Docs/explore-exploit-principles-mode.md`
2. Principles profile:
   `data/optimization/principles_frontier_profile_v1.json`
3. Canonical source mapping:
   `Docs/sources` remains the source-of-truth bundle and legacy ISO paths are
   mapped via `Docs/items/legacy_migration_map.csv`.

### 11.2 Readiness Artifact Links

1. `outputs/readiness/architecture/architecture_gap_ledger.json`
2. `outputs/readiness/architecture/edge_coverage_report.json`
3. `outputs/readiness/architecture/key_parity_report_f0_vs_f2.json`
4. `outputs/readiness/architecture/workflow_probe_results.json`
5. `outputs/readiness/architecture/pipeline_arch_readiness_summary.md`

## 12. Multi-Objective Production Hardening Status (2026-03-04)

This appendix tracks strict-production hardening for multi-objective
optimization execution (no surrogate training scope).

### 12.1 Status Summary

1. Strict production profile is default across Pareto, explore-exploit, and
   orchestration entry points (`strict_prod`).
2. `allow_nonproduction_paths` is explicit and auditable; non-production runs
   are tagged with `nonproduction_overrides` and are not production-gate pass.
3. NSGA-III default execution is bounded (`partitions=4`,
   `nsga3_max_ref_dirs=192`) with deterministic partition fallback and emitted
   effective-capacity metadata.
4. Candidate-level evaluator exceptions are isolated with deterministic
   penalties, counted in diagnostics, and enforced as strict failures via
   production gate when nonzero.
5. Candidate ranking/downselect scoring is normalized using robust objective
   scales (`p95 - p05`, floored at `1e-9`) to remove objective-unit dominance.
6. Deterministic archive ordering and artifact hash metadata are emitted for
   Pareto outputs.

### 12.2 Operator Reference

1. Production runbook:
   `Docs/multi-objective-production-runbook.md`
2. Shared gate implementation:
   `src/larrak2/optimization/production_gate.py`
3. Core gate diagnostics keys:
   `production_profile`, `production_gate_pass`, `production_gate_failures`,
   `fallback_paths_used`, `nonproduction_overrides`, `n_eval_errors`,
   `algorithm_used`, `fidelity`, `constraint_phase`

### 12.3 CasADi Artifact Matrix (Strict Production)

This matrix defines canonical artifact expectations for CasADi entrypoints.

| Requested fidelity | Stack artifact (canonical) | Thermo symbolic artifact (canonical) | Strict behavior |
| --- | --- | --- | --- |
| 0 | `outputs/artifacts/surrogates/stack_f0/stack_f0_surrogate.npz` | `outputs/artifacts/surrogates/thermo_symbolic_f0/thermo_symbolic_f0.npz` | Missing/mismatch fails fast; no cross-fidelity fallback |
| 1 | `outputs/artifacts/surrogates/stack_f1/stack_f1_surrogate.npz` | `outputs/artifacts/surrogates/thermo_symbolic_f1/thermo_symbolic_f1.npz` | Missing/mismatch fails fast; legacy thermo path may auto-resolve with warning |
| 2 | `outputs/artifacts/surrogates/stack_f2/stack_f2_surrogate.npz` | `outputs/artifacts/surrogates/thermo_symbolic_f2/thermo_symbolic_f2.npz` | Missing/mismatch fails fast; no fallback |

Readiness checks:

1. Every CasADi workflow run (`explore-exploit`, `orchestrate`, `refine_pareto`)
   resolves artifacts against requested fidelity before solve execution.
2. Missing artifacts include remediation commands:
   - `python -m larrak2.cli.run train-stack-surrogate --fidelity <f>`
   - `python -m larrak2.cli.run train-thermo-symbolic --fidelity <f>`
3. CI includes a dedicated CasADi lane (`.[dev,casadi]`) executing symbolic
   stack/slice tests and strict no-fallback backend behavior checks.
4. Strict orchestration (`strict_data=true`) additionally enforces readiness
   evidence artifacts:
   - `outputs/readiness/pipeline_readiness_summary.md`
   - `outputs/readiness/f2_blockers.json`
   - `outputs/readiness/artifact_contract_audit.json`
