# Workflow Branch Policy

Status: active  
Last updated: 2026-03-24

## Branches

- `main` is the only integration branch and the only legal code-sharing path between workflow branches.
- Long-lived workflow branches:
  - `dev/simulation`
  - `dev/training`
  - `dev/optimization`
  - `dev/analysis`
  - `dev/cem-orchestration`
- Short-lived task branches should be cut from one workflow branch at a time and use the form `codex/<workflow>/<topic>`.

## Promotion Rules

- Code changes land on the owning workflow branch first.
- Code never moves directly from one workflow branch to another.
- To share code across workflows, promote the owning branch change to `main`, then pull or rebase the dependent workflow branches from `main`.
- Artifacts may be shared across branches for validation and comparison, but artifacts do not bypass the code-promotion rule above.

## Ownership

- `dev/simulation`: `simulation_validation`, OpenFOAM pipeline code, solver adapters, templates, mechanism/runtime-table packaging.
- `dev/training`: `training`, surrogate training/inference packaging, training CLIs.
- `dev/optimization`: optimization, promote/explore-exploit/orchestration CLI flow.
- `dev/analysis`: analysis, readiness, telemetry summaries, reporting helpers.
- `dev/cem-orchestration`: CEM, real-world models, orchestration backends/adapters, production-gate/report assembly.

## Compatibility

- Simulation-produced datasets are shared through versioned manifest bundles under the architecture contract layer.
- Training and replay consumers must support the current simulation API version and the immediately previous version.
- CI on `main` is the compatibility gate that catches contract drift between producer and consumer workflows.
