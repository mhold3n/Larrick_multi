# Agent Context — Larrick Multi Repository

This file is read by OpenAI Codex, Google Gemini, and other AI coding agents.
Before making changes, read this file in full.

## Repository Purpose

Larrick Multi is an engine simulation and surrogate-model optimization platform.
The codebase still groups work by engineering domain, but branch governance is
no longer part of the repo workflow. The domain map below is kept as
documentation only and is intended to help a future split into separate
projects.

## Workflow Model

- `main` is the only documented repo workflow branch.
- Direct commits to `main` are the default working model for this repository.
- Temporary Git branches may still exist, but the repo does not define,
  validate, or automate any branch naming convention.
- Do not assume any workflow-routing, promotion, or branch-ownership automation
  exists.

## Domain Ownership (Documentation Only)

| Domain | Primary Paths |
|---|---|
| Simulation pipeline | `src/larrak2/simulation_validation/`, `src/larrak2/pipelines/openfoam.py`, `src/larrak2/adapters/openfoam.py`, `src/larrak2/adapters/docker_openfoam.py`, `openfoam_custom_solvers/`, `openfoam_templates/`, `mechanisms/openfoam/` |
| Surrogate training | `src/larrak2/training/`, `src/larrak2/surrogate/`, `src/larrak2/cli/train.py` |
| Optimization and orchestration | `src/larrak2/optimization/`, `src/larrak2/promote/`, `src/larrak2/cli/run.py`, `src/larrak2/cli/run_workflows.py`, `src/larrak2/orchestration/simulation_inputs.py` |
| Analysis and telemetry | `src/larrak2/analysis/` |
| CEM and real-world backends | `src/larrak2/cem/`, `src/larrak2/realworld/`, `src/larrak2/orchestration/` except `src/larrak2/orchestration/simulation_inputs.py` |
| Shared architecture contracts | `src/larrak2/architecture/` |

Treat this mapping as guidance for review and future extraction work, not as an
enforced branch or PR policy.

## Working Expectations

- Prefer small, reviewable changes even when working directly on `main`.
- If a change spans multiple domains, call that out clearly in the summary and
  keep interfaces stable where possible.
- Do not force-push protected branches.
- Do not add `contents: write` permission to CI workflows.

## CI Contract

The repository uses a single `CI` workflow for pushes to `main` and pull
requests targeting `main`. The fast lane runs:

- `ruff format --check` and `ruff check`
- `mypy` on the maintained typed entrypoints
- `pytest -q`

Self-hosted heavy lanes may still exist in the future, but they are not tied to
branch-specific workflow wrappers anymore.

## Simulation Dataset Contract

Simulation outputs are shared through versioned manifest bundles
(`simulation_dataset_bundle.json`). Training and replay consumers must support
the current simulation API version and the immediately previous version.
Use `load_simulation_dataset_bundle()` from
`src/larrak2/architecture/workflow_contracts.py` to read these bundles rather
than parsing the JSON directly.
