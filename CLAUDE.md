# Claude Agent Context — Larrick Multi-Workflow Repository

This file is read by Anthropic Claude Code and claude.ai agents.
Before making any changes, read this file in full.

## Repository Purpose

Larrick Multi is an engine simulation and surrogate-model optimization platform.
It is organized into five long-lived workflow branches, each owned by a distinct
engineering domain. Code never moves directly between workflow branches — all
cross-workflow sharing goes through `main`.

## Branch Ownership

| Branch | Owner Domain | Owned Paths |
|---|---|---|
| `dev/simulation` | Simulation pipeline | `src/larrak2/simulation_validation/`, `src/larrak2/pipelines/openfoam.py`, `src/larrak2/adapters/openfoam.py`, `src/larrak2/adapters/docker_openfoam.py`, `openfoam_custom_solvers/`, `openfoam_templates/`, `mechanisms/openfoam/` |
| `dev/training` | Surrogate training | `src/larrak2/training/`, `src/larrak2/surrogate/`, `src/larrak2/cli/train.py` |
| `dev/optimization` | Optimization & orchestration | `src/larrak2/optimization/`, `src/larrak2/promote/`, `src/larrak2/cli/run.py`, `src/larrak2/cli/run_workflows.py`, `src/larrak2/orchestration/simulation_inputs.py` |
| `dev/analysis` | Analysis & telemetry | `src/larrak2/analysis/` |
| `dev/cem-orchestration` | CEM & real-world backends | `src/larrak2/cem/`, `src/larrak2/realworld/`, `src/larrak2/orchestration/` (except `simulation_inputs.py`) |

Shared contract layer (read by all, modified via `main` PRs only):
- `src/larrak2/architecture/`

## Task Branch Naming

When starting a task, create a branch from the owning workflow branch:

```bash
git checkout dev/<workflow>
git checkout -b codex/<workflow>/<short-topic>
```

Examples: `codex/simulation/fix-doe-paths`, `codex/training/add-manifest-schema`

## Same-Machine Parallelism

For same-machine parallel work, use one `git worktree` per active task and
never reuse a worktree for a different task. The supported bootstrap flow is:

```bash
scripts/start_parallel_task.sh --mode worktree <workflow> <topic> [issue-number]
```

That script creates a sibling worktree on `codex/<workflow>/<topic>` from the
owning `dev/<workflow>` branch, initializes task-local runtime state, writes
`.task-runtime/task.json` for Codex MCP handoff, and keeps parallel local tasks
from colliding with each other. Codex should use that task JSON together with
`scripts/plan_github_concierge.py` and GitHub MCP tools to manage task PRs,
promotion PRs, and audit flows.

## Cloud Thread Routing

For cloud or already-isolated current workspaces, treat the local checkout as a
clean control repo on `main` and let the cloud workspace own the active
`codex/*` task branch. On the first substantive user prompt in a new cloud
thread, classify the request into `simulation`, `training`, `optimization`,
`analysis`, `cem-orchestration`, or `main` before doing any edits. The
supported bootstrap flow is a single command:

```bash
python3 scripts/route_current_thread.py --prompt "<initial-prompt>" --thread-id "$CODEX_THREAD_ID" --thread-name "<thread-name>"
```

That command uses `scripts/plan_github_concierge.py route-thread` to classify
the prompt and then, when confidence is high, runs
`scripts/start_parallel_task.sh --mode current` to create
`codex/<workflow>/<thread-derived-topic>-<threadid8>` from the owning
`dev/<workflow>` branch in the current cloud workspace. If routing confidence is
low or the request looks cross-workflow, ask the user which workflow branch
should own the task instead of guessing.

Do not synthesize workflow branches manually with raw `git checkout -b
dev/<workflow>` or `git checkout -b codex/<workflow>/...` in cloud routing
mode. Use the router/bootstrap scripts so `.task-runtime/task.json`, routing
metadata, and lifecycle bookkeeping are created consistently.

Explicit workflow markers override the classifier. Support both:
- token: `workflow:simulation`
- phrase: `working on simulation development`

If the bootstrap reports `PR duplicate check: mcp-required`, use GitHub MCP
search before opening or syncing the task PR. Cloud agents should not depend on
local `gh` auth to route or branch correctly.

Cloud task PRs are the durable lifecycle record. Keep the structured PR metadata
block current, treat `archive_eligible` as the cross-thread replacement for true
UI archival, delete merged `codex/*` branches after merge, and report
archive-eligible merged/blocked tasks in the weekly audit inbox.

## Promotion Rules

1. Land code on the owning `dev/<workflow>` branch first (via PR from `codex/*` task branch).
2. Promote to `main` via PR from `dev/<workflow>`.
3. Other workflow branches pull from `main` — never directly from each other.
4. Artifacts (bundles, manifests) may be shared across branches for validation, but artifact sharing never substitutes for the code-promotion rule.

## What NOT to Touch

- Do **not** push directly to `main` or any `dev/*` branch — PRs are required.
- Do **not** edit files outside the owning workflow's paths unless explicitly instructed.
- Do **not** force-push to any protected branch.
- Do **not** add `contents: write` permission to any CI workflow.
- Do **not** merge one `dev/*` branch directly into another `dev/*` branch.

## CI Contract

Every PR must pass `Fast Checks (<workflow-name>)` before merge. The fast lane runs:
- `ruff format --check` + `ruff check` (lint)
- `mypy` on workflow-scoped paths
- `pytest -q` (full test suite or targeted subset per workflow)

Self-hosted heavy lanes (`heavy-self-hosted` jobs) are scaffolded but inactive
until hardware is attached. They are gated on `vars.LARRAK_ENABLE_SELF_HOSTED_*`
repo variables and run only on `workflow_dispatch` with `run_heavy: true`.

## Simulation Dataset Contract

Simulation outputs are shared through versioned manifest bundles
(`simulation_dataset_bundle.json`). Training and replay consumers must support
the current simulation API version and the immediately previous version.
Use `load_simulation_dataset_bundle()` from `src/larrak2/architecture/workflow_contracts.py`
to read these bundles — do not parse the JSON directly.
