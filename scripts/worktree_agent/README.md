# Worktree-Agent Portability Stubs

These scripts are the platform-neutral workflow contract for worktree-isolated
agent development across the five Larrick workflow branches:

- `dev/simulation`
- `dev/training`
- `dev/optimization`
- `dev/analysis`
- `dev/cem-orchestration`

Use these stubs as the stable interface, regardless of AI/editor platform
(Cursor, a VS Code fork, or another agent host).

## Stub responsibilities

- `worktree_bootstrap.sh`: project bootstrap inside a worktree.
- `branch_policy_guard.sh`: branch naming/protected-branch policy checks.
- `pre_pr_checks.sh`: reproducible pre-PR CLI validation gate.
- `platform_adapter.template.json`: mapping template for any editor/agent host.

## Adapter pattern

Platform-specific integrations should call these scripts instead of duplicating
policy/bootstrapping logic:

- Cursor: `.cursor/worktrees.json` + plugin hooks call into this directory.
- Other platforms: command palette tasks, extension hooks, or agent runtimes
  call into the same scripts.

This keeps workflow behavior consistent across toolchains.

## Minimal non-Cursor adapter example (VS Code forks)

Reference example files:

- `scripts/worktree_agent/examples/vscode/adapter.vscode.json`
- `scripts/worktree_agent/examples/vscode/tasks.json`
- `scripts/worktree_agent/examples/vscode/pre-push.example.sh`

Suggested setup:

1. Copy `scripts/worktree_agent/examples/vscode/tasks.json` to `.vscode/tasks.json`.
2. Optionally copy `pre-push.example.sh` to `.githooks/pre-push`, make it executable,
   and configure `git config core.hooksPath .githooks`.
3. Run tasks from command palette:
   - `Larrick: Worktree Bootstrap`
   - `Larrick: Pre-PR Checks`
   - `Larrick: Branch Policy Check (Current Branch)`
