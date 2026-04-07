---
name: worktree-runbook
description: Run the standard Larrick worktree workflow in Cursor for manual and parallel-agent tasks.
---

# Worktree Runbook

## When to use

- Starting domain-scoped work in a dedicated worktree.
- Running a Cursor Parallel Agent and applying results safely.
- Preparing a worktree for CI-compatible validation checks.

## Workflow

1. Confirm owning workflow and branch naming (`codex/<workflow>/<short-topic>`).
2. Use one Cursor window per manual `git worktree` checkout.
3. For Cursor Parallel Agents, rely on Cursor-managed worktrees and use Apply to merge back.
4. Run CLI checks before PR:
   - `ruff format --check .`
   - `ruff check .`
   - `mypy <workflow-scoped-paths>`
   - `pytest -q`
5. Keep project bootstrap logic in `.cursor/worktrees.json`; do not duplicate dependency setup in plugin assets.
