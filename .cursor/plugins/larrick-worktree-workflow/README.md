# Larrick Worktree Workflow Plugin

This plugin standardizes the Larrick domain-separated worktree workflow in Cursor.

## Scope

- Provides workflow policy as Cursor `rules`.
- Provides reusable runbook guidance as a Cursor `skill`.
- Provides optional shell guardrails via Cursor `hooks`.
- Delegates policy/bootstrapping to repository-owned portable stubs in `scripts/worktree_agent/`.

## Why this boundary exists

Cursor Parallel Agents use Cursor-managed worktrees and merge through Apply.
This plugin does not attempt to replace Cursor internals; it standardizes behavior around them.
To keep this workflow portable, Cursor-specific elements call shared scripts in
`scripts/worktree_agent/` that can be reused by open-source replacements.

## Local prototype usage

1. Symlink plugin into local plugin directory:

```bash
mkdir -p ~/.cursor/plugins/local
ln -s /Users/maxholden/GitHub/Larrick_multi/.cursor/plugins/larrick-worktree-workflow ~/.cursor/plugins/local/larrick-worktree-workflow
```

2. Reload Cursor window.
3. Ensure hooks are enabled by referencing this plugin hooks path in plugin install context.
4. Validate on:
   - one manual domain worktree
   - one Cursor Parallel Agent run with Apply

## Team Marketplace readiness

- Manifest path: `.cursor-plugin/plugin.json`
- Versioning: semantic (`0.1.0` currently)
- Publish by mirroring this plugin folder into a dedicated plugin repository or marketplace collection.
- Rollout recommendation: optional-first, then required for selected groups.

## Validation gates

- Branch policy:
  - branch created as `codex/<workflow>/<short-topic>`
  - no direct pushes to `main` or `dev/*`
- Worktree setup:
  - `.cursor/worktrees.json` calls `scripts/worktree_agent/worktree_bootstrap.sh`
- Pre-PR checks:
  - `ruff format --check .`
  - `ruff check .`
  - `mypy` on workflow-owned paths
  - `pytest -q` (or approved subset)
- CI compatibility:
  - PR fast checks match expected workflow wrapper behavior on dedicated hardware pipeline.
