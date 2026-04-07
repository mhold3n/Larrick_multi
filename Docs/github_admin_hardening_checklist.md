# GitHub Admin Hardening Checklist

This checklist documents the one-time repository settings assumed by the local
parallel task concierge and GitHub MCP workflow.

## Main Branch

- Require pull requests before merge.
- Require the `Branch Ownership` check.
- Require the `Fast Checks (main)` check.
- Require the `Repo Layout` check.
- Require conversation resolution before merge.
- Require at least one human approval.
- Enable merge queue.
- Block force-pushes.

## Workflow Branches

Apply the following settings to each `dev/*` branch:

- Require pull requests before merge.
- Require the `Branch Ownership` check.
- Require the matching `Fast Checks (<workflow>)` check.
- Require conversation resolution before merge.
- Block force-pushes.
- Allow zero required human approvals so green `codex/* -> dev/*` task PRs can auto-merge.

## Repository Settings

- Enable automatic deletion of head branches after merge.
- Keep `main` and every `dev/*` branch protected.
- Do not allow direct code movement between `dev/*` branches.
- Preserve read-only CI permissions; do not add `contents: write` to workflow files.
