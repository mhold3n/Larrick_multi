# Component Map: Requirements vs Cursor Limits

| Workflow Requirement | Plugin Component | Implementation | Platform Limit Handling |
|---|---|---|---|
| Enforce branch naming and domain policy | `rules/workflow-ownership.mdc` | Always-on policy guidance | Guidance only; does not override Git internals |
| Keep behavior aligned with Cursor worktree model | `rules/worktree-platform-limits.mdc` | Always-on limits note | Explicitly acknowledges Cursor-managed worktrees and Apply |
| Reusable operating procedure for agents | `skills/worktree-runbook/SKILL.md` | Standard manual + parallel-agent runbook | Uses CLI checks where LSP is unavailable |
| Block risky Git operations from agent shell | `hooks/hooks.json` + `scripts/enforce_git_policy.sh` | Deny direct pushes to protected branches and invalid `-b` names | Hook executes before shell, fail-closed only on policy matches |
| Inject startup context each session | `scripts/session_hint.sh` | Adds concise policy context | Non-invasive; does not mutate project bootstrap |
| Keep project setup deterministic | repo `.cursor/worktrees.json` | Venv + deps + smoke tests in worktree setup | Kept outside plugin to avoid project-setup duplication |
