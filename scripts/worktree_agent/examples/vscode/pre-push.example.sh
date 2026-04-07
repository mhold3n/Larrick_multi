#!/bin/bash
set -euo pipefail

# Example git pre-push hook for non-Cursor environments.
# Copy to .githooks/pre-push and configure:
#   git config core.hooksPath .githooks

branch="$(git rev-parse --abbrev-ref HEAD)"
cmd="git push origin ${branch}"

printf '%s' "{\"command\":\"${cmd}\"}" | bash scripts/worktree_agent/branch_policy_guard.sh
