#!/bin/bash
set -euo pipefail

cat >/dev/null
printf '{"continue":true,"additionalContext":"Worktree policy: use codex/<workflow>/<short-topic>; keep project setup in .cursor/worktrees.json; run CLI checks before PR."}\n'
