#!/bin/bash
set -euo pipefail

# Portable branch-policy guard stub for AI/editor integrations.
# Input: JSON via stdin with optional {"command":"..."} for hook-style checks.
# Exit codes:
# - 0 allow
# - 2 deny (policy violation)

payload="$(cat || true)"
cmd="$(python3 -c 'import json,sys; print((json.load(sys.stdin).get("command","")).strip())' <<<"${payload}" 2>/dev/null || true)"

if [[ -z "${cmd}" ]]; then
  exit 0
fi

if [[ "${cmd}" =~ ^git[[:space:]]+push[[:space:]]+.*[[:space:]](main|dev/[a-zA-Z0-9._-]+)$ ]]; then
  printf '{"permission":"deny","reason":"Pushes to main/dev branches are blocked. Open a PR from codex/<workflow>/<short-topic>."}\n'
  exit 2
fi

if [[ "${cmd}" =~ ^git[[:space:]]+(checkout|switch)[[:space:]]+-b[[:space:]]+([^[:space:]]+) ]]; then
  branch="${BASH_REMATCH[2]}"
  if [[ ! "${branch}" =~ ^codex/[a-z0-9._-]+/[a-z0-9._-]+$ ]]; then
    printf '{"permission":"deny","reason":"Branch must match codex/<workflow>/<short-topic>."}\n'
    exit 2
  fi
fi

exit 0
