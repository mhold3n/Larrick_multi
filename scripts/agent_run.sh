#!/bin/bash
# scripts/agent_run.sh
# Wrapper to enforce layout before running agent
set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

echo "Checking repository layout..."
python3 scripts/check_repo_layout.py

if [ $? -eq 0 ]; then
    echo "Layout OK. Starting agent..."
    # Placeholder for actual agent command or just exit success
    # "$@"
    exit 0
else
    echo "Layout violation detected. Aborting agent run."
    exit 1
fi
