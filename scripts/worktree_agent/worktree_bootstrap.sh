#!/bin/bash
set -euo pipefail

# Portable bootstrap for any worktree-hosted agent runtime.
# Cursor and non-Cursor adapters should call this script.

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

if [[ -n "${ROOT_WORKTREE_PATH:-}" ]] && [[ -f "${ROOT_WORKTREE_PATH}/.env" ]]; then
  cp "${ROOT_WORKTREE_PATH}/.env" .env
fi

python -m pytest -q tests/ci/test_workflow_ci_layout.py tests/ci/test_agent_docs.py
