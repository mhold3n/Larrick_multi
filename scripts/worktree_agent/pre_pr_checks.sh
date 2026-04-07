#!/bin/bash
set -euo pipefail

# Portable pre-PR validation gate used by any worktree adapter.

. .venv/bin/activate
ruff format --check .
ruff check .
mypy src
pytest -q
