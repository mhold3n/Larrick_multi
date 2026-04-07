---
name: pre-pr-check
description: Run Larrick's standard pre-PR checks from current worktree.
---

# Pre-PR Check

Run these commands from the active worktree:

```bash
ruff format --check .
ruff check .
mypy src
pytest -q
```

If a workflow only owns a subset of paths, scope `mypy` and `pytest` accordingly.
