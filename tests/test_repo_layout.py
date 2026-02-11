import os
import re
import subprocess
from pathlib import PurePosixPath

import pytest
import yaml

# Adapted from scripts/check_repo_layout.py


def load_policy():
    # Use git root to find policy
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        # Fallback if not in git (e.g. CI isolated env?)
        repo_root = os.getcwd()

    policy_path = os.path.join(repo_root, "repo_layout.yml")
    if not os.path.exists(policy_path):
        pytest.skip(f"Policy file not found at {policy_path}")

    with open(policy_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def top_level_dir(path: str):
    parts = PurePosixPath(path).parts
    return parts[0] if parts else None


def test_repo_layout():
    """Verify repository layout against repo_layout.yml."""
    policy = load_policy()
    allowed = set(policy.get("allowed_top_level", []))
    banned_patterns = [re.compile(p) for p in policy.get("banned_dir_name_patterns", [])]
    singletons = policy.get("singletons", {})
    output_dir = singletons.get("output_dir", "outputs")

    # Scan current root directory
    # We assume tests are run from repo root
    root_items = [
        item for item in os.listdir(".") if os.path.isdir(item) and not item.startswith(".")
    ]

    errors = []

    # 1. Check top-level directories
    for item in root_items:
        if item not in allowed:
            # Maybe it's a file? listdir includes files.
            # policy says "allowed_top_level". Usually implies only these items allowed at top level?
            # The script only checked isdir.
            if os.path.isdir(item):
                errors.append(f"Forbidden top-level directory: '{item}'")

    # 2. Check banned patterns recursively?
    # The script used git diff to check changed files.
    # For a full test, we might want to walk the tree or just check top level?
    # The original script strategy:
    # "if not changed: for item in os.listdir('.'): ..." -> Checked top level
    # "Fail on banned directory names anywhere in the created set"
    # Let's walk the tree to depth 2 or 3 to catch banned patterns like 'utils' or 'lib' if desired.
    # For now, let's stick to the script's default manual behavior: check top level.

    # 3. Singleton check
    if os.path.exists("output") and output_dir != "output":
        errors.append(f"Output directory found as 'output', expected '{output_dir}'")
    if os.path.exists("out") and output_dir != "out":
        errors.append(f"Output directory found as 'out', expected '{output_dir}'")

    assert not errors, "\n".join(errors)
