"""Controlled import strategy for v1 modules.

This is the ONLY file allowed to touch sys.path or import v1 modules.
Currently, we copy pure functions instead of importing to avoid side effects.

This module exists as a placeholder for future dynamic imports if needed.
"""

from __future__ import annotations

import os
from pathlib import Path

# Environment flag to enable v1 port
V1_ENABLED = os.environ.get("LARRAK2_ENABLE_V1", "0") == "1"

# Expected v1 repo location (relative to this workspace)
V1_REPO_PATH = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "Larrak"


def is_v1_available() -> bool:
    """Check if v1 repo is available.

    Returns True if:
    - LARRAK2_ENABLE_V1=1 is set
    - AND v1 repo exists at expected location
    """
    if not V1_ENABLED:
        return False

    # Check if v1 repo exists
    v1_campro = V1_REPO_PATH / "campro"
    return v1_campro.is_dir()


def get_v1_repo_path() -> Path | None:
    """Get path to v1 repo if available."""
    if is_v1_available():
        return V1_REPO_PATH
    return None


# Note: We intentionally do NOT provide a get_v1_module() function.
# All v1 code has been copied into this package to avoid import-time side effects.
# If you need to import v1 modules dynamically, use importlib with care.
