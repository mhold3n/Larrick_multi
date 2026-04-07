#!/usr/bin/env bash
# Install pinned Larrak packages from ``requirements-external.txt``, then link monorepo ``data/``
# into the venv so wheel-installed ``larrak_runtime`` resolves thermo JSON like CI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LARRICK_MULTI_ROOT="${ROOT}"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT}/requirements-external.txt"

# Wheel-installed ``larrak_runtime`` resolves ``data/thermo/*.json`` under
# ``Path(__file__).parents[3]/data`` (i.e. ``$venv/lib/pythonX.Y/data``), not the monorepo.
python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import site

repo = Path(os.environ["LARRICK_MULTI_ROOT"]).resolve()
roots = [Path(p) for p in site.getsitepackages()]
anchor: Path | None = None
for r in roots:
    cand = r / "larrak_runtime" / "thermo" / "timing_profile.py"
    if cand.is_file():
        anchor = cand
        break
if anchor is None:
    raise SystemExit("install_external_larrak: larrak_runtime timing_profile not in site-packages")

data_parent = anchor.resolve().parents[3]
target = data_parent / "data"
src = (repo / "data").resolve()
if not src.is_dir():
    raise SystemExit(f"install_external_larrak: expected monorepo data/ at {src}")
target.unlink(missing_ok=True)
target.symlink_to(src, target_is_directory=True)
print(f"Linked {target} -> {src}")
PY
