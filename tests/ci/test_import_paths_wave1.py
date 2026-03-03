"""Import and path-hygiene checks for Wave 1 migration."""

from __future__ import annotations

import importlib
from pathlib import Path


MODULES = [
    "larrak2.optimization.numerical.casadi_problem_spec",
    "larrak2.optimization.solvers.ipopt",
    "larrak2.optimization.scaling",
    "larrak2.optimization.initialization.surrogate_adapter",
    "larrak2.optimization.slicing.active_set",
    "larrak2.optimization.slicing.slice_problem",
    "larrak2.optimization.slicing.symbolic_slice_problem",
    "larrak2.surrogate.hifi.ensemble",
    "larrak2.surrogate.hifi.models",
    "larrak2.surrogate.stack.artifact",
    "larrak2.surrogate.stack.runtime",
    "larrak2.surrogate.stack.symbolic",
    "larrak2.surrogate.stack.train",
    "larrak2.training.hifi_schema",
    "larrak2.training.hifi_train",
]


def test_wave1_modules_importable():
    for mod in MODULES:
        loaded = importlib.import_module(mod)
        assert loaded is not None


def test_no_sys_path_hacks_in_wave1_files():
    repo_root = Path(__file__).resolve().parents[2]
    files = [
        repo_root / "src/larrak2/training/hifi_train.py",
        repo_root / "src/larrak2/training/hifi_schema.py",
        repo_root / "src/larrak2/surrogate/hifi/models.py",
        repo_root / "src/larrak2/surrogate/hifi/ensemble.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "sys.path.insert" not in text
