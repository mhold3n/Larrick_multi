"""Workflow CI layout tests for the main-only repo model."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

REMOVED_WORKFLOWS = [
    "simulation.yml",
    "training.yml",
    "optimization.yml",
    "analysis.yml",
    "cem-orchestration.yml",
    "ownership.yml",
    "reusable-fast.yml",
]


def _load_yaml(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_ci_workflow_targets_main_only_and_is_read_only() -> None:
    workflow = _load_yaml(WORKFLOW_DIR / "ci.yml")
    assert tuple(workflow["on"]["push"]["branches"]) == ("main",)
    assert tuple(workflow["on"]["pull_request"]["branches"]) == ("main",)
    assert workflow["permissions"]["contents"] == "read"
    text = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    assert "git push" not in text
    assert "contents: write" not in text
    assert "dev/" not in text
    assert "codex/" not in text


def test_ci_workflow_inlines_fast_checks() -> None:
    workflow = _load_yaml(WORKFLOW_DIR / "ci.yml")
    steps = workflow["jobs"]["fast-checks"]["steps"]
    rendered = "\n".join(str(step) for step in steps)
    assert "actions/cache" in rendered
    assert "ruff format --check" in rendered
    assert "ruff check" in rendered
    assert "mypy" in rendered
    assert "pytest -q" in rendered


def test_removed_wrapper_workflows_are_absent() -> None:
    for filename in REMOVED_WORKFLOWS:
        assert not (WORKFLOW_DIR / filename).exists(), f"{filename} should be removed"


def test_ci_telemetry_tracks_the_unified_ci_workflow() -> None:
    workflow = _load_yaml(WORKFLOW_DIR / "ci-telemetry.yml")
    assert tuple(workflow["on"]["workflow_run"]["workflows"]) == ("CI",)
    assert tuple(workflow["on"]["workflow_run"]["branches"]) == ("main",)
    assert workflow["permissions"]["actions"] == "read"
