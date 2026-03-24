"""Tests that agent context documents exist and contain required branch information."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

AGENT_FILES = {
    "AGENTS.md": REPO_ROOT / "AGENTS.md",
    "CLAUDE.md": REPO_ROOT / "CLAUDE.md",
    "copilot-instructions.md": REPO_ROOT / ".github" / "copilot-instructions.md",
}

REQUIRED_BRANCHES = [
    "dev/simulation",
    "dev/training",
    "dev/optimization",
    "dev/analysis",
    "dev/cem-orchestration",
]

REQUIRED_CONCEPTS = [
    "codex/",  # task-branch naming convention
    "main",  # integration branch
    "contents: write" not in "",  # placeholder — checked below via separate assertion
]


@pytest.mark.parametrize("label,path", list(AGENT_FILES.items()))
def test_agent_doc_exists_and_is_nonempty(label: str, path: Path) -> None:
    assert path.exists(), f"{label} must exist at {path}"
    content = path.read_text(encoding="utf-8")
    assert len(content) > 100, f"{label} must have substantive content"


@pytest.mark.parametrize("label,path", list(AGENT_FILES.items()))
def test_agent_doc_contains_all_workflow_branches(label: str, path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    for branch in REQUIRED_BRANCHES:
        assert branch in content, f"{label} must mention branch '{branch}'"


@pytest.mark.parametrize("label,path", list(AGENT_FILES.items()))
def test_agent_doc_contains_codex_branch_convention(label: str, path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    assert "codex/" in content, (
        f"{label} must document the codex/<workflow>/<topic> naming convention"
    )


@pytest.mark.parametrize("label,path", list(AGENT_FILES.items()))
def test_agent_doc_does_not_permit_contents_write(label: str, path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    # The docs must not instruct agents to add write permissions.
    # Checking for the permissive phrasing; the prohibition text ("Do not add contents: write")
    # is expected and desirable.
    assert "contents: write" not in content.replace("Do **not** add `contents: write`", "").replace(
        "Do not add `contents: write`", ""
    ), f"{label} must not instruct agents to add contents:write permissions"
