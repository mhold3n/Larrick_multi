"""Tests that agent context docs reflect the main-only workflow model."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

AGENT_FILES = {
    "AGENTS.md": REPO_ROOT / "AGENTS.md",
    "CLAUDE.md": REPO_ROOT / "CLAUDE.md",
    "copilot-instructions.md": REPO_ROOT / ".github" / "copilot-instructions.md",
}

REQUIRED_DOMAIN_PATHS = [
    "src/larrak2/simulation_validation/",
    "src/larrak2/training/",
    "src/larrak2/optimization/",
    "src/larrak2/analysis/",
    "src/larrak2/cem/",
    "src/larrak2/architecture/",
]

REMOVED_AUTOMATION_TERMS = [
    "dev/",
    "codex/",
    "archive_eligible",
    "scripts/start_parallel_task.sh",
    "scripts/route_current_thread.py",
    "scripts/plan_github_concierge.py",
]


@pytest.mark.parametrize("label,path", list(AGENT_FILES.items()))
def test_agent_doc_exists_and_is_nonempty(label: str, path: Path) -> None:
    assert path.exists(), f"{label} must exist at {path}"
    content = path.read_text(encoding="utf-8")
    assert len(content) > 100, f"{label} must have substantive content"


@pytest.mark.parametrize("label,path", list(AGENT_FILES.items()))
def test_agent_doc_mentions_main_only_workflow(label: str, path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    assert "`main` is the only documented repo workflow branch." in content
    assert "Direct commits to `main` are the default working model" in content
    assert "documentation only" in content


@pytest.mark.parametrize("label,path", list(AGENT_FILES.items()))
def test_agent_doc_retains_domain_path_guidance(label: str, path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    for snippet in REQUIRED_DOMAIN_PATHS:
        assert snippet in content, f"{label} must mention '{snippet}'"


@pytest.mark.parametrize("label,path", list(AGENT_FILES.items()))
def test_agent_doc_omits_removed_branch_automation_terms(label: str, path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    for term in REMOVED_AUTOMATION_TERMS:
        assert term not in content, f"{label} must not mention '{term}'"


@pytest.mark.parametrize("label,path", list(AGENT_FILES.items()))
def test_agent_doc_does_not_permit_contents_write(label: str, path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    assert "contents: write" not in content.replace("Do not add `contents: write`", "").replace(
        "Do **not** add `contents: write`", ""
    ), f"{label} must not instruct agents to add contents:write permissions"
