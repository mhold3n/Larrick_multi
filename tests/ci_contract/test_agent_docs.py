"""Ensure agent docs reflect current repo workflow constraints."""

from __future__ import annotations

from pathlib import Path

import pytest

AGENT_FILES = {
    "AGENTS.md": Path(__file__).resolve().parents[2] / "AGENTS.md",
    "CLAUDE.md": Path(__file__).resolve().parents[2] / "CLAUDE.md",
}

REMOVED_AUTOMATION_TERMS = (
    "codex/",
    "workflow_ownership.yml",
    "start_parallel_task.sh",
)


@pytest.mark.parametrize("label,path", list(AGENT_FILES.items()))
def test_agent_doc_omits_removed_branch_automation_terms(label: str, path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    for term in REMOVED_AUTOMATION_TERMS:
        assert term not in content, f"{label} must not mention '{term}'"
