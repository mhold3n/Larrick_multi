"""JSONL provenance backend for orchestration events."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class JSONLProvenanceBackend:
    """Appends one JSON event per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        except Exception as exc:
            LOGGER.warning("Failed writing provenance JSONL event to %s: %s", self.path, exc)

    def close(self) -> None:
        return None
