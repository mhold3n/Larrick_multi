"""Local-file control backend for orchestration signals."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

LOGGER = logging.getLogger(__name__)


class FileControlBackend:
    """Control backend using JSON signal files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send_signal(self, run_id: str, signal: str, payload: dict[str, Any] | None = None) -> bool:
        data = {
            "run_id": str(run_id),
            "signal": str(signal).upper(),
            "payload": payload or {},
            "timestamp": time.time(),
        }
        try:
            self.path.write_text(json.dumps(data), encoding="utf-8")
            return True
        except Exception as exc:
            LOGGER.warning("Failed writing control signal file %s: %s", self.path, exc)
            return False

    def get_signal(self, run_id: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed reading control signal file %s: %s", self.path, exc)
            return None

        target = str(data.get("run_id", ""))
        if target and target != str(run_id):
            return None
        return data

    def clear_signal(self, run_id: str) -> None:  # noqa: ARG002
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception as exc:
            LOGGER.warning("Failed clearing control signal file %s: %s", self.path, exc)

