"""Optional Redis control backend (fail-soft)."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

LOGGER = logging.getLogger(__name__)


class RedisControlBackend:
    """Control backend backed by Redis string keys."""

    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis = None
        self._module_error: str | None = None

        try:
            import redis as redis_module

            self._redis = redis_module
        except Exception as exc:
            self._module_error = str(exc)
            LOGGER.warning("Redis backend unavailable; control signals will be ignored: %s", exc)

        self._client = None

    def _conn(self):
        if self._redis is None:
            return None
        if self._client is not None:
            return self._client
        try:
            self._client = self._redis.from_url(self.redis_url)
            return self._client
        except Exception as exc:
            LOGGER.warning("Failed to connect to Redis control backend: %s", exc)
            return None

    def send_signal(self, run_id: str, signal: str, payload: dict[str, Any] | None = None) -> bool:
        conn = self._conn()
        if conn is None:
            return False
        key = f"larrak2:orch:control:{run_id}"
        data = {
            "run_id": str(run_id),
            "signal": str(signal).upper(),
            "payload": payload or {},
            "timestamp": time.time(),
        }
        try:
            conn.set(key, json.dumps(data))
            return True
        except Exception as exc:
            LOGGER.warning("Failed to send Redis signal: %s", exc)
            return False

    def get_signal(self, run_id: str) -> dict[str, Any] | None:
        conn = self._conn()
        if conn is None:
            return None
        key = f"larrak2:orch:control:{run_id}"
        try:
            data = conn.get(key)
            if data is None:
                return None
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return json.loads(str(data))
        except Exception as exc:
            LOGGER.warning("Failed to read Redis signal: %s", exc)
            return None

    def clear_signal(self, run_id: str) -> None:
        conn = self._conn()
        if conn is None:
            return
        key = f"larrak2:orch:control:{run_id}"
        try:
            conn.delete(key)
        except Exception as exc:
            LOGGER.warning("Failed to clear Redis signal: %s", exc)
