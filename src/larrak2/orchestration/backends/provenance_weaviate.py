"""Optional Weaviate provenance backend (fail-soft)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class WeaviateProvenanceBackend:
    """Attempts to write provenance events to Weaviate when available."""

    def __init__(
        self,
        *,
        weaviate_url: str | None = None,
        collection: str = "OrchestrationEvent",
        mirror_jsonl: str | Path | None = None,
    ) -> None:
        self.weaviate_url = weaviate_url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.collection = str(collection)
        self.path = Path(mirror_jsonl) if mirror_jsonl else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._enabled = False

        try:
            import weaviate

            self._client = weaviate.connect_to_custom(
                http_host=self.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                http_port=int(
                    self.weaviate_url.split(":")[-1]
                    if ":" in self.weaviate_url.replace("http://", "").replace("https://", "")
                    else 8080
                ),
                http_secure=self.weaviate_url.startswith("https://"),
                grpc_host=self.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
                grpc_secure=False,
            )
            self._enabled = True
        except Exception as exc:
            LOGGER.warning("Weaviate backend unavailable; running without it: %s", exc)
            self._enabled = False

    def _mirror(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            import json

            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        except Exception as exc:
            LOGGER.warning("Failed to mirror Weaviate provenance event: %s", exc)

    def log_event(self, event: dict[str, Any]) -> None:
        if not self._enabled or self._client is None:
            self._mirror(event)
            return
        try:
            collection = self._client.collections.get(self.collection)
            collection.data.insert(properties={k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in event.items()})
        except Exception as exc:
            LOGGER.warning("Weaviate event insert failed; disabling backend: %s", exc)
            self._enabled = False
        finally:
            self._mirror(event)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

