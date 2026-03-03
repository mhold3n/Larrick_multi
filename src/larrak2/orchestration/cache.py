"""Evaluation cache for expensive backend calls."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """One cache entry."""

    result: Any
    param_hash: str
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = int(self.hits + self.misses)
        return float(self.hits / total) if total > 0 else 0.0


class EvaluationCache:
    """Stable-hash memoization cache with optional persistence."""

    def __init__(self, max_size: int = 10_000, persist_path: str | Path | None = None) -> None:
        if int(max_size) <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self.max_size = int(max_size)
        self.persist_path = Path(persist_path) if persist_path else None

        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self._stats = CacheStats()

        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

    def _to_serializable(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": obj.tolist(),
                "__dtype__": str(obj.dtype),
                "__shape__": list(obj.shape),
            }
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        return obj

    def _compute_hash(self, params: dict[str, Any]) -> str:
        serializable = self._to_serializable(params)
        payload = json.dumps(serializable, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def _touch(self, key: str) -> None:
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_lru(self) -> None:
        while len(self._cache) > self.max_size:
            if not self._access_order:
                break
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
            self._stats.evictions += 1

    def get(self, params: dict[str, Any]) -> Any | None:
        key = self._compute_hash(params)
        entry = self._cache.get(key)
        if entry is None:
            self._stats.misses += 1
            return None
        entry.hit_count += 1
        self._stats.hits += 1
        self._touch(key)
        return entry.result

    def put(self, params: dict[str, Any], result: Any) -> None:
        key = self._compute_hash(params)
        self._cache[key] = CacheEntry(result=result, param_hash=key)
        self._touch(key)
        self._evict_lru()

    def get_or_compute(
        self,
        params: dict[str, Any],
        compute_fn: Callable[[dict[str, Any]], Any],
    ) -> tuple[Any, bool]:
        cached = self.get(params)
        if cached is not None:
            return cached, True
        result = compute_fn(params)
        self.put(params, result)
        return result, False

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

    def save_to_disk(self) -> None:
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("wb") as handle:
            pickle.dump(
                {
                    "cache": self._cache,
                    "access_order": self._access_order,
                    "stats": self._stats,
                },
                handle,
            )
        LOGGER.info("Saved cache to %s (%d entries)", self.persist_path, len(self._cache))

    def _load_from_disk(self) -> None:
        if self.persist_path is None:
            return
        try:
            with self.persist_path.open("rb") as handle:
                data = pickle.load(handle)
            self._cache = data.get("cache", {})
            self._access_order = data.get("access_order", [])
            loaded_stats = data.get("stats")
            if isinstance(loaded_stats, CacheStats):
                self._stats = loaded_stats
            LOGGER.info("Loaded cache from %s (%d entries)", self.persist_path, len(self._cache))
        except Exception as exc:
            LOGGER.warning("Failed to load cache %s: %s", self.persist_path, exc)

    def get_statistics(self) -> dict[str, Any]:
        return {
            "size": int(len(self._cache)),
            "max_size": int(self.max_size),
            "hits": int(self._stats.hits),
            "misses": int(self._stats.misses),
            "hit_rate": float(self._stats.hit_rate),
            "evictions": int(self._stats.evictions),
        }


__all__ = [
    "CacheEntry",
    "CacheStats",
    "EvaluationCache",
]
