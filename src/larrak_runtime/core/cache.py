"""Evaluation cache with memoization.

Caches evaluation results keyed by (hash(x), ctx fields, model version).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .types import EvalContext, EvalResult

# Global cache version (bump when model changes)
CACHE_VERSION = "0.1.0"


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key for evaluation results."""

    x_hash: str
    rpm: float
    torque: float
    fidelity: int
    seed: int
    version: str


def _hash_array(x: np.ndarray) -> str:
    """Compute stable hash of array."""
    return hashlib.sha256(x.tobytes()).hexdigest()[:16]


def make_cache_key(x: np.ndarray, ctx: EvalContext) -> CacheKey:
    """Create cache key from inputs.

    Args:
        x: Decision vector.
        ctx: Evaluation context.

    Returns:
        CacheKey for lookup.
    """
    return CacheKey(
        x_hash=_hash_array(np.asarray(x, dtype=np.float64)),
        rpm=ctx.rpm,
        torque=ctx.torque,
        fidelity=ctx.fidelity,
        seed=ctx.seed,
        version=CACHE_VERSION,
    )


class EvalCache:
    """In-memory cache for evaluation results."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._cache: dict[CacheKey, EvalResult] = {}
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, x: np.ndarray, ctx: EvalContext) -> EvalResult | None:
        """Retrieve cached result if available.

        Args:
            x: Decision vector.
            ctx: Evaluation context.

        Returns:
            Cached EvalResult or None if not found.
        """
        key = make_cache_key(x, ctx)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, x: np.ndarray, ctx: EvalContext, result: EvalResult) -> None:
        """Store result in cache.

        Args:
            x: Decision vector.
            ctx: Evaluation context.
            result: Evaluation result to cache.
        """
        # Simple LRU-like eviction: clear oldest half when full
        if len(self._cache) >= self._maxsize:
            keys = list(self._cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._cache[k]

        key = make_cache_key(x, ctx)
        self._cache[key] = result

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }


# Global cache instance
_global_cache = EvalCache()


def get_cache() -> EvalCache:
    """Get global evaluation cache."""
    return _global_cache
