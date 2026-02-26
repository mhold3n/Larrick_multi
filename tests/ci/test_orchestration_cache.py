"""Tests for orchestration cache."""

from __future__ import annotations

from pathlib import Path

from larrak2.orchestration.cache import EvaluationCache


def test_evaluation_cache_get_or_compute_and_persistence(tmp_path: Path) -> None:
    cache_path = tmp_path / "eval_cache.pkl"
    cache = EvaluationCache(max_size=2, persist_path=cache_path)

    calls = {"n": 0}

    def _compute(params: dict[str, object]) -> dict[str, object]:
        calls["n"] += 1
        return {"objective": float(params["x"])}  # type: ignore[index]

    result1, cached1 = cache.get_or_compute({"x": 1.0}, _compute)
    result2, cached2 = cache.get_or_compute({"x": 1.0}, _compute)

    assert cached1 is False
    assert cached2 is True
    assert calls["n"] == 1
    assert result1 == result2

    cache.get_or_compute({"x": 2.0}, _compute)
    cache.get_or_compute({"x": 3.0}, _compute)  # should evict one entry (LRU)
    stats = cache.get_statistics()
    assert stats["evictions"] >= 1

    cache.save_to_disk()
    assert cache_path.exists()

    cache2 = EvaluationCache(max_size=2, persist_path=cache_path)
    stats2 = cache2.get_statistics()
    assert stats2["size"] >= 1

