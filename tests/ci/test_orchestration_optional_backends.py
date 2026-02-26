"""Fail-soft behavior for optional orchestration backends."""

from __future__ import annotations

from pathlib import Path

from larrak2.orchestration.backends import RedisControlBackend, WeaviateProvenanceBackend


def test_optional_backends_fail_soft(tmp_path: Path) -> None:
    redis_backend = RedisControlBackend(redis_url="redis://127.0.0.1:1/0")
    # Must not raise even if Redis module/server is unavailable.
    _ = redis_backend.get_signal("test-run")
    redis_backend.clear_signal("test-run")
    redis_backend.send_signal("test-run", "STOP")

    mirror_path = tmp_path / "weaviate_mirror.jsonl"
    prov_backend = WeaviateProvenanceBackend(
        weaviate_url="http://127.0.0.1:1",
        mirror_jsonl=mirror_path,
    )
    prov_backend.log_event({"type": "smoke", "run_id": "test-run"})
    prov_backend.close()
    assert mirror_path.exists()

