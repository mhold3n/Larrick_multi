"""Fail-soft behavior for optional orchestration backends."""

from __future__ import annotations

from pathlib import Path

from larrak2.orchestration.backends import RedisControlBackend, WeaviateProvenanceBackend


def test_orchestration_backends_are_not_importable_as_deep_modules() -> None:
    """Deep import paths are intentionally not part of the stable API surface.

    Backends are provided via `larrak2.orchestration.backends` which re-exports
    extracted implementations. Keeping per-module deep paths would duplicate code
    across repos and is intentionally avoided.
    """

    import importlib

    for name in (
        "larrak2.orchestration.backends.control_file",
        "larrak2.orchestration.backends.control_redis",
        "larrak2.orchestration.backends.provenance_jsonl",
        "larrak2.orchestration.backends.provenance_weaviate",
    ):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"Deep backend module import unexpectedly succeeded: {name}")


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
