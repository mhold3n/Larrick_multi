"""Lightweight runtime-package resolution for staged engine benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_package_manifest(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "package_manifest.json"
    if not manifest_path.exists():
        return {}
    payload = _load_json(manifest_path)
    payload.setdefault("package_dir", str(package_dir.resolve()))
    return payload


def resolve_engine_runtime_package(
    *,
    config_path: str | Path,
    package_label: str = "",
    refresh_packages: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a runtime chemistry package from a strategy config.

    The current simulation benchmark path only needs file-based selection by
    label and package directory. This helper keeps that resolution explicit and
    side-effect free; `refresh_packages` is accepted for call compatibility but
    is currently a no-op.
    """

    del refresh_packages
    strategy = _load_json(config_path)
    runtime_entry = dict(strategy.get("runtime_package", {}) or {})
    checkpoint_entries = [dict(item) for item in list(strategy.get("checkpoint_packages", []) or [])]

    selected_entry: dict[str, Any] | None = None
    normalized_label = str(package_label).strip()
    if normalized_label:
        if str(runtime_entry.get("label", "")).strip() == normalized_label:
            selected_entry = runtime_entry
        else:
            selected_entry = next(
                (
                    entry
                    for entry in checkpoint_entries
                    if str(entry.get("label", "")).strip() == normalized_label
                ),
                None,
            )
        if selected_entry is None:
            raise ValueError(f"Package label '{package_label}' not found in {config_path}")
    else:
        selected_entry = runtime_entry

    package_dir_raw = str(selected_entry.get("package_dir", "")).strip()
    if not package_dir_raw:
        raise ValueError(f"Resolved package entry in {config_path} is missing package_dir")

    package_dir = Path(package_dir_raw)
    if not package_dir.is_absolute():
        cwd_candidate = (Path.cwd() / package_dir).resolve()
        if cwd_candidate.exists():
            package_dir = cwd_candidate
        else:
            package_dir = (Path(config_path).resolve().parent / package_dir).resolve()
    else:
        package_dir = package_dir.resolve()
    if not package_dir.exists():
        raise FileNotFoundError(f"Runtime package directory not found: {package_dir}")

    manifest = _load_package_manifest(package_dir)
    if not manifest:
        manifest = {
            "package_id": str(selected_entry.get("label", package_dir.name)),
            "package_hash": "",
            "package_dir": str(package_dir),
        }
    manifest.setdefault("package_dir", str(package_dir))
    return package_dir, manifest


__all__ = ["resolve_engine_runtime_package"]
