#!/usr/bin/env python3
"""Validate repository top-level layout against repo_layout.yml."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
import sys
from typing import Iterable

import yaml


DEFAULT_CONFIG = "repo_layout.yml"


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Layout config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _normalize_list(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    return [str(value) for value in values]


def _is_allowed_file(filename: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)


def _check_full_layout(repo_root: Path, config: dict) -> list[str]:
    errors: list[str] = []
    allowed_dirs = set(_normalize_list(config.get("allowed_top_level_dirs")))
    allowed_files = _normalize_list(config.get("allowed_top_level_files"))
    ignored = set(_normalize_list(config.get("ignored_top_level")))
    required_dirs = set(_normalize_list(config.get("required_top_level_dirs")))

    entries = [entry for entry in repo_root.iterdir() if entry.name not in ignored]
    for entry in entries:
        if entry.is_dir():
            if entry.name not in allowed_dirs:
                errors.append(f"Unexpected top-level directory: {entry.name}")
        else:
            if not _is_allowed_file(entry.name, allowed_files):
                errors.append(f"Unexpected top-level file: {entry.name}")

    missing = sorted(required_dirs - {entry.name for entry in entries if entry.is_dir()})
    for name in missing:
        errors.append(f"Missing required top-level directory: {name}")

    return errors


def _check_changed_paths(changed_file: Path, config: dict) -> list[str]:
    errors: list[str] = []
    allowed_dirs = set(_normalize_list(config.get("allowed_top_level_dirs")))
    allowed_files = _normalize_list(config.get("allowed_top_level_files"))

    with changed_file.open("r", encoding="utf-8") as handle:
        paths = [line.strip() for line in handle if line.strip()]

    for path in paths:
        top_level = path.split("/", 1)[0]
        if "/" in path:
            if top_level not in allowed_dirs:
                errors.append(
                    f"Changed path uses disallowed top-level directory: {top_level} ({path})"
                )
        else:
            if not _is_allowed_file(top_level, allowed_files):
                errors.append(
                    f"Changed path uses disallowed top-level file: {top_level}"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to repo layout config (default: repo_layout.yml)",
    )
    parser.add_argument(
        "--changed-file",
        help="Optional file listing changed paths (one per line).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / args.config

    try:
        config = _load_config(config_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.changed_file:
        errors = _check_changed_paths(Path(args.changed_file), config)
    else:
        errors = _check_full_layout(repo_root, config)

    if errors:
        print("Repo layout check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repo layout check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
