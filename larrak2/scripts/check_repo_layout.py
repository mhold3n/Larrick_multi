#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate top-level repository layout.")
    parser.add_argument(
        "--config",
        default="repo_layout.yml",
        help="Path to the repo layout configuration file.",
    )
    parser.add_argument(
        "--changed-file",
        help="Path to a file containing changed paths (one per line).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repository root directory (defaults to repo root).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def match_patterns(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def is_allowed_file(name: str, allowed_files: set[str], allowed_patterns: list[str]) -> bool:
    return name in allowed_files or match_patterns(name, allowed_patterns)


def collect_top_level_targets(paths: Iterable[str]) -> set[str]:
    targets = set()
    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        targets.add(path.split("/", 1)[0])
    return targets


def validate_full_layout(root: Path, config: dict) -> list[str]:
    allowed_dirs = set(config.get("allowed_top_level_dirs", []))
    required_dirs = set(config.get("required_top_level_dirs", []))
    allowed_files = set(config.get("allowed_top_level_files", []))
    allowed_patterns = list(config.get("allowed_top_level_file_patterns", []))
    ignore_dirs = set(config.get("ignore_top_level_dirs", [])) | {".git"}

    violations: list[str] = []
    top_level_entries = {entry.name: entry for entry in root.iterdir()}

    for name, entry in top_level_entries.items():
        if entry.is_dir():
            if name in ignore_dirs:
                continue
            if name not in allowed_dirs:
                violations.append(f"Unexpected top-level directory: {name}")
        else:
            if not is_allowed_file(name, allowed_files, allowed_patterns):
                violations.append(f"Unexpected top-level file: {name}")

    for required in sorted(required_dirs):
        if required not in top_level_entries:
            violations.append(f"Missing required top-level directory: {required}")

    return violations


def validate_changed_paths(root: Path, config: dict, changed_paths: list[str]) -> list[str]:
    allowed_dirs = set(config.get("allowed_top_level_dirs", []))
    allowed_files = set(config.get("allowed_top_level_files", []))
    allowed_patterns = list(config.get("allowed_top_level_file_patterns", []))

    violations: list[str] = []
    targets = collect_top_level_targets(changed_paths)

    for target in sorted(targets):
        candidate = root / target
        if "/" in target:
            continue
        if candidate.is_dir():
            if target not in allowed_dirs:
                violations.append(f"Unexpected top-level directory in PR: {target}")
        elif candidate.exists():
            if not is_allowed_file(target, allowed_files, allowed_patterns):
                violations.append(f"Unexpected top-level file in PR: {target}")

    return violations


def main() -> int:
    args = parse_args()
    script_root = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve() if args.root else script_root
    config_path = (root / args.config).resolve()

    config = load_config(config_path)

    if args.changed_file:
        changed_path = Path(args.changed_file)
        changed_paths = changed_path.read_text(encoding="utf-8").splitlines()
        violations = validate_changed_paths(root, config, changed_paths)
    else:
        violations = validate_full_layout(root, config)

    if violations:
        print("Repo layout violations detected:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("Repo layout check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
