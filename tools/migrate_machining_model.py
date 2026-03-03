"""Migrate legacy machining NN artifact from src/ into outputs/artifacts/..."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from larrak2.core.artifact_paths import DEFAULT_MACHINING_NN_ARTIFACT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate machining NN artifact to canonical path")
    parser.add_argument(
        "--legacy-path",
        type=str,
        default="src/larrak2/surrogate/machining_surrogate.pth",
        help="Legacy src-relative artifact path",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(DEFAULT_MACHINING_NN_ARTIFACT),
        help="Destination artifact path under outputs/artifacts/...",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move file instead of copy",
    )
    args = parser.parse_args(argv)

    legacy = Path(args.legacy_path)
    dest = Path(args.dest)
    if not legacy.exists():
        raise FileNotFoundError(f"Legacy artifact not found: {legacy}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if args.move:
        shutil.move(str(legacy), str(dest))
        action = "moved"
    else:
        shutil.copy2(str(legacy), str(dest))
        action = "copied"

    print(f"{action} machining surrogate artifact to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
