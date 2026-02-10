#!/usr/bin/env python3
# scripts/check_repo_layout.py
import os
import re
import subprocess
import sys
from pathlib import PurePosixPath

try:
    import yaml  # pip install pyyaml
except ImportError:
    print("Missing dependency: pyyaml. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


def git_changed_paths():
    # In CI you might want: origin/main...HEAD; locally staged is best signal.
    # We try provided args first, then staged changes
    if len(sys.argv) > 1:
        # If args provided (e.g. from pre-commit passing filenames), use them
        # BUT pre-commit with pass_filenames: false passes no args, so we check diff
        pass

    cmd = ["git", "diff", "--cached", "--name-only"]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return [p for p in out.splitlines() if p.strip()]
    except subprocess.CalledProcessError:
        # If not a git repo or error, fallback (or could return empty)
        return []


def load_policy():
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    policy_path = os.path.join(repo_root, "repo_layout.yml")
    if not os.path.exists(policy_path):
        print(f"Policy file not found at {policy_path}", file=sys.stderr)
        sys.exit(2)

    with open(policy_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def top_level_dir(path: str):
    parts = PurePosixPath(path).parts
    return parts[0] if parts else None


def main():
    try:
        policy = load_policy()
    except subprocess.CalledProcessError:
        print("Not a git repository or git error.", file=sys.stderr)
        sys.exit(1)

    allowed = set(policy.get("allowed_top_level", []))
    banned_patterns = [re.compile(p) for p in policy.get("banned_dir_name_patterns", [])]
    singletons = policy.get("singletons", {})
    output_dir = singletons.get("output_dir", "outputs")

    changed = git_changed_paths()
    if not changed:
        # No changed files, nothing to check implies success?
        # Or maybe we want to scan the whole tree if manually run?
        # For now, let's scan all directories in root if no changes detected (manual run)
        # OR just exit 0.
        # Let's check if we are in a CI/hook context.
        # If manually run, maybe we want to validate the whole current state.
        pass

    # If manual run (no staged changes), check all top-level dirs
    # Use os.listdir(".") but exclude files
    dirs_to_check = set()
    if not changed:
        for item in os.listdir("."):
            if os.path.isdir(item):
                dirs_to_check.add(item)
    else:
        for p in changed:
            tl = top_level_dir(p)
            if tl and os.path.isdir(
                tl
            ):  # Only check if it looks like a directory or path implies one
                dirs_to_check.add(tl)

            # Also capture implied dirs for banned pattern check
            parts = PurePosixPath(p).parts
            accum = ""
            for part in parts[:-1]:
                accum = f"{accum}/{part}" if accum else part
                # dirs_to_check.add(accum) # We track all implied dirs?
                # The logic in proposal was: track created dirs.
                pass

    # Actually, the proposal logic was robust:
    # "Track any created dirs by scanning changed paths and checking filesystem"

    created_dirs = set()

    # If using git diff
    if changed:
        for p in changed:
            tl = top_level_dir(p)
            if tl is None:
                continue

            # Hard fail on unknown top-level dirs if they exist
            if tl not in allowed:
                # Check if it is a directory in filesystem?
                # If deleted, it won't exist. If created, it will.
                if os.path.exists(tl):  # and os.path.isdir(tl) - strictly allow files?
                    # Allowed allowed_top_level usually implies dirs.
                    # If it is a file, strictly speaking it is top level item.
                    # Let's assume allowed_top_level applies to everything?
                    # "allowed_top_level directories". Files might be exempt?
                    if os.path.isdir(tl):
                        print(
                            f"[LAYOUT] Forbidden top-level directory: '{tl}' (path: {p})",
                            file=sys.stderr,
                        )
                        sys.exit(1)

            # Collect all implied directories
            parts = PurePosixPath(p).parts
            accum = ""
            for part in parts[:-1]:
                accum = f"{accum}/{part}" if accum else part
                created_dirs.add(accum)
    else:
        # Fallback to scanning current directory for "manual" clean run
        # This double checks existing structure
        for item in os.listdir("."):
            if os.path.isdir(item):
                if item not in allowed:
                    print(f"[LAYOUT] Forbidden top-level directory: '{item}'", file=sys.stderr)
                    sys.exit(1)
                created_dirs.add(item)

    # Fail on banned directory names anywhere in the created set (or observed set)
    for d in sorted(created_dirs, key=len):
        name = PurePosixPath(d).name
        for pat in banned_patterns:
            if pat.match(name):
                # Ignore if explicitly allowed?
                # E.g. if we have a pattern but it's in allowlist?
                # Unlikely for deep dirs, but for top level:
                if d in allowed:
                    continue
                print(
                    f"[LAYOUT] Banned directory name '{name}' matched '{pat.pattern}' (dir: {d})",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Singleton sanity
    for d in created_dirs:
        # Only check top-level for output dir singleton enforcement usually
        if top_level_dir(d) == PurePosixPath(d).parts[0]:
            name = PurePosixPath(d).name
            if name != output_dir and re.match(r"^(out|output|outputs)[-_ ]?\d*$", name):
                # If it is explicitly allowed, skip?
                if name in allowed and name == output_dir:
                    continue
                print(
                    f"[LAYOUT] Output directory must be '{output_dir}', not '{name}'",
                    file=sys.stderr,
                )
                sys.exit(1)

    print("[LAYOUT] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
