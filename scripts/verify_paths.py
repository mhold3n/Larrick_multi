import subprocess
import sys
from pathlib import Path

# Add src to path to allow import
# Root is now the directory where this script runs (or parent/scripts)
# We are running from root.
repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
src_path = Path(repo_root) / "src"
sys.path.append(str(src_path))

try:
    from larrak2 import paths

    print(f"Calculated Root: {paths.REPO_ROOT}")
    print(f"Actual Root:     {repo_root}")

    if str(paths.REPO_ROOT) == repo_root:
        print("SUCCESS: REPO_ROOT matches git root")
    else:
        print("FAILURE: REPO_ROOT mismatch")
        sys.exit(1)

except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
