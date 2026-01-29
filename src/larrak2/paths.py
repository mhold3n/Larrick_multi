from pathlib import Path

# Calculate repository root relative to this file
# src/larrak2/paths.py -> src/larrak2 -> src -> ROOT
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Canonical directories
OUTPUT_DIR = REPO_ROOT / "outputs"
FRONTEND_DIR = REPO_ROOT / "frontend"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DOCS_DIR = REPO_ROOT / "docs"

# Ensure output directory exists?
# Usually better to let the application create it, but helpful to encourage existence.
# if not OUTPUT_DIR.exists():
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
