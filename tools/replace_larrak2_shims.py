"""Inventory and remove `larrak2` compatibility shims.

This repo is transitioning from a monolithic `larrak2.*` namespace to a thin
integration surface that depends on external canonical packages:

- `larrak_runtime` (from `larrak-core`)
- `larrak_engines`
- `larrak_simulation`
- `larrak_optimization`
- `larrak_analysis`

During the migration, many `src/larrak2/**` modules were replaced with shims
(`import *`, `__getattr__` delegation, `sys.modules` aliasing). This script
supports the plan to:

1) Inventory shims and emit a JSON manifest (committable for traceability)
2) Rewrite import sites away from `larrak2.*` to canonical packages

Deletion of shim modules is intentionally *not* automated in the first pass;
we rely on the manifest + `rg`/tests to confirm no remaining references before
deleting files.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
LARRAK2_ROOT = SRC_ROOT / "larrak2"


@dataclass(frozen=True)
class ShimRecord:
    path: str
    module: str
    kind: str  # shim_star | shim_delegate | shim_alias | real_module
    target_module: str | None
    adds_symbols: list[str]


_STAR_RE = re.compile(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+\*\s*(#.*)?$")
_GETATTR_RE = re.compile(r"^\s*def\s+__getattr__\s*\(\s*name\s*:\s*str\s*\)")
_SYS_MODULES_RE = re.compile(r"sys\.modules\[\s*__name__\s*\]\s*=\s*")


def _path_to_module(py_file: Path) -> str:
    rel = py_file.relative_to(SRC_ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_ast(path: Path, text: str) -> ast.AST | None:
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError:
        return None


def _detect_shim_kind(text: str) -> str:
    # Fast heuristics: if a file uses module aliasing, that's the strongest signal.
    if _SYS_MODULES_RE.search(text) is not None:
        return "shim_alias"
    if _GETATTR_RE.search(text) is not None:
        return "shim_delegate"
    for line in text.splitlines():
        if _STAR_RE.match(line):
            return "shim_star"
    return "real_module"


def _detect_star_target(text: str) -> str | None:
    for line in text.splitlines():
        m = _STAR_RE.match(line)
        if m:
            return m.group(1)
    return None


def _detect_delegate_target(ast_tree: ast.AST) -> str | None:
    # Look for: from X import ... as _impl  OR  import X as _impl
    target: str | None = None
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == "_impl":
                    target = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for alias in node.names:
                if alias.asname == "_impl":
                    target = node.module
    return target


def _detect_alias_target(ast_tree: ast.AST) -> str | None:
    # Look for `_impl = importlib.import_module("...")` or `import ... as _impl`.
    target: str | None = None
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == "_impl":
                    target = alias.name
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_impl":
                    if isinstance(node.value, ast.Call) and isinstance(
                        node.value.func, ast.Attribute
                    ):
                        # importlib.import_module("x")
                        if (
                            isinstance(node.value.func.value, ast.Name)
                            and node.value.func.value.id == "importlib"
                            and node.value.func.attr == "import_module"
                            and node.value.args
                            and isinstance(node.value.args[0], ast.Constant)
                            and isinstance(node.value.args[0].value, str)
                        ):
                            target = node.value.args[0].value
    return target


def _detect_added_symbols(text: str, target_module: str | None) -> list[str]:
    if target_module is None:
        return []
    added: list[str] = []
    # Heuristic: assignments that reference names likely imported from target.
    # Example: ENCODING_VERSION_V0_4 = LEGACY_ENCODING_VERSION
    for line in text.splitlines():
        if "=" not in line:
            continue
        lhs = line.split("=", 1)[0].strip()
        if lhs.isidentifier() and lhs.isupper():
            added.append(lhs)
    return sorted(set(added))


def inventory_shims(py_files: Iterable[Path]) -> list[ShimRecord]:
    records: list[ShimRecord] = []
    for path in sorted(py_files):
        text = _read_text(path)
        kind = _detect_shim_kind(text)
        mod = _path_to_module(path)
        tree = _parse_ast(path, text)
        target: str | None = None
        if kind == "shim_star":
            target = _detect_star_target(text)
        elif tree is not None and kind == "shim_delegate":
            target = _detect_delegate_target(tree)
        elif tree is not None and kind == "shim_alias":
            target = _detect_alias_target(tree)
        adds = _detect_added_symbols(text, target)
        records.append(
            ShimRecord(
                path=str(path.relative_to(REPO_ROOT)),
                module=mod,
                kind=kind,
                target_module=target,
                adds_symbols=adds,
            )
        )
    return records


def _iter_python_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # Skip hidden and venv-ish folders defensively.
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p


def cmd_inventory(args: argparse.Namespace) -> int:
    if not LARRAK2_ROOT.exists():
        raise SystemExit(f"Missing expected directory: {LARRAK2_ROOT}")
    records = inventory_shims(_iter_python_files(LARRAK2_ROOT))
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "generated_by": "tools/replace_larrak2_shims.py inventory",
        "records": [asdict(r) for r in records],
        "counts": {
            "total": len(records),
            "shim_star": sum(1 for r in records if r.kind == "shim_star"),
            "shim_delegate": sum(1 for r in records if r.kind == "shim_delegate"),
            "shim_alias": sum(1 for r in records if r.kind == "shim_alias"),
            "real_module": sum(1 for r in records if r.kind == "real_module"),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(REPO_ROOT)} ({payload['counts']})")
    return 0


def _rewrite_module_prefix(mod: str, mapping: dict[str, str]) -> str:
    for src_prefix, dst_prefix in mapping.items():
        if mod == src_prefix or mod.startswith(src_prefix + "."):
            return dst_prefix + mod[len(src_prefix) :]
    return mod


def _rewrite_import_lines(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Rewrite `import` and `from ... import ...` lines only.

    This is intentionally conservative to avoid rewriting strings, comments, etc.
    """
    changed = 0
    out_lines: list[str] = []
    for line in text.splitlines(keepends=False):
        new_line = line
        m_from = re.match(r"^(\s*)from\s+([a-zA-Z0-9_\.]+)\s+import\s+", line)
        if m_from:
            indent, mod = m_from.group(1), m_from.group(2)
            new_mod = _rewrite_module_prefix(mod, mapping)
            if new_mod != mod:
                new_line = line.replace(
                    f"{indent}from {mod} import ", f"{indent}from {new_mod} import ", 1
                )
        else:
            m_imp = re.match(r"^(\s*)import\s+([a-zA-Z0-9_\.]+)(\s+as\s+\w+)?\s*$", line)
            if m_imp:
                indent, mod = m_imp.group(1), m_imp.group(2)
                new_mod = _rewrite_module_prefix(mod, mapping)
                if new_mod != mod:
                    new_line = line.replace(f"{indent}import {mod}", f"{indent}import {new_mod}", 1)
        if new_line != line:
            changed += 1
        out_lines.append(new_line)
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else ""), changed


def cmd_rewrite_imports(args: argparse.Namespace) -> int:
    mapping = {
        "larrak2.core": "larrak_runtime.core",
        "larrak2.architecture": "larrak_runtime.architecture",
        "larrak2.surrogate.quality_contract": "larrak_runtime.surrogate.quality_contract",
        "larrak2.training": "larrak_simulation.training",
        "larrak2.simulation_validation": "larrak_simulation.simulation_validation",
        "larrak2.pipelines.openfoam": "larrak_simulation.pipelines.openfoam",
        "larrak2.cem": "larrak_engines.cem",
        "larrak2.realworld": "larrak_engines.realworld",
        "larrak2.thermo": "larrak_engines.thermo",
        "larrak2.gear": "larrak_engines.gear",
        "larrak2.analysis": "larrak_analysis",
        # Principles pipelines were extracted to `larrak_optimization.pipelines.*`
        "larrak2.pipelines.principles_core": "larrak_optimization.pipelines.principles_core",
        "larrak2.pipelines.principles_evaluator": "larrak_optimization.pipelines.principles_evaluator",
        "larrak2.pipelines.principles_frontier": "larrak_optimization.pipelines.principles_frontier",
        "larrak2.pipelines.principles_search": "larrak_optimization.pipelines.principles_search",
        # Orchestration loop was extracted to `larrak-orchestration`.
        "larrak2.orchestration.cache": "larrak_orchestration.legacy_loop.cache",
        "larrak2.orchestration.budget": "larrak_orchestration.legacy_loop.budget",
        "larrak2.orchestration.trust_region": "larrak_orchestration.legacy_loop.trust_region",
        "larrak2.orchestration.backends": "larrak_orchestration.legacy_loop.backends",
        # Solver adapter extracted under `larrak-optimization` integrations.
        "larrak2.orchestration.adapters.solver_adapter": "larrak_optimization.integrations.orchestration",
    }

    roots = [REPO_ROOT / p for p in args.roots]
    py_files: list[Path] = []
    for r in roots:
        if not r.exists():
            continue
        py_files.extend(list(_iter_python_files(r)))

    total_line_edits = 0
    touched_files = 0
    for path in sorted(set(py_files)):
        original = _read_text(path)
        rewritten, edits = _rewrite_import_lines(original, mapping)
        if edits == 0:
            continue
        total_line_edits += edits
        touched_files += 1
        if not args.check:
            path.write_text(rewritten, encoding="utf-8")
        else:
            # Emit minimal info for check-only mode.
            print(f"{path.relative_to(REPO_ROOT)}: {edits} import-line rewrites")

    mode = "CHECK" if args.check else "APPLY"
    print(f"{mode}: touched_files={touched_files} total_line_edits={total_line_edits}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="replace_larrak2_shims")
    sub = p.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("inventory", help="Inventory shim modules under src/larrak2/**.")
    inv.add_argument(
        "--out",
        default="tools/shim_manifest_larrak2.json",
        help="Output JSON path (repo-relative).",
    )
    inv.set_defaults(func=cmd_inventory)

    rw = sub.add_parser("rewrite-imports", help="Rewrite larrak2.* imports to canonical packages.")
    rw.add_argument(
        "--roots",
        nargs="+",
        default=["src", "tests", "tools", "scripts"],
        help="Repo-relative roots to scan for Python files.",
    )
    rw.add_argument(
        "--check", action="store_true", help="Print intended changes without modifying files."
    )
    rw.set_defaults(func=cmd_rewrite_imports)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
