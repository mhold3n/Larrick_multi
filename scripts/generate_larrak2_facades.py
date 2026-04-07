#!/usr/bin/env python3
"""Generate thin larrak2 facade modules that mirror submodule packages (including private names).

Star-import facades omit leading-underscore symbols; tests import those, so we copy the
full public namespace of each canonical module into the facade via `globals().update`.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "larrak2"

HEADER = '''\
"""Facade: canonical implementation lives in `{canonical}` (submodule package).

This file is part of the Larrick_multi integration distribution only.
"""

from __future__ import annotations

import importlib

_canonical = importlib.import_module("{canonical}")
for _k, _v in vars(_canonical).items():
    if _k.startswith("__"):
        continue
    globals()[_k] = _v
del importlib, _canonical, _k, _v
'''


def _module_from_rel(rel: Path, canonical_prefix: str) -> str:
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    suffix = ".".join(parts)
    return f"{canonical_prefix}.{suffix}" if suffix else canonical_prefix


def write_facades(local_subdir: str, canonical_prefix: str) -> int:
    base = SRC / local_subdir
    if not base.is_dir():
        raise SystemExit(f"missing {base}")
    count = 0
    for py in sorted(base.rglob("*.py")):
        rel = py.relative_to(base)
        canonical = _module_from_rel(rel, canonical_prefix)
        py.write_text(HEADER.format(canonical=canonical), encoding="utf-8")
        count += 1
    return count


def main() -> None:
    total = 0
    total += write_facades("optimization", "larrak_optimization.optimization")
    total += write_facades("core", "larrak_runtime.core")
    sur_base = SRC / "surrogate"
    for py in sorted(sur_base.rglob("*.py")):
        rel = py.relative_to(sur_base)
        rel_s = str(rel)
        if rel_s in ("openfoam_authority.py", "stack/train.py"):
            continue
        canonical = _module_from_rel(rel, "larrak_runtime.surrogate")
        py.write_text(HEADER.format(canonical=canonical), encoding="utf-8")
        total += 1
    for sub in ("gear", "cem", "realworld", "ports", "thermo"):
        total += write_facades(sub, f"larrak_runtime.{sub}")
    sim_base = SRC / "simulation_validation"
    sub_sim = ROOT / "larrak-simulation" / "src" / "larrak_simulation" / "simulation_validation"
    for py in sorted(sim_base.rglob("*.py")):
        rel = py.relative_to(sim_base)
        if rel.name == "engine_calibration.py":
            continue
        canon_rel = sub_sim / rel
        if not canon_rel.exists():
            raise SystemExit(f"submodule missing mirror for {rel} -> {canon_rel}")
        canonical = _module_from_rel(rel, "larrak_simulation.simulation_validation")
        py.write_text(HEADER.format(canonical=canonical), encoding="utf-8")
        total += 1
    print(f"wrote {total} facade modules")


if __name__ == "__main__":
    main()
