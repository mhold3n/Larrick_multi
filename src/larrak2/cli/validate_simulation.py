"""Simulation validation CLI — delegates to larrak-simulation; re-exports test helpers."""

from __future__ import annotations

from typing import Any


def main(argv: list[str] | None = None) -> int:
    from larrak_simulation.cli.validate_simulation import main as _main

    return _main(argv)


def __getattr__(name: str) -> Any:
    """Forward attributes (e.g. `run_validation_preflight`) to the canonical CLI module."""

    from larrak_simulation.cli import validate_simulation as _mod

    return getattr(_mod, name)


def __dir__() -> list[str]:
    from larrak_simulation.cli import validate_simulation as _mod

    return sorted(set(globals()) | set(vars(_mod).keys()))


if __name__ == "__main__":
    raise SystemExit(main())
