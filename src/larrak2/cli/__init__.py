"""CLI modules for running optimization and evaluation.

Note: avoid importing submodules at import-time. This keeps `python -m larrak2.cli.<cmd>`
free of `runpy` warnings and avoids side effects from eager imports.
"""

from __future__ import annotations


def run_pareto_main(argv: list[str] | None = None) -> int:
    """Lazy wrapper for `larrak2.cli.run_pareto.main`."""

    from .run_pareto import main

    return main(argv)


def run_single_main(argv: list[str] | None = None) -> int:
    """Lazy wrapper for `larrak2.cli.run_single.main`."""

    from .run_single import main

    return main(argv)


def validate_main(argv: list[str] | None = None) -> int:
    """Lazy wrapper for `larrak2.cli.validate.main`."""

    from .validate import main

    return main(argv)


def validate_simulation_main(argv: list[str] | None = None) -> int:
    """Lazy wrapper for `larrak2.cli.validate_simulation.main`."""

    from .validate_simulation import main as sim_main

    return sim_main(argv)


__all__ = ["run_pareto_main", "run_single_main", "validate_main", "validate_simulation_main"]
