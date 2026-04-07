"""Contract tests for the post-shim import graph.

These tests intentionally focus on the *integration surface* responsibilities
of `Larrick_multi` after extraction:

- Canonical packages (`larrak_runtime`, `larrak_engines`, `larrak_simulation`,
  `larrak_optimization`, `larrak_analysis`) are importable.
- Deleted `larrak2.*` shim namespaces are not importable.

Detailed physics/simulation correctness is owned by the extracted repos and
should be tested there.
"""

from __future__ import annotations


def _assert_not_importable(module: str) -> None:
    try:
        __import__(module)
    except ModuleNotFoundError:
        return
    raise AssertionError(f"Expected `{module}` to be unimportable (shim removed).")


def test_canonical_packages_importable() -> None:
    import larrak_analysis  # noqa: F401
    import larrak_engines  # noqa: F401
    import larrak_optimization  # noqa: F401
    import larrak_runtime  # noqa: F401
    import larrak_simulation  # noqa: F401


def test_deleted_larrak2_shim_namespaces_are_unimportable() -> None:
    for mod in [
        # Removed shim modules (the containing directories may remain as namespace packages).
        "larrak2.core.encoding",
        "larrak2.core.types",
        "larrak2.core.evaluator",
        "larrak2.architecture.contracts",
        "larrak2.architecture.workflow_contracts",
        "larrak2.training.workflows",
        "larrak2.training.overnight_campaign",
        "larrak2.analysis.workflows",
        "larrak2.analysis.sensitivity",
        "larrak2.cem.tribology",
        "larrak2.realworld.constraints",
        "larrak2.thermo.symbolic_artifact",
        "larrak2.gear.picogk_adapter",
        "larrak2.pipelines.openfoam",
        "larrak2.simulation_validation.__init__",
        "larrak2.orchestration.cache",
        "larrak2.orchestration.budget",
        "larrak2.orchestration.trust_region",
        "larrak2.orchestration.backends.control_file",
        "larrak2.orchestration.backends.control_redis",
        "larrak2.orchestration.backends.provenance_jsonl",
        "larrak2.orchestration.backends.provenance_weaviate",
        "larrak2.orchestration.adapters.solver_adapter",
    ]:
        _assert_not_importable(mod)
