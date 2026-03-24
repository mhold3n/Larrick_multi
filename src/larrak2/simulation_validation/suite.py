"""Suite orchestrator for canonical or custom validation-regime ladders."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .gating import build_unblock_criteria, check_prerequisites
from .models import (
    RegimeStatus,
    SourceType,
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationRunManifest,
    ValidationSuiteManifest,
    ValidationSuiteProfile,
)
from .regimes import CanonicalRegime, canonical_prerequisite_names
from .runners.chemistry import ChemistryRunner
from .runners.closed_cylinder import ClosedCylinderRunner
from .runners.full_handoff import FullHandoffRunner
from .runners.reacting_flow import ReactingFlowRunner
from .runners.spray import SprayRunner
from .solver_adapters import resolve_simulation_inputs

logger = logging.getLogger(__name__)

_RUNNER_MAP = {
    CanonicalRegime.CHEMISTRY: ChemistryRunner,
    CanonicalRegime.SPRAY: SprayRunner,
    CanonicalRegime.REACTING_FLOW: ReactingFlowRunner,
    CanonicalRegime.CLOSED_CYLINDER: ClosedCylinderRunner,
    CanonicalRegime.FULL_HANDOFF: FullHandoffRunner,
}


def run_single_regime(
    regime: CanonicalRegime,
    dataset: ValidationDatasetManifest,
    case_spec: ValidationCaseSpec,
    simulation_data: dict[str, Any],
    *,
    prior_results: dict[str, ValidationRunManifest] | None = None,
    prior_simulation_data: dict[str, dict[str, Any]] | None = None,
) -> ValidationRunManifest:
    """Run validation for a single regime."""
    if (
        dataset.source_type == SourceType.SYNTHETIC
        and dataset.provenance.get("source") == "UNSOURCED_PLACEHOLDER"
    ):
        return ValidationRunManifest(
            regime=regime.value,
            case_spec=case_spec,
            status=RegimeStatus.NOT_RUN,
            messages=[
                "Dataset is an unsourced placeholder; add measured provenance "
                "before running this regime."
            ],
        )

    runner_cls = _RUNNER_MAP.get(regime)
    if runner_cls is None:
        raise ValueError(f"No runner registered for regime '{regime.value}'")
    runner = runner_cls()
    resolved = resolve_simulation_inputs(
        regime.value,
        dataset,
        case_spec,
        simulation_data,
        prior_results=prior_results,
        prior_simulation_data=prior_simulation_data,
    )
    run_manifest = runner.run(dataset, case_spec, resolved.simulation_data)
    run_manifest.messages = [*resolved.messages, *run_manifest.messages]
    run_manifest.solver_artifacts.update(resolved.solver_artifacts)
    run_manifest.solver_artifacts["resolved_simulation_data_json"] = json.dumps(
        resolved.simulation_data,
        sort_keys=True,
    )
    return run_manifest


def run_suite(
    regime_configs: dict[str, dict[str, Any]],
    *,
    suite_profile: ValidationSuiteProfile | None = None,
) -> ValidationSuiteManifest:
    """Run the full validation suite across a configured regime ladder.

    Args:
        regime_configs: dict keyed by regime name, each containing:
            - 'dataset': ValidationDatasetManifest
            - 'case_spec': ValidationCaseSpec
            - 'simulation_data': dict[str, Any]
            If a regime key is missing, that regime is skipped.
        suite_profile: Optional ordered regime profile and prerequisite graph.
            When omitted, the canonical five-regime suite is used.

    Returns:
        ValidationSuiteManifest with scoreboard and per-regime results.
    """
    profile = suite_profile or ValidationSuiteProfile(
        suite_id="canonical_v1",
        regime_order=CanonicalRegime.ordered_names(),
        prerequisites=canonical_prerequisite_names(),
    )
    ordered_regimes = profile.normalized_regime_order()
    prerequisite_map = (
        dict(profile.prerequisites) if profile.prerequisites else canonical_prerequisite_names()
    )

    suite = ValidationSuiteManifest(
        suite_id=profile.suite_id,
        regime_order=ordered_regimes,
        prerequisites=prerequisite_map,
    )
    results: dict[str, ValidationRunManifest] = {}
    resolved_inputs: dict[str, dict[str, Any]] = {}

    for regime_name in ordered_regimes:
        config = regime_configs.get(regime_name)

        # Check prerequisites first
        gate = check_prerequisites(
            regime_name,
            results,
            prerequisite_map=prerequisite_map,
        )
        if not gate.allowed:
            logger.info(
                "Regime '%s' blocked by prerequisites: %s",
                regime_name,
                gate.blockers,
            )
            blocked_manifest = ValidationRunManifest(
                regime=regime_name,
                case_spec=ValidationCaseSpec(
                    case_id=f"{regime_name}_blocked",
                    regime=regime_name,
                ),
                status=RegimeStatus.BLOCKED_BY_PREREQUISITE,
                blocked_by=gate.blockers,
                messages=[f"Blocked by prerequisite(s): {', '.join(gate.blockers)}"],
            )
            results[regime_name] = blocked_manifest
            continue

        if config is None:
            logger.info("Regime '%s' has no config, marking NOT_RUN", regime_name)
            results[regime_name] = ValidationRunManifest(
                regime=regime_name,
                case_spec=ValidationCaseSpec(
                    case_id=f"{regime_name}_not_configured",
                    regime=regime_name,
                ),
                status=RegimeStatus.NOT_RUN,
                messages=["No configuration provided for this regime"],
            )
            continue

        dataset = config["dataset"]
        case_spec = config["case_spec"]
        simulation_data = config["simulation_data"]

        logger.info("Running validation for regime '%s'", regime_name)
        t0 = time.perf_counter()

        try:
            regime_enum = CanonicalRegime(regime_name)
        except ValueError as exc:
            raise ValueError(f"Unsupported regime '{regime_name}'") from exc
        run_manifest = run_single_regime(
            regime_enum,
            dataset,
            case_spec,
            simulation_data,
            prior_results=results,
            prior_simulation_data=resolved_inputs,
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Regime '%s' completed in %.2fs: %s",
            regime_name,
            elapsed,
            run_manifest.status.value,
        )
        results[regime_name] = run_manifest
        resolved_json = run_manifest.solver_artifacts.get("resolved_simulation_data_json", "")
        if resolved_json:
            resolved_inputs[regime_name] = json.loads(resolved_json)
        else:
            resolved_inputs[regime_name] = dict(simulation_data)

    suite.regime_results = results
    suite.unblock_criteria = build_unblock_criteria(
        results,
        regime_order=ordered_regimes,
    )
    suite.build_scoreboard()
    return suite


def suite_to_dict(suite: ValidationSuiteManifest) -> dict[str, Any]:
    """Serialize suite manifest to a JSON-serializable dict."""
    regime_order = list(suite.regime_order) or CanonicalRegime.ordered_names()
    return {
        "suite_id": suite.suite_id,
        "regime_order": regime_order,
        "prerequisites": suite.prerequisites,
        "overall_passed": suite.overall_passed,
        "first_blocking_regime": suite.first_blocking_regime,
        "first_blocking_metric_group": suite.first_blocking_metric_group,
        "scoreboard": [
            {
                "regime": e.regime,
                "status": e.status.value,
                "n_metrics_total": e.n_metrics_total,
                "n_metrics_passed": e.n_metrics_passed,
                "n_metrics_failed": e.n_metrics_failed,
                "blocked_by": e.blocked_by,
                "first_failing_metric": e.first_failing_metric,
            }
            for e in suite.scoreboard
        ],
        "unblock_criteria": suite.unblock_criteria,
        "regime_results": {
            name: {
                "regime": run.regime,
                "status": run.status.value,
                "blocked_by": run.blocked_by,
                "n_metrics": len(run.metric_results),
                "metrics": [
                    {
                        "metric_id": r.metric_id,
                        "measured_value": r.measured_value,
                        "simulated_value": r.simulated_value,
                        "error": r.error,
                        "tolerance_used": r.tolerance_used,
                        "passed": r.passed,
                        "source_type": r.source_type.value,
                        "units": r.units,
                    }
                    for r in run.metric_results
                ],
                "messages": run.messages,
            }
            for name, run in suite.regime_results.items()
        },
        "messages": suite.messages,
    }


def suite_to_json(suite: ValidationSuiteManifest, path: str | Path) -> None:
    """Write suite manifest as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite_to_dict(suite), indent=2), encoding="utf-8")


def suite_to_markdown(suite: ValidationSuiteManifest) -> str:
    """Generate human-readable markdown summary of suite results."""
    regime_order = list(suite.regime_order) or CanonicalRegime.ordered_names()
    lines: list[str] = []
    lines.append("# Simulation Validation Suite Report")
    lines.append("")
    lines.append(f"**Suite ID:** `{suite.suite_id}`")
    lines.append(f"**Regime Order:** {', '.join(regime_order)}")
    lines.append("")

    overall = "✅ ALL PASSED" if suite.overall_passed else "❌ NOT PASSED"
    lines.append(f"**Overall Status:** {overall}")
    lines.append("")

    if suite.first_blocking_regime:
        lines.append(
            f"> ⚠️ **First blocker:** `{suite.first_blocking_regime}` "
            f"(metric: `{suite.first_blocking_metric_group}`)"
        )
        lines.append("")

    # Scoreboard table
    lines.append("## Regime Scoreboard")
    lines.append("")
    lines.append("| Regime | Status | Passed | Failed | Total | Blocked By |")
    lines.append("|--------|--------|--------|--------|-------|------------|")

    status_icons = {
        "passed": "🟢",
        "failed": "🔴",
        "blocked_by_prerequisite": "🟡",
        "not_run": "⚪",
    }

    for entry in suite.scoreboard:
        icon = status_icons.get(entry.status.value, "❓")
        blocked = ", ".join(entry.blocked_by) if entry.blocked_by else "—"
        lines.append(
            f"| {entry.regime} | {icon} {entry.status.value} | "
            f"{entry.n_metrics_passed} | {entry.n_metrics_failed} | "
            f"{entry.n_metrics_total} | {blocked} |"
        )
    lines.append("")

    # Per-regime details
    lines.append("## Per-Regime Details")
    lines.append("")

    for regime_name in regime_order:
        run = suite.regime_results.get(regime_name)
        if run is None:
            continue
        lines.append(f"### {regime_name}")
        lines.append("")
        lines.append(f"**Status:** {run.status.value}")
        if run.blocked_by:
            lines.append(f"**Blocked by:** {', '.join(run.blocked_by)}")
        lines.append("")

        if run.metric_results:
            lines.append("| Metric | Measured | Simulated | Error | Tolerance | Pass |")
            lines.append("|--------|----------|-----------|-------|-----------|------|")
            for r in run.metric_results:
                icon = "✅" if r.passed else "❌"
                lines.append(
                    f"| {r.metric_id} | {r.measured_value:.4g} | "
                    f"{r.simulated_value:.4g} | {r.error:.4g} | "
                    f"{r.tolerance_used:.4g} | {icon} |"
                )
            lines.append("")

        if run.messages:
            lines.append("**Messages:**")
            for msg in run.messages:
                lines.append(f"- {msg}")
            lines.append("")

    # Unblock criteria
    if suite.unblock_criteria:
        lines.append("## Unblock Criteria")
        lines.append("")
        for regime, criteria in suite.unblock_criteria.items():
            lines.append(f"### {regime}")
            for c in criteria:
                lines.append(f"- {c}")
            lines.append("")

    return "\n".join(lines)
