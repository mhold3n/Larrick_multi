"""Prerequisite gating logic for validation-suite regime ladders."""

from __future__ import annotations

from dataclasses import dataclass

from .models import RegimeStatus, ValidationRunManifest
from .regimes import CanonicalRegime, canonical_prerequisite_names


@dataclass
class GateResult:
    """Result of a prerequisite gate check."""

    allowed: bool
    blockers: list[str]
    blocker_details: dict[str, str]


def check_prerequisites(
    regime: CanonicalRegime | str,
    suite_results: dict[str, ValidationRunManifest],
    *,
    prerequisite_map: dict[str, list[str]] | None = None,
) -> GateResult:
    """Check whether a regime is allowed to run given current suite results.

    Returns a GateResult with allowed=True if all prerequisites passed,
    or allowed=False with the list of blocking regimes.
    """
    regime_name = regime.value if isinstance(regime, CanonicalRegime) else str(regime)
    prerequisites = (
        dict(prerequisite_map) if prerequisite_map is not None else canonical_prerequisite_names()
    ).get(regime_name, [])
    if not prerequisites:
        return GateResult(allowed=True, blockers=[], blocker_details={})

    blockers: list[str] = []
    details: dict[str, str] = {}

    for prereq in prerequisites:
        prereq_name = prereq.value if isinstance(prereq, CanonicalRegime) else str(prereq)
        run = suite_results.get(prereq_name)
        if run is None:
            blockers.append(prereq_name)
            details[prereq_name] = "not_run"
        elif run.status != RegimeStatus.PASSED:
            blockers.append(prereq_name)
            # Find first failing metric for context
            first_fail = ""
            for r in run.metric_results:
                if not r.passed:
                    first_fail = r.metric_id
                    break
            details[prereq_name] = f"status={run.status.value}" + (
                f", first_failing_metric={first_fail}" if first_fail else ""
            )

    return GateResult(
        allowed=len(blockers) == 0,
        blockers=blockers,
        blocker_details=details,
    )


def build_unblock_criteria(
    suite_results: dict[str, ValidationRunManifest],
    *,
    regime_order: list[str] | None = None,
) -> dict[str, list[str]]:
    """For each non-passed regime, list what must be fixed to unblock downstream.

    Returns dict mapping regime name → list of required actions.
    """
    criteria: dict[str, list[str]] = {}
    ordered_regimes = list(regime_order) if regime_order else CanonicalRegime.ordered_names()

    for regime_name in ordered_regimes:
        run = suite_results.get(regime_name)
        if run is None:
            criteria[regime_name] = [f"Run {regime_name} validation"]
            continue
        if run.status == RegimeStatus.BLOCKED_BY_PREREQUISITE:
            criteria[regime_name] = [f"Resolve upstream blocker(s): {', '.join(run.blocked_by)}"]
            continue
        if run.status == RegimeStatus.FAILED:
            failing = [r.metric_id for r in run.metric_results if not r.passed]
            criteria[regime_name] = (
                [f"Fix failing metric(s): {', '.join(failing)}"]
                if failing
                else [f"Resolve {regime_name} failure"]
            )

    return criteria


# ---------------------------------------------------------------------------
# Data-source constraint helpers
# ---------------------------------------------------------------------------


FOUR_STROKE_ALLOWED_REGIMES = frozenset({CanonicalRegime.CLOSED_CYLINDER.value})


def validate_four_stroke_data_usage(regime: str) -> bool:
    """Four-stroke engine data is allowed only in closed_cylinder, never for
    scavenging or charge-preparation validation."""
    return regime in FOUR_STROKE_ALLOWED_REGIMES
