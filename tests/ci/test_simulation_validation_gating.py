"""Unit tests — prerequisite ordering and blocked_by_prerequisite behaviour."""

from __future__ import annotations

from larrak2.simulation_validation.gating import (
    build_unblock_criteria,
    check_prerequisites,
    validate_four_stroke_data_usage,
)
from larrak2.simulation_validation.models import (
    RegimeStatus,
    SourceType,
    ValidationCaseSpec,
    ValidationMetricResult,
    ValidationRunManifest,
)
from larrak2.simulation_validation.regimes import PREREQUISITE_MAP, CanonicalRegime


def _make_passing_run(regime: str) -> ValidationRunManifest:
    return ValidationRunManifest(
        regime=regime,
        case_spec=ValidationCaseSpec(case_id=f"{regime}_c1", regime=regime),
        metric_results=[
            ValidationMetricResult(
                metric_id=f"{regime}_m1",
                measured_value=1.0,
                simulated_value=1.0,
                error=0.0,
                tolerance_used=0.1,
                passed=True,
                source_type=SourceType.MEASURED,
            ),
        ],
        status=RegimeStatus.PASSED,
    )


def _make_failing_run(regime: str) -> ValidationRunManifest:
    return ValidationRunManifest(
        regime=regime,
        case_spec=ValidationCaseSpec(case_id=f"{regime}_c1", regime=regime),
        metric_results=[
            ValidationMetricResult(
                metric_id=f"{regime}_m1",
                measured_value=1.0,
                simulated_value=5.0,
                error=4.0,
                tolerance_used=0.1,
                passed=False,
                source_type=SourceType.MEASURED,
            ),
        ],
        status=RegimeStatus.FAILED,
    )


class TestPrerequisiteChecks:
    def test_chemistry_has_no_prerequisites(self):
        gate = check_prerequisites(CanonicalRegime.CHEMISTRY, {})
        assert gate.allowed is True
        assert gate.blockers == []

    def test_spray_has_no_prerequisites(self):
        gate = check_prerequisites(CanonicalRegime.SPRAY, {})
        assert gate.allowed is True

    def test_reacting_flow_blocked_without_chemistry(self):
        gate = check_prerequisites(CanonicalRegime.REACTING_FLOW, {})
        assert gate.allowed is False
        assert "chemistry" in gate.blockers

    def test_reacting_flow_allowed_with_passing_chemistry(self):
        results = {"chemistry": _make_passing_run("chemistry")}
        gate = check_prerequisites(CanonicalRegime.REACTING_FLOW, results)
        assert gate.allowed is True

    def test_reacting_flow_blocked_with_failing_chemistry(self):
        results = {"chemistry": _make_failing_run("chemistry")}
        gate = check_prerequisites(CanonicalRegime.REACTING_FLOW, results)
        assert gate.allowed is False
        assert "chemistry" in gate.blockers

    def test_full_handoff_needs_spray_reacting_closed(self):
        prereqs = PREREQUISITE_MAP[CanonicalRegime.FULL_HANDOFF]
        prereq_names = {p.value for p in prereqs}
        assert prereq_names == {"spray", "reacting_flow", "closed_cylinder"}

    def test_full_handoff_blocked_without_all_prereqs(self):
        gate = check_prerequisites(CanonicalRegime.FULL_HANDOFF, {})
        assert gate.allowed is False
        assert set(gate.blockers) == {"spray", "reacting_flow", "closed_cylinder"}

    def test_full_handoff_allowed_with_all_passing(self):
        results = {
            "spray": _make_passing_run("spray"),
            "reacting_flow": _make_passing_run("reacting_flow"),
            "closed_cylinder": _make_passing_run("closed_cylinder"),
        }
        gate = check_prerequisites(CanonicalRegime.FULL_HANDOFF, results)
        assert gate.allowed is True

    def test_full_handoff_blocked_by_single_failure(self):
        results = {
            "spray": _make_passing_run("spray"),
            "reacting_flow": _make_failing_run("reacting_flow"),
            "closed_cylinder": _make_passing_run("closed_cylinder"),
        }
        gate = check_prerequisites(CanonicalRegime.FULL_HANDOFF, results)
        assert gate.allowed is False
        assert "reacting_flow" in gate.blockers


class TestBlockedByPrerequisite:
    """Downstream regimes marked blocked_by_prerequisite don't count as
    failed or passed."""

    def test_blocked_is_not_passed(self):
        run = ValidationRunManifest(
            regime="full_handoff",
            case_spec=ValidationCaseSpec(case_id="fh1", regime="full_handoff"),
            status=RegimeStatus.BLOCKED_BY_PREREQUISITE,
            blocked_by=["chemistry"],
        )
        assert run.passed is False

    def test_blocked_is_not_failed(self):
        run = ValidationRunManifest(
            regime="full_handoff",
            case_spec=ValidationCaseSpec(case_id="fh1", regime="full_handoff"),
            status=RegimeStatus.BLOCKED_BY_PREREQUISITE,
        )
        assert run.status != RegimeStatus.FAILED


class TestUnblockCriteria:
    def test_failing_regime_produces_criteria(self):
        results = {
            "chemistry": _make_failing_run("chemistry"),
        }
        criteria = build_unblock_criteria(results)
        assert "chemistry" in criteria
        assert any("Fix" in c for c in criteria["chemistry"])

    def test_blocked_regime_references_upstream(self):
        results = {
            "full_handoff": ValidationRunManifest(
                regime="full_handoff",
                case_spec=ValidationCaseSpec(case_id="fh1", regime="full_handoff"),
                status=RegimeStatus.BLOCKED_BY_PREREQUISITE,
                blocked_by=["spray"],
            ),
        }
        criteria = build_unblock_criteria(results)
        assert "full_handoff" in criteria
        assert any("spray" in c for c in criteria["full_handoff"])

    def test_not_run_regime_suggests_running(self):
        criteria = build_unblock_criteria({})
        for regime in CanonicalRegime.ordered():
            assert regime.value in criteria
            assert any("Run" in c for c in criteria[regime.value])


class TestFourStrokeConstraint:
    def test_allowed_in_closed_cylinder(self):
        assert validate_four_stroke_data_usage("closed_cylinder") is True

    def test_rejected_for_chemistry(self):
        assert validate_four_stroke_data_usage("chemistry") is False

    def test_rejected_for_spray(self):
        assert validate_four_stroke_data_usage("spray") is False

    def test_rejected_for_full_handoff(self):
        assert validate_four_stroke_data_usage("full_handoff") is False
