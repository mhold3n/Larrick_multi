"""Acceptance tests — end-to-end suite scenarios with green paths and negative tests."""

from __future__ import annotations

import pytest

from larrak2.simulation_validation.dataset_registry import (
    DatasetRegistry,
    DatasetRegistryError,
)
from larrak2.simulation_validation.models import (
    ComparisonMode,
    RegimeStatus,
    SourceType,
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationMetricSpec,
)
from larrak2.simulation_validation.suite import run_suite

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(regime: str, dataset_id: str | None = None) -> ValidationDatasetManifest:
    """Build a simple measured dataset for a given regime."""
    did = dataset_id or f"{regime}_dataset"
    return ValidationDatasetManifest(
        dataset_id=did,
        regime=regime,
        fuel_family="gasoline",
        source_type=SourceType.MEASURED,
        metrics=[
            ValidationMetricSpec(
                metric_id=f"{regime}_metric_1",
                units="unit",
                comparison_mode=ComparisonMode.ABSOLUTE,
                tolerance_band=1.0,
                source_type=SourceType.MEASURED,
            ),
        ],
    )


def _make_case(regime: str) -> ValidationCaseSpec:
    return ValidationCaseSpec(case_id=f"{regime}_case_1", regime=regime)


def _passing_sim(regime: str) -> dict:
    return {
        f"{regime}_metric_1": 1.0,
        f"{regime}_metric_1_measured": 1.0,
    }


def _failing_sim(regime: str) -> dict:
    return {
        f"{regime}_metric_1": 10.0,
        f"{regime}_metric_1_measured": 1.0,
    }


def _regime_config(regime: str, passing: bool = True) -> dict:
    return {
        "dataset": _make_dataset(regime),
        "case_spec": _make_case(regime),
        "simulation_data": _passing_sim(regime) if passing else _failing_sim(regime),
    }


# ---------------------------------------------------------------------------
# Green path scenarios
# ---------------------------------------------------------------------------


class TestGreenPaths:
    def test_one_green_chemistry_case_set(self):
        configs = {"chemistry": _regime_config("chemistry", passing=True)}
        suite = run_suite(configs)
        chem = suite.regime_results["chemistry"]
        assert chem.status == RegimeStatus.PASSED

    def test_one_green_spray_case_set(self):
        configs = {"spray": _regime_config("spray", passing=True)}
        suite = run_suite(configs)
        assert suite.regime_results["spray"].status == RegimeStatus.PASSED

    def test_one_green_tnf_case(self):
        configs = {
            "chemistry": _regime_config("chemistry", passing=True),
            "reacting_flow": _regime_config("reacting_flow", passing=True),
        }
        suite = run_suite(configs)
        assert suite.regime_results["reacting_flow"].status == RegimeStatus.PASSED

    def test_one_green_closed_cylinder_case(self):
        configs = {"closed_cylinder": _regime_config("closed_cylinder", passing=True)}
        suite = run_suite(configs)
        assert suite.regime_results["closed_cylinder"].status == RegimeStatus.PASSED

    def test_end_to_end_full_handoff_after_all_green(self):
        """Full handoff only runs after all upstream regimes are green."""
        configs = {
            "chemistry": _regime_config("chemistry", passing=True),
            "spray": _regime_config("spray", passing=True),
            "reacting_flow": _regime_config("reacting_flow", passing=True),
            "closed_cylinder": _regime_config("closed_cylinder", passing=True),
            "full_handoff": _regime_config("full_handoff", passing=True),
        }
        suite = run_suite(configs)
        assert suite.regime_results["full_handoff"].status == RegimeStatus.PASSED
        assert suite.overall_passed is True


# ---------------------------------------------------------------------------
# Negative scenarios
# ---------------------------------------------------------------------------


class TestNegativeScenarios:
    def test_chemistry_failure_blocks_full_handoff(self):
        """When chemistry fails, full_handoff must be blocked_by_prerequisite."""
        configs = {
            "chemistry": _regime_config("chemistry", passing=False),
            "spray": _regime_config("spray", passing=True),
            "reacting_flow": _regime_config("reacting_flow", passing=True),
            "closed_cylinder": _regime_config("closed_cylinder", passing=True),
            "full_handoff": _regime_config("full_handoff", passing=True),
        }
        suite = run_suite(configs)

        # Chemistry failed → reacting_flow blocked → full_handoff blocked
        assert suite.regime_results["chemistry"].status == RegimeStatus.FAILED
        assert suite.regime_results["reacting_flow"].status == RegimeStatus.BLOCKED_BY_PREREQUISITE
        assert suite.regime_results["full_handoff"].status == RegimeStatus.BLOCKED_BY_PREREQUISITE
        assert suite.overall_passed is False

    def test_synthetic_metric_missing_provenance_rejected(self):
        """Synthetic metric without measured-anchor provenance is rejected
        at the registry level."""
        reg = DatasetRegistry()
        # Register measured first
        reg.register(
            ValidationDatasetManifest(
                dataset_id="measured_ds",
                regime="chemistry",
                fuel_family="gasoline",
                source_type=SourceType.MEASURED,
            )
        )

        # Try synthetic without anchors/basis
        bad_synthetic = ValidationDatasetManifest(
            dataset_id="bad_synthetic",
            regime="chemistry",
            fuel_family="gasoline",
            source_type=SourceType.SYNTHETIC,
            measured_anchor_ids=[],
            governing_basis="",
        )
        with pytest.raises(DatasetRegistryError, match="Provenance validation failed"):
            reg.register(bad_synthetic)

    def test_blocked_regimes_not_counted_as_failed(self):
        """Blocked regimes should have status=BLOCKED, not FAILED."""
        configs = {
            "chemistry": _regime_config("chemistry", passing=False),
        }
        suite = run_suite(configs)
        rf = suite.regime_results["reacting_flow"]
        assert rf.status == RegimeStatus.BLOCKED_BY_PREREQUISITE
        assert rf.status != RegimeStatus.FAILED
        assert rf.status != RegimeStatus.PASSED

    def test_unsourced_placeholder_dataset_is_not_run(self):
        configs = {
            "closed_cylinder": {
                "dataset": ValidationDatasetManifest(
                    dataset_id="closed_cylinder_placeholder",
                    regime="closed_cylinder",
                    fuel_family="gasoline",
                    source_type=SourceType.SYNTHETIC,
                    provenance={"source": "UNSOURCED_PLACEHOLDER"},
                ),
                "case_spec": _make_case("closed_cylinder"),
                "simulation_data": _passing_sim("closed_cylinder"),
            }
        }

        suite = run_suite(configs)
        closed_cylinder = suite.regime_results["closed_cylinder"]
        assert closed_cylinder.status == RegimeStatus.NOT_RUN
        assert any("unsourced placeholder" in m.lower() for m in closed_cylinder.messages)


class TestSuiteSummary:
    def test_scoreboard_surfaces_first_blocker(self):
        configs = {
            "chemistry": _regime_config("chemistry", passing=False),
        }
        suite = run_suite(configs)
        suite.build_scoreboard()
        assert suite.first_blocking_regime == "chemistry"
        assert suite.first_blocking_metric_group != ""

    def test_full_green_scoreboard(self):
        configs = {
            "chemistry": _regime_config("chemistry", passing=True),
            "spray": _regime_config("spray", passing=True),
            "reacting_flow": _regime_config("reacting_flow", passing=True),
            "closed_cylinder": _regime_config("closed_cylinder", passing=True),
            "full_handoff": _regime_config("full_handoff", passing=True),
        }
        suite = run_suite(configs)
        assert suite.overall_passed is True
        assert len(suite.scoreboard) == 5
        assert all(e.status == RegimeStatus.PASSED for e in suite.scoreboard)
