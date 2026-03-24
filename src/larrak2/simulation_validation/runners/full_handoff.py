"""Full engine phase-handoff regime runner.

Runs only after all upstream regimes pass. Validates state transfer
correctness between phases, conservation across handoffs, restart
consistency, and time-step coupling at transition windows.
"""

from __future__ import annotations

from typing import Any

from ..models import (
    SourceType,
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationMetricResult,
    ValidationRunManifest,
)
from .base import BaseRegimeRunner, evaluate_metric

FULL_HANDOFF_METRIC_CATEGORIES = frozenset(
    {
        "state_conservation",
        "transition_residual",
        "restart_consistency",
        "timestep_coupling",
        "end_to_end_comparison",
    }
)


class FullHandoffRunner(BaseRegimeRunner):
    """Runner for full engine phase-handoff regime.

    Validates:
    - State transfer correctness between phases
    - Conservation across handoffs
    - Restart consistency at regime transitions
    - Sparse canonical time-step coupling behaviour at transition windows
    """

    @property
    def regime_name(self) -> str:
        return "full_handoff"

    def run(
        self,
        dataset: ValidationDatasetManifest,
        case_spec: ValidationCaseSpec,
        simulation_data: dict[str, Any],
    ) -> ValidationRunManifest:
        """Run full-handoff validation.

        simulation_data keys expected:
        - For each metric in dataset.metrics: metric_id → simulated value
        - '{metric_id}_measured' → measured/expected value
        - 'handoff_states': list of dicts with per-transition state data
        - 'full_handoff_provenance': optional provenance dict
        """
        results: list[ValidationMetricResult] = []
        messages: list[str] = []

        for spec in dataset.metrics:
            sim_key = spec.metric_id
            if sim_key not in simulation_data:
                if spec.required:
                    messages.append(f"Missing simulation result for required metric '{sim_key}'")
                    results.append(
                        ValidationMetricResult(
                            metric_id=spec.metric_id,
                            measured_value=0.0,
                            simulated_value=0.0,
                            error=float("inf"),
                            tolerance_used=spec.tolerance_band,
                            passed=False,
                            source_type=spec.source_type,
                            units=spec.units,
                            details={"reason": "missing_simulation_data"},
                        )
                    )
                else:
                    messages.append(f"Skipping optional metric '{sim_key}' (no data)")
                continue

            measured_key = f"{spec.metric_id}_measured"
            measured = float(simulation_data.get(measured_key, 0.0))
            simulated = float(simulation_data[sim_key])

            result = evaluate_metric(spec, measured, simulated)
            results.append(result)

        # Handoff-specific state conservation checks
        handoff_states = simulation_data.get("handoff_states", [])
        for i, state in enumerate(handoff_states):
            conservation_error = float(state.get("conservation_error", 0.0))
            tolerance = float(state.get("conservation_tolerance", 1e-6))
            passed = conservation_error <= tolerance
            results.append(
                ValidationMetricResult(
                    metric_id=f"handoff_conservation_{i}",
                    measured_value=0.0,
                    simulated_value=conservation_error,
                    error=conservation_error,
                    tolerance_used=tolerance,
                    passed=passed,
                    source_type=dataset.metrics[0].source_type
                    if dataset.metrics
                    else SourceType.DERIVED_CONSTRAINT,
                    units="",
                    details={
                        "transition_from": state.get("from_phase", ""),
                        "transition_to": state.get("to_phase", ""),
                    },
                )
            )

        solver_artifacts: dict[str, str] = {}
        provenance = simulation_data.get("full_handoff_provenance", {})
        if provenance:
            solver_artifacts["full_handoff_provenance"] = str(provenance)

        return self._build_run_manifest(
            case_spec=case_spec,
            metric_results=results,
            solver_artifacts=solver_artifacts,
            messages=messages,
        )

    def build_acceptance_outputs(
        self,
        run_manifest: ValidationRunManifest,
    ) -> dict[str, Any]:
        """Structure acceptance outputs for full-handoff validation."""
        outputs: dict[str, Any] = {
            "state_conservation_checks": {},
            "transition_residual_reports": {},
            "end_to_end_comparisons": {},
        }

        for r in run_manifest.metric_results:
            entry = {
                "simulated": r.simulated_value,
                "error": r.error,
                "passed": r.passed,
                "details": r.details,
            }
            if "conservation" in r.metric_id or "handoff" in r.metric_id:
                outputs["state_conservation_checks"][r.metric_id] = entry
            elif "residual" in r.metric_id or "transition" in r.metric_id:
                outputs["transition_residual_reports"][r.metric_id] = entry
            else:
                outputs["end_to_end_comparisons"][r.metric_id] = entry

        outputs["full_handoff_provenance"] = run_manifest.solver_artifacts.get(
            "full_handoff_provenance", ""
        )
        return outputs
