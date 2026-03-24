"""Turbulent reacting-flow regime runner.

Primary validation source for v1: one TNF piloted jet flame case with
sufficient temperature, species, and velocity coverage.
"""

from __future__ import annotations

from typing import Any

from ..models import (
    ValidationCaseSpec,
    ValidationDatasetManifest,
    ValidationMetricResult,
    ValidationRunManifest,
)
from .base import BaseRegimeRunner, evaluate_metric

REACTING_FLOW_METRIC_CATEGORIES = frozenset(
    {
        "temperature",
        "species",
        "velocity",
        "scalar_dissipation",
    }
)


class ReactingFlowRunner(BaseRegimeRunner):
    """Runner for turbulent reacting-flow regime validation.

    Validates:
    - Turbulence-chemistry interaction assumptions
    - Reduced chemistry behaviour under mixing and strain
    - Transport / scalar-dissipation sensitivity in a controlled geometry
    """

    @property
    def regime_name(self) -> str:
        return "reacting_flow"

    def run(
        self,
        dataset: ValidationDatasetManifest,
        case_spec: ValidationCaseSpec,
        simulation_data: dict[str, Any],
    ) -> ValidationRunManifest:
        """Run reacting-flow validation.

        simulation_data keys expected:
        - For each metric in dataset.metrics: metric_id → simulated value
        - '{metric_id}_measured' → measured value
        - 'reacting_flow_provenance': optional provenance dict
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

        solver_artifacts: dict[str, str] = {}
        provenance = simulation_data.get("reacting_flow_provenance", {})
        if provenance:
            solver_artifacts["reacting_flow_provenance"] = str(provenance)

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
        """Structure acceptance outputs — temperature, species, velocity, scalar dissipation."""
        outputs: dict[str, Any] = {
            "temperature_comparisons": {},
            "species_comparisons": {},
            "velocity_comparisons": {},
            "scalar_dissipation_diagnostics": {},
        }

        for r in run_manifest.metric_results:
            entry = {
                "measured": r.measured_value,
                "simulated": r.simulated_value,
                "error": r.error,
                "passed": r.passed,
            }
            if "temperature" in r.metric_id:
                outputs["temperature_comparisons"][r.metric_id] = entry
            elif "species" in r.metric_id:
                outputs["species_comparisons"][r.metric_id] = entry
            elif "velocity" in r.metric_id:
                outputs["velocity_comparisons"][r.metric_id] = entry
            elif "scalar_dissipation" in r.metric_id:
                outputs["scalar_dissipation_diagnostics"][r.metric_id] = entry

        outputs["reacting_flow_provenance"] = run_manifest.solver_artifacts.get(
            "reacting_flow_provenance", ""
        )
        return outputs
