"""Closed-cylinder combustion / expansion regime runner.

Validates against engine pressure-trace / optical / combustion datasets.
Four-stroke data is acceptable here if it is the best available
closed-cylinder source.
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

CLOSED_CYLINDER_METRIC_CATEGORIES = frozenset(
    {
        "pressure_trace",
        "ca10",
        "ca50",
        "ca90",
        "apparent_heat_release",
        "work_imep",
        "peak_pressure",
        "peak_pressure_timing",
        "expansion_pressure_decay",
        "burn_duration",
        "ignition_delay_soc",
    }
)


class ClosedCylinderRunner(BaseRegimeRunner):
    """Runner for closed-cylinder combustion / expansion regime.

    Validates:
    - Ignition delay / SOC
    - Pressure rise
    - Apparent heat-release trend
    - Burn duration
    - Peak pressure and timing
    - Expansion pressure decay
    """

    @property
    def regime_name(self) -> str:
        return "closed_cylinder"

    def run(
        self,
        dataset: ValidationDatasetManifest,
        case_spec: ValidationCaseSpec,
        simulation_data: dict[str, Any],
    ) -> ValidationRunManifest:
        """Run closed-cylinder validation.

        simulation_data keys expected:
        - For each metric in dataset.metrics: metric_id → simulated value
        - '{metric_id}_measured' → measured value
        - 'closed_cylinder_provenance': optional provenance dict
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
        provenance = simulation_data.get("closed_cylinder_provenance", {})
        if provenance:
            solver_artifacts["closed_cylinder_provenance"] = str(provenance)

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
        """Structure acceptance outputs for combustion/expansion validation."""
        outputs: dict[str, Any] = {
            "pressure_trace_comparisons": {},
            "burn_metrics": {},
            "heat_release_comparisons": {},
            "work_comparisons": {},
        }

        for r in run_manifest.metric_results:
            entry = {
                "measured": r.measured_value,
                "simulated": r.simulated_value,
                "error": r.error,
                "passed": r.passed,
            }
            if "pressure" in r.metric_id or "expansion" in r.metric_id:
                outputs["pressure_trace_comparisons"][r.metric_id] = entry
            elif any(k in r.metric_id for k in ("ca10", "ca50", "ca90", "burn", "ignition", "soc")):
                outputs["burn_metrics"][r.metric_id] = entry
            elif "heat_release" in r.metric_id:
                outputs["heat_release_comparisons"][r.metric_id] = entry
            elif "work" in r.metric_id or "imep" in r.metric_id:
                outputs["work_comparisons"][r.metric_id] = entry

        outputs["closed_cylinder_provenance"] = run_manifest.solver_artifacts.get(
            "closed_cylinder_provenance", ""
        )
        return outputs
