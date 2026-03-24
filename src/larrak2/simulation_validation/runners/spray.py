"""Spray / evaporation / premixing regime runner.

Primary validation source for v1 gasoline path: ECN Spray G.
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

SPRAY_METRIC_CATEGORIES = frozenset(
    {
        "liquid_penetration",
        "vapor_penetration",
        "droplet_size",
        "droplet_velocity",
        "gas_velocity",
    }
)


class SprayRunner(BaseRegimeRunner):
    """Runner for spray / evaporation / premixing regime.

    Validates:
    - Injector abstraction
    - Breakup / droplet model choice
    - Evaporation trend
    - Entrainment / vapor penetration behaviour
    - Gas-velocity and droplet-size behaviour where data exists
    """

    @property
    def regime_name(self) -> str:
        return "spray"

    def run(
        self,
        dataset: ValidationDatasetManifest,
        case_spec: ValidationCaseSpec,
        simulation_data: dict[str, Any],
    ) -> ValidationRunManifest:
        """Run spray validation.

        simulation_data keys expected:
        - For each metric in dataset.metrics: metric_id → simulated value
        - '{metric_id}_measured' → measured value
        - 'spray_provenance': optional provenance dict
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
        provenance = simulation_data.get("spray_provenance", {})
        if provenance:
            solver_artifacts["spray_case_provenance"] = str(provenance)

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
        """Structure acceptance outputs for reporting."""
        outputs: dict[str, Any] = {
            "liquid_penetration": {},
            "vapor_penetration": {},
            "droplet_comparisons": {},
            "gas_velocity_comparison": {},
        }

        for r in run_manifest.metric_results:
            entry = {
                "measured": r.measured_value,
                "simulated": r.simulated_value,
                "error": r.error,
                "passed": r.passed,
            }
            if "liquid_penetration" in r.metric_id:
                outputs["liquid_penetration"][r.metric_id] = entry
            elif "vapor_penetration" in r.metric_id:
                outputs["vapor_penetration"][r.metric_id] = entry
            elif "droplet" in r.metric_id:
                outputs["droplet_comparisons"][r.metric_id] = entry
            elif "gas_velocity" in r.metric_id:
                outputs["gas_velocity_comparison"][r.metric_id] = entry

        outputs["spray_case_provenance"] = run_manifest.solver_artifacts.get(
            "spray_case_provenance", ""
        )
        return outputs
