"""Chemistry-only regime runner.

Validates Cantera mechanism choice and reduced fuel-surrogate chemistry
fidelity against shock-tube ignition delay, RCM ignition delay,
JSR/flow-reactor species evolution, and laminar flame speed datasets.
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

# Expected metric categories for chemistry regime
CHEMISTRY_METRIC_CATEGORIES = frozenset(
    {
        "ignition_delay",
        "species_profile",
        "flame_speed",
    }
)

# Default data sources for v1 gasoline path
CHEMISTRY_DATA_SOURCES = [
    "shock_tube_ignition_delay",
    "rcm_ignition_delay",
    "jsr_flow_reactor_species",
    "laminar_flame_speed",
]


class ChemistryRunner(BaseRegimeRunner):
    """Runner for chemistry-only regime validation.

    Validates:
    - Cantera mechanism choice
    - Reduced fuel-surrogate chemistry fidelity
    - Low/intermediate-temperature ignition behaviour
    - Species pathway credibility before any CFD coupling
    """

    @property
    def regime_name(self) -> str:
        return "chemistry"

    def run(
        self,
        dataset: ValidationDatasetManifest,
        case_spec: ValidationCaseSpec,
        simulation_data: dict[str, Any],
    ) -> ValidationRunManifest:
        """Run chemistry validation.

        simulation_data keys expected:
        - For each metric in dataset.metrics, a key matching metric_id with
          the simulated value (float).
        - 'mechanism_provenance': dict with mechanism metadata.
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

            # Extract measured value from dataset operating bounds / provenance
            measured_key = f"{spec.metric_id}_measured"
            measured = float(simulation_data.get(measured_key, 0.0))
            simulated = float(simulation_data[sim_key])

            result = evaluate_metric(spec, measured, simulated)
            results.append(result)

        solver_artifacts: dict[str, str] = {}
        provenance = simulation_data.get("mechanism_provenance", {})
        if provenance:
            solver_artifacts["mechanism_provenance"] = str(provenance)

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
        ignition_delay_errors = {}
        species_comparisons = {}
        flame_speed_comparisons = {}

        for r in run_manifest.metric_results:
            if "ignition_delay" in r.metric_id:
                ignition_delay_errors[r.metric_id] = {
                    "measured": r.measured_value,
                    "simulated": r.simulated_value,
                    "error": r.error,
                    "passed": r.passed,
                }
            elif "species" in r.metric_id:
                species_comparisons[r.metric_id] = {
                    "measured": r.measured_value,
                    "simulated": r.simulated_value,
                    "error": r.error,
                    "passed": r.passed,
                }
            elif "flame_speed" in r.metric_id:
                flame_speed_comparisons[r.metric_id] = {
                    "measured": r.measured_value,
                    "simulated": r.simulated_value,
                    "error": r.error,
                    "passed": r.passed,
                }

        return {
            "ignition_delay_error_table": ignition_delay_errors,
            "species_profile_comparisons": species_comparisons,
            "flame_speed_comparisons": flame_speed_comparisons,
            "mechanism_provenance": run_manifest.solver_artifacts.get("mechanism_provenance", ""),
        }
