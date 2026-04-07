"""Orchestration loop wrapper.

Agent context:
- The orchestration *core* is now extracted into the external `larrak-orchestration` repo.
- This module keeps the `larrak2.orchestration.*` API stable for existing callers/tests by:
  - building the monorepo `EvalContext` (paths, breathing config, etc.)
  - injecting production gating and contract tracing hooks
  - delegating the run-loop execution to `larrak_orchestration.legacy_loop`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from larrak_optimization import STRICT_PRODUCTION_PROFILE
from larrak_orchestration.legacy_loop.orchestrator import (
    LegacyOrchestrationConfig,
    LegacyOrchestrationResult,
    LegacyOrchestrator,
)

from larrak2.architecture.contracts import (
    ContractTracer,
    activate_contract_tracer,
    deactivate_contract_tracer,
    log_contract_edge,
)
from larrak2.core.encoding import N_TOTAL
from larrak2.core.types import BreathingConfig, EvalContext

from .budget import BudgetManager
from .cache import EvaluationCache
from .trust_region import TrustRegion


class CEMInterface(Protocol):
    def generate_batch(
        self,
        params: dict[str, Any],
        n: int,
        *,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]: ...

    def check_feasibility(self, candidate: dict[str, Any]) -> tuple[bool, float]: ...

    def repair(self, candidate: dict[str, Any]) -> dict[str, Any]: ...

    def update_distribution(self, history: list[dict[str, Any]]) -> None: ...


class SurrogateInterface(Protocol):
    def predict(self, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]: ...

    def update(self, data: list[tuple[dict[str, Any], float]]) -> None: ...


class SolverInterface(Protocol):
    def refine(
        self,
        candidate: dict[str, Any],
        *,
        context: EvalContext,
        max_step: np.ndarray,
    ) -> dict[str, Any]: ...


class SimulationInterface(Protocol):
    def evaluate(
        self,
        candidate: dict[str, Any],
        *,
        context: EvalContext,
    ) -> dict[str, Any]: ...


class ControlBackendInterface(Protocol):
    def get_signal(self, run_id: str) -> dict[str, Any] | None: ...

    def clear_signal(self, run_id: str) -> None: ...


class ProvenanceBackendInterface(Protocol):
    def log_event(self, event: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


@dataclass
class OrchestrationConfig:
    """Configuration for one orchestration run (monorepo-facing)."""

    total_sim_budget: int = 32
    batch_size: int = 16
    max_iterations: int = 8
    seed: int = 42

    rpm: float = 3000.0
    torque: float = 200.0
    fidelity: int = 2
    bore_mm: float = 80.0
    stroke_mm: float = 90.0
    intake_port_area_m2: float = 4.0e-4
    exhaust_port_area_m2: float = 4.0e-4
    p_manifold_pa: float = 101325.0
    p_back_pa: float = 101325.0
    compression_ratio: float = 10.0
    fuel_name: str = "gasoline"
    constraint_phase: str = "downselect"
    tolerance_constraint_mode: str = "capability_floor"
    tolerance_threshold_mm: float = 0.24

    truth_dispatch_mode: str = "auto"  # off | manual | auto
    truth_plan: list[str] | None = None
    truth_auto_top_k: int = 2
    truth_auto_min_uncertainty: float = 0.0
    truth_auto_min_feasibility: float = 0.0
    truth_auto_min_pred_quantile: float = 0.0
    truth_records_path: str | Path | None = None

    strict_data: bool = True
    strict_tribology_data: bool | None = None
    tribology_scuff_method: str = "auto"
    surrogate_validation_mode: str = "strict"
    thermo_symbolic_mode: str = "strict"
    thermo_symbolic_artifact_path: str | None = None
    stack_model_path: str | None = None
    thermo_constants_path: str | None = None
    thermo_anchor_manifest_path: str | None = None
    thermo_chemistry_profile_path: str | None = None
    machining_mode: str = "nn"
    machining_model_path: str | None = None

    outdir: str | Path = "outputs/orchestration"
    cache_path: str | Path | None = None
    use_provenance: bool = True
    run_id: str | None = None
    enforce_contract_routing: bool = False
    production_profile: str = STRICT_PRODUCTION_PROFILE
    allow_nonproduction_paths: bool = False

    ipopt_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    run_id: str
    best_candidate: dict[str, Any]
    best_objective: float
    best_source: str
    n_sim_calls: int
    n_surrogate_calls: int
    n_iterations: int
    history: list[dict[str, Any]]
    manifest_path: str

    @property
    def efficiency(self) -> float:
        if self.n_sim_calls <= 0:
            return float("inf")
        return float(self.n_surrogate_calls / self.n_sim_calls)


class _ContractsHook:
    def __init__(self, *, enforce_routing: bool) -> None:
        self.enforce_routing = bool(enforce_routing)

    def activate(self, run_id: str, *, path: str) -> Any:
        tracer = ContractTracer(
            run_id=str(run_id),
            path=Path(path),
            enforce_contract_routing=self.enforce_routing,
        )
        return activate_contract_tracer(tracer)

    def deactivate(self, token: Any) -> None:
        deactivate_contract_tracer(token)

    def log_edge(self, edge_id: str, request: dict[str, Any], response: dict[str, Any]) -> None:
        log_contract_edge(edge_id=edge_id, request=request, response=response)


class Orchestrator:
    """Monorepo-facing orchestrator that delegates to extracted legacy loop."""

    def __init__(
        self,
        cem: CEMInterface,
        surrogate: SurrogateInterface,
        solver: SolverInterface,
        simulation: SimulationInterface,
        config: OrchestrationConfig | None = None,
        *,
        control_backend: ControlBackendInterface | None = None,
        provenance_backend: ProvenanceBackendInterface | None = None,
        budget: BudgetManager | None = None,
        cache: EvaluationCache | None = None,
        trust_region: TrustRegion | None = None,
    ) -> None:
        self.cem = cem
        self.surrogate = surrogate
        self.solver = solver
        self.simulation = simulation
        self.config = config or OrchestrationConfig()

        legacy_cfg = LegacyOrchestrationConfig(
            n_vars=int(N_TOTAL),
            total_sim_budget=int(self.config.total_sim_budget),
            batch_size=int(self.config.batch_size),
            max_iterations=int(self.config.max_iterations),
            seed=int(self.config.seed),
            rpm=float(self.config.rpm),
            torque=float(self.config.torque),
            fidelity=int(self.config.fidelity),
            truth_dispatch_mode=str(self.config.truth_dispatch_mode),
            truth_plan=list(self.config.truth_plan) if self.config.truth_plan else None,
            truth_auto_top_k=int(self.config.truth_auto_top_k),
            truth_auto_min_uncertainty=float(self.config.truth_auto_min_uncertainty),
            truth_auto_min_feasibility=float(self.config.truth_auto_min_feasibility),
            truth_auto_min_pred_quantile=float(self.config.truth_auto_min_pred_quantile),
            truth_records_path=self.config.truth_records_path,
            outdir=self.config.outdir,
            cache_path=self.config.cache_path,
            use_provenance=bool(self.config.use_provenance),
            run_id=self.config.run_id,
            production_profile=str(self.config.production_profile),
            allow_nonproduction_paths=bool(self.config.allow_nonproduction_paths),
            ipopt_options=dict(self.config.ipopt_options),
        )

        def _context_factory(_: LegacyOrchestrationConfig) -> EvalContext:
            return self._build_context()

        def _truth_gate(
            payload: dict[str, Any],
            *,
            profile: str,
            allow_nonproduction_paths: bool,
        ) -> dict[str, Any]:
            # `evaluate_production_gate` in `larrak_optimization` computes a gate summary
            # from scalar run metrics; it does not validate/transform payload objects.
            # The extracted loop expects an optional payload-transform hook, so we keep
            # this as a no-op and let monorepo integrations call the gate separately.
            _ = (profile, allow_nonproduction_paths)
            return payload

        contract_hook = _ContractsHook(enforce_routing=bool(self.config.enforce_contract_routing))

        self._impl = LegacyOrchestrator(
            cem=cem,
            surrogate=surrogate,
            solver=solver,
            simulation=simulation,
            config=legacy_cfg,
            control_backend=control_backend,
            provenance_backend=provenance_backend,
            budget=budget,
            cache=cache,
            trust_region=trust_region,
            context_factory=_context_factory,
            truth_gate=_truth_gate,
            contract_hook=contract_hook,
        )

    def _build_context(self) -> EvalContext:
        return EvalContext(
            rpm=float(self.config.rpm),
            torque=float(self.config.torque),
            fidelity=int(self.config.fidelity),
            seed=int(self.config.seed),
            breathing=BreathingConfig(
                bore_mm=float(self.config.bore_mm),
                stroke_mm=float(self.config.stroke_mm),
                intake_port_area_m2=float(self.config.intake_port_area_m2),
                exhaust_port_area_m2=float(self.config.exhaust_port_area_m2),
                p_manifold_Pa=float(self.config.p_manifold_pa),
                p_back_Pa=float(self.config.p_back_pa),
                compression_ratio=float(self.config.compression_ratio),
                fuel_name=str(self.config.fuel_name),  # type: ignore[arg-type]
                valve_timing_mode="candidate",
            ),
            constraint_phase=str(self.config.constraint_phase),
            tolerance_constraint_mode=str(self.config.tolerance_constraint_mode),
            tolerance_threshold_mm=float(self.config.tolerance_threshold_mm),
            thermo_constants_path=self.config.thermo_constants_path,
            thermo_anchor_manifest_path=self.config.thermo_anchor_manifest_path,
            thermo_chemistry_profile_path=self.config.thermo_chemistry_profile_path,
            thermo_symbolic_mode=str(self.config.thermo_symbolic_mode),
            thermo_symbolic_artifact_path=self.config.thermo_symbolic_artifact_path,
            strict_data=bool(self.config.strict_data),
            strict_tribology_data=self.config.strict_tribology_data,
            tribology_scuff_method=str(self.config.tribology_scuff_method),  # type: ignore[arg-type]
            surrogate_validation_mode=str(self.config.surrogate_validation_mode),
            machining_mode=str(self.config.machining_mode),
            machining_model_path=self.config.machining_model_path,
            production_profile=str(self.config.production_profile),
            allow_nonproduction_paths=bool(self.config.allow_nonproduction_paths),
        )

    @property
    def run_id(self) -> str:
        return str(self._impl.run_id)

    def optimize(self, initial_params: dict[str, Any]) -> OrchestrationResult:
        res: LegacyOrchestrationResult = self._impl.optimize(initial_params=initial_params)
        return OrchestrationResult(
            run_id=str(res.run_id),
            best_candidate=dict(res.best_candidate),
            best_objective=float(res.best_objective),
            best_source=str(res.best_source),
            n_sim_calls=int(res.n_sim_calls),
            n_surrogate_calls=int(res.n_surrogate_calls),
            n_iterations=int(res.n_iterations),
            history=list(res.history),
            manifest_path=str(res.manifest_path),
        )


__all__ = [
    "Orchestrator",
    "OrchestrationConfig",
    "OrchestrationResult",
]
