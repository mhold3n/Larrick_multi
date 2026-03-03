"""Core backend orchestration loop."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from larrak2.core.encoding import N_TOTAL
from larrak2.core.types import EvalContext

from .budget import BudgetManager
from .cache import EvaluationCache
from .trust_region import TrustRegion

LOGGER = logging.getLogger(__name__)


class CEMInterface(Protocol):
    """Feasibility/generation interface used by the orchestrator."""

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
    """Surrogate prediction/update interface."""

    def predict(self, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]: ...

    def update(self, data: list[tuple[dict[str, Any], float]]) -> None: ...


class SolverInterface(Protocol):
    """Local-refinement interface."""

    def refine(
        self,
        candidate: dict[str, Any],
        *,
        context: EvalContext,
        max_step: np.ndarray,
    ) -> dict[str, Any]: ...


class SimulationInterface(Protocol):
    """Truth-evaluation interface."""

    def evaluate(
        self,
        candidate: dict[str, Any],
        *,
        context: EvalContext,
    ) -> dict[str, Any]: ...


class ControlBackendInterface(Protocol):
    """Control signal backend interface."""

    def get_signal(self, run_id: str) -> dict[str, Any] | None: ...

    def clear_signal(self, run_id: str) -> None: ...


class ProvenanceBackendInterface(Protocol):
    """Provenance backend interface."""

    def log_event(self, event: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class _NullControlBackend:
    def get_signal(self, run_id: str) -> dict[str, Any] | None:  # noqa: ARG002
        return None

    def clear_signal(self, run_id: str) -> None:  # noqa: ARG002
        return None


class _NullProvenanceBackend:
    path: Path | None = None

    def log_event(self, event: dict[str, Any]) -> None:  # noqa: ARG002
        return None

    def close(self) -> None:
        return None


@dataclass
class OrchestrationConfig:
    """Configuration for one orchestration run."""

    total_sim_budget: int = 32
    batch_size: int = 16
    max_iterations: int = 8
    seed: int = 42

    rpm: float = 3000.0
    torque: float = 200.0
    fidelity: int = 0
    constraint_phase: str = "explore"
    tolerance_constraint_mode: str = "capability_floor"
    tolerance_threshold_mm: float = 0.24

    truth_dispatch_mode: str = "off"  # off | manual
    truth_plan: list[str] | None = None

    outdir: str | Path = "outputs/orchestration"
    cache_path: str | Path | None = None
    use_provenance: bool = True
    run_id: str | None = None

    ipopt_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.total_sim_budget) < 0:
            raise ValueError("total_sim_budget must be >= 0")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be > 0")
        if self.truth_dispatch_mode not in {"off", "manual"}:
            raise ValueError(
                "truth_dispatch_mode must be one of {'off', 'manual'}, "
                f"got {self.truth_dispatch_mode!r}"
            )
        if self.truth_dispatch_mode == "manual" and not self.truth_plan:
            raise ValueError("truth_plan is required when truth_dispatch_mode='manual'")


@dataclass
class OrchestrationResult:
    """Result from an orchestration run."""

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


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class Orchestrator:
    """Backend orchestration loop (generate -> predict -> refine -> select -> log)."""

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

        self.outdir = Path(self.config.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.outdir / "orchestrate_manifest.json"

        self.budget = budget or BudgetManager(total_sim_calls=int(self.config.total_sim_budget))
        self.cache = cache or EvaluationCache(persist_path=self.config.cache_path)
        self.trust_region = trust_region or TrustRegion(n_vars=N_TOTAL)

        self.control = control_backend
        if self.control is None:
            try:
                from .backends.control_file import FileControlBackend

                self.control = FileControlBackend(path=self.outdir / "control_signal.json")
            except Exception:
                self.control = _NullControlBackend()

        self.provenance = provenance_backend
        if self.provenance is None:
            if self.config.use_provenance:
                try:
                    from .backends.provenance_jsonl import JSONLProvenanceBackend

                    self.provenance = JSONLProvenanceBackend(
                        path=self.outdir / "provenance_events.jsonl"
                    )
                except Exception:
                    self.provenance = _NullProvenanceBackend()
            else:
                self.provenance = _NullProvenanceBackend()

        self._history: list[dict[str, Any]] = []
        self._iterations: list[dict[str, Any]] = []
        self._best_candidate: dict[str, Any] | None = None
        self._best_objective: float = float("-inf")
        self._best_source: str = "none"
        self._n_surrogate_calls = 0
        self._candidate_counter = 0
        self._run_started_at = 0.0
        self._run_id = self.config.run_id or uuid.uuid4().hex[:12]
        self._truth_plan_tokens = self._normalize_truth_plan(self.config.truth_plan)

    @property
    def run_id(self) -> str:
        return self._run_id

    def _normalize_truth_plan(self, truth_plan: list[str] | None) -> set[str]:
        tokens: set[str] = set()
        if truth_plan is None:
            return tokens
        for item in truth_plan:
            text = str(item).strip()
            if text:
                tokens.add(text)
        return tokens

    def _candidate_key(self, candidate: dict[str, Any], iteration: int, idx: int) -> str:
        if "id" in candidate:
            return str(candidate["id"])
        return f"{int(iteration)}:{int(idx)}"

    def _candidate_allowed_for_manual_truth(
        self,
        candidate: dict[str, Any],
        iteration: int,
        local_idx: int,
    ) -> bool:
        if self.config.truth_dispatch_mode != "manual":
            return False
        if not self._truth_plan_tokens:
            return False
        options = {
            str(candidate.get("id", "")),
            str(candidate.get("global_index", "")),
            f"{int(iteration)}:{int(local_idx)}",
            str(int(local_idx)),
        }
        options.discard("")
        return any(token in self._truth_plan_tokens for token in options)

    def _extract_objective(self, payload: dict[str, Any]) -> float:
        if "objective" in payload:
            try:
                return float(payload["objective"])
            except (TypeError, ValueError):
                return float("-inf")
        if "score" in payload:
            try:
                return float(payload["score"])
            except (TypeError, ValueError):
                return float("-inf")
        return float("-inf")

    def _handle_control_signal(self) -> bool:
        signal_data = self.control.get_signal(self._run_id) if self.control else None
        if not signal_data:
            return False
        signal = str(signal_data.get("signal", "")).upper()
        if signal == "STOP":
            LOGGER.warning("Received STOP control signal for run %s", self._run_id)
            if self.control:
                self.control.clear_signal(self._run_id)
            return True
        if signal == "PAUSE":
            LOGGER.info("Received PAUSE signal for run %s", self._run_id)
            while True:
                time.sleep(1.0)
                signal_data = self.control.get_signal(self._run_id) if self.control else None
                if not signal_data:
                    continue
                next_signal = str(signal_data.get("signal", "")).upper()
                if next_signal == "STOP":
                    if self.control:
                        self.control.clear_signal(self._run_id)
                    return True
                if next_signal in {"RESUME", "PAUSE"}:
                    if self.control:
                        self.control.clear_signal(self._run_id)
                    return False
        if self.control:
            self.control.clear_signal(self._run_id)
        return False

    def _emit_event(self, event_type: str, **payload: Any) -> None:
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "run_id": self._run_id,
            **payload,
        }
        try:
            self.provenance.log_event(_json_safe(event))
        except Exception as exc:
            LOGGER.warning("Failed to log provenance event '%s': %s", event_type, exc)

    def _build_context(self) -> EvalContext:
        return EvalContext(
            rpm=float(self.config.rpm),
            torque=float(self.config.torque),
            fidelity=int(self.config.fidelity),
            seed=int(self.config.seed),
            constraint_phase=str(self.config.constraint_phase),
            tolerance_constraint_mode=str(self.config.tolerance_constraint_mode),
            tolerance_threshold_mm=float(self.config.tolerance_threshold_mm),
        )

    def _clone_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in candidate.items():
            if isinstance(value, np.ndarray):
                out[key] = np.array(value, dtype=np.float64, copy=True)
            else:
                out[key] = value
        return out

    def _refine_candidates(
        self,
        candidates: list[dict[str, Any]],
        predictions: np.ndarray,
        uncertainty: np.ndarray,
        *,
        context: EvalContext,
    ) -> list[dict[str, Any]]:
        n = int(len(candidates))
        if n == 0:
            return []

        preds = np.asarray(predictions, dtype=np.float64).reshape(-1)
        unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
        if preds.size != n:
            preds = np.resize(preds, n)
        if unc.size != n:
            unc = np.resize(unc, n)

        n_refine = min(n, max(1, n // 3))
        top_indices = sorted(range(n), key=lambda i: (-float(preds[i]), int(i)))[:n_refine]
        refined = [self._clone_candidate(c) for c in candidates]

        for idx in top_indices:
            current = self._clone_candidate(refined[idx])
            try:
                max_step = self.trust_region.bound_step(
                    proposed_step=np.ones(N_TOTAL, dtype=np.float64) * 0.1,
                    uncertainty=float(abs(unc[idx])),
                )
                updated = self.solver.refine(current, context=context, max_step=max_step)
                updated["id"] = current.get("id", f"refined-{idx}")
                updated.setdefault("global_index", current.get("global_index"))
                refined[idx] = updated
            except Exception as exc:
                current["refine_error"] = str(exc)
                refined[idx] = current
        return refined

    def optimize(
        self,
        initial_params: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> OrchestrationResult:
        """Execute orchestration loop."""
        if run_id:
            self._run_id = str(run_id)

        self._run_started_at = time.time()
        params = dict(initial_params or {})
        rng = np.random.default_rng(int(self.config.seed))
        context = self._build_context()

        self._emit_event("run_start", config=asdict(self.config))
        stop_requested = False

        for iteration in range(1, int(self.config.max_iterations) + 1):
            if self.budget.exhausted():
                break

            if self._handle_control_signal():
                stop_requested = True
                break

            candidates = self.cem.generate_batch(params, n=int(self.config.batch_size), rng=rng)
            if not candidates:
                break

            enriched: list[dict[str, Any]] = []
            feasibility = np.zeros(len(candidates), dtype=np.float64)
            for local_idx, candidate in enumerate(candidates):
                self._candidate_counter += 1
                c = self._clone_candidate(candidate)
                c["iteration"] = int(iteration)
                c["local_index"] = int(local_idx)
                c["global_index"] = int(self._candidate_counter)
                c["id"] = self._candidate_key(c, iteration, local_idx)

                is_feasible, score = self.cem.check_feasibility(c)
                if not is_feasible:
                    c = self.cem.repair(c)
                    is_feasible, score = self.cem.check_feasibility(c)
                c["feasible"] = bool(is_feasible)
                c["feasibility_score"] = float(score)

                enriched.append(c)
                feasibility[local_idx] = float(score)

            pred, unc = self.surrogate.predict(enriched)
            self._n_surrogate_calls += len(enriched)

            refined = self._refine_candidates(enriched, pred, unc, context=context)
            ref_pred, ref_unc = self.surrogate.predict(refined)
            self._n_surrogate_calls += len(refined)

            selected = self.budget.select(
                refined,
                ref_pred,
                ref_unc,
                cem_feasibility=feasibility,
                batch_size=min(int(self.config.batch_size), len(refined)),
            )

            truth_indices: list[int] = []
            if self.config.truth_dispatch_mode == "manual":
                for idx in selected:
                    if self._candidate_allowed_for_manual_truth(refined[idx], iteration, idx):
                        truth_indices.append(int(idx))

            truth_data: list[tuple[dict[str, Any], float]] = []
            truth_records: list[dict[str, Any]] = []

            if truth_indices:
                for idx in truth_indices:
                    cand = refined[idx]
                    payload, was_cached = self.cache.get_or_compute(
                        cand,
                        lambda item: self.simulation.evaluate(item, context=context),
                    )
                    objective = self._extract_objective(
                        payload if isinstance(payload, dict) else {}
                    )
                    payload_dict = (
                        payload if isinstance(payload, dict) else {"objective": objective}
                    )

                    truth_records.append(
                        {
                            "idx": int(idx),
                            "candidate_id": str(cand.get("id", idx)),
                            "objective": float(objective),
                            "cached": bool(was_cached),
                            "payload": _json_safe(payload_dict),
                        }
                    )
                    truth_data.append((cand, float(objective)))

                    self._history.append(
                        {
                            "iteration": int(iteration),
                            "candidate_id": str(cand.get("id", idx)),
                            "objective": float(objective),
                            "source": "truth",
                        }
                    )

                    if np.isfinite(objective) and objective > self._best_objective:
                        self._best_objective = float(objective)
                        self._best_candidate = self._clone_candidate(cand)
                        self._best_source = "truth"

            best_ref_idx = (
                int(np.argmax(ref_pred))
                if isinstance(ref_pred, np.ndarray) and ref_pred.size > 0
                else None
            )
            if best_ref_idx is not None and self._best_source != "truth":
                best_pred = float(np.asarray(ref_pred, dtype=np.float64)[best_ref_idx])
                if np.isfinite(best_pred) and best_pred > self._best_objective:
                    self._best_objective = best_pred
                    self._best_candidate = self._clone_candidate(refined[best_ref_idx])
                    self._best_source = "surrogate"

            if truth_data:
                self.surrogate.update(truth_data)
                first_idx = truth_indices[0]
                try:
                    pred_val = float(np.asarray(ref_pred, dtype=np.float64)[first_idx])
                    act_val = float(truth_data[0][1])
                    unc_val = float(np.asarray(ref_unc, dtype=np.float64)[first_idx])
                    self.trust_region.update(
                        predicted_improvement=pred_val,
                        actual_improvement=act_val,
                        uncertainty_at_step=unc_val,
                    )
                except Exception:
                    pass

            self.cem.update_distribution(self._history)

            selected_payload = [
                {
                    "idx": int(idx),
                    "candidate_id": str(refined[idx].get("id", idx)),
                    "predicted_objective": float(np.asarray(ref_pred, dtype=np.float64)[idx]),
                    "uncertainty": float(np.asarray(ref_unc, dtype=np.float64)[idx]),
                }
                for idx in selected
            ]

            iter_summary = {
                "iteration": int(iteration),
                "n_generated": int(len(enriched)),
                "n_selected": int(len(selected)),
                "n_truth_evaluated": int(len(truth_indices)),
                "selected_candidates": _json_safe(selected_payload),
                "truth_results": _json_safe(truth_records),
                "best_objective": float(self._best_objective),
                "best_source": str(self._best_source),
                "budget_remaining": int(self.budget.remaining()),
            }
            self._iterations.append(iter_summary)
            self._emit_event("iteration_end", **iter_summary)

        self.cache.save_to_disk()
        self._emit_event(
            "run_end",
            stopped=bool(stop_requested),
            n_iterations=int(len(self._iterations)),
            best_objective=float(self._best_objective),
            best_source=str(self._best_source),
            n_sim_calls=int(self.budget.state.used),
            n_surrogate_calls=int(self._n_surrogate_calls),
        )
        try:
            self.provenance.close()
        except Exception:
            pass

        return self._write_manifest_and_result(stop_requested=stop_requested)

    def _write_manifest_and_result(self, *, stop_requested: bool) -> OrchestrationResult:
        elapsed = float(time.time() - self._run_started_at)
        provenance_path = str(getattr(self.provenance, "path", "")) if self.provenance else ""

        manifest: dict[str, Any] = {
            "workflow": "orchestrate",
            "run_id": self._run_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._run_started_at)),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": elapsed,
            "stopped": bool(stop_requested),
            "config": _json_safe(asdict(self.config)),
            "backend": {
                "control": type(self.control).__name__ if self.control else "None",
                "provenance": type(self.provenance).__name__ if self.provenance else "None",
            },
            "result": {
                "best_objective": float(self._best_objective),
                "best_source": str(self._best_source),
                "best_candidate": _json_safe(self._best_candidate or {}),
                "n_iterations": int(len(self._iterations)),
                "n_sim_calls": int(self.budget.state.used),
                "n_surrogate_calls": int(self._n_surrogate_calls),
                "efficiency": float(self._n_surrogate_calls / max(1, self.budget.state.used)),
            },
            "iterations": _json_safe(self._iterations),
            "stats": {
                "budget": _json_safe(self.budget.get_statistics()),
                "cache": _json_safe(self.cache.get_statistics()),
                "trust_region": _json_safe(self.trust_region.get_statistics()),
            },
            "files": {
                "orchestrate_manifest": str(self.manifest_path),
                "provenance_events": provenance_path,
            },
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        result = OrchestrationResult(
            run_id=self._run_id,
            best_candidate=self._best_candidate or {},
            best_objective=float(self._best_objective),
            best_source=str(self._best_source),
            n_sim_calls=int(self.budget.state.used),
            n_surrogate_calls=int(self._n_surrogate_calls),
            n_iterations=int(len(self._iterations)),
            history=_json_safe(self._history),
            manifest_path=str(self.manifest_path),
        )
        return result


__all__ = [
    "CEMInterface",
    "ControlBackendInterface",
    "OrchestrationConfig",
    "OrchestrationResult",
    "Orchestrator",
    "ProvenanceBackendInterface",
    "SimulationInterface",
    "SolverInterface",
    "SurrogateInterface",
]
