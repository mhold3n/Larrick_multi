"""Contract spec + tracing utilities for architecture readiness diagnostics."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = "connection_contract_v1"
ENGINE_MODES = ("placeholder", "hybrid", "production")
EDGE_IDS = (
    "edge.decode_candidate",
    "edge.thermo.forward",
    "edge.gear.forward",
    "edge.machining.forward",
    "edge.realworld.forward",
    "edge.constraints.combine",
    "edge.objectives.assemble",
    "edge.surrogate.predict",
    "edge.solver.refine",
    "edge.truth.evaluate",
    "edge.lifetime.extract",
)

CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C = (
    "F",
    "G",
    "diag.thermo.valve_timing.intake_open_offset_from_bdc",
    "diag.thermo.valve_timing.intake_duration_deg",
    "diag.thermo.valve_timing.exhaust_open_offset_from_expansion_tdc",
    "diag.thermo.valve_timing.exhaust_duration_deg",
    "diag.thermo.valve_timing.intake_open_deg",
    "diag.thermo.valve_timing.intake_close_deg",
    "diag.thermo.valve_timing.exhaust_open_deg",
    "diag.thermo.valve_timing.exhaust_close_deg",
    "diag.thermo.valve_timing.overlap_deg",
    "diag.thermo.mixture_preparation.delivered_vapor_fraction",
    "diag.thermo.mixture_preparation.wall_film_fraction",
    "diag.thermo.mixture_preparation.mixture_inhomogeneity",
    "diag.thermo.ignition_stage.spark_timing_deg_from_compression_tdc",
    "diag.thermo.ignition_stage.preignition_margin",
    "diag.thermo.ignition_stage.ignitability_margin",
    "diag.thermo.ignition_stage.soc_deg",
    "diag.thermo.ignition_stage.ca10_deg",
    "diag.constraints",
    "diag.constraints[].name",
    "diag.constraints[].kind",
    "diag.constraints[].scaled",
    "diag.constraints[].scaled_raw",
    "diag.constraints[].feasible",
    "diag.realworld.lambda_min",
    "diag.realworld.scuff_margin_flash_C",
    "diag.realworld.scuff_margin_integral_C",
    "diag.realworld.micropitting_safety",
    "diag.realworld.material_temp_margin_C",
    "diag.realworld.life_damage.D_total",
    "diag.realworld.life_damage.life_damage_status",
    "diag.realworld.life_damage.life_damage_input_mode",
    "release_readiness.release_ready",
    "release_readiness.reasons",
)

CONNECTION_CONTRACT_V1: dict[str, Any] = {
    "contract_version": CONTRACT_VERSION,
    "engine_modes": list(ENGINE_MODES),
    "required_edges": [
        {
            "edge_id": "edge.decode_candidate",
            "required_request_keys": ["x", "ctx.fidelity"],
            "required_response_keys": ["candidate"],
        },
        {
            "edge_id": "edge.thermo.forward",
            "required_request_keys": [
                "ctx.fidelity",
                "ctx.rpm",
                "ctx.torque",
                "ctx.fuel_name",
                "thermo_params.compression_duration",
                "thermo_params.expansion_duration",
                "thermo_params.heat_release_center",
                "thermo_params.heat_release_width",
                "thermo_params.lambda_af",
                "thermo_params.intake_open_offset_from_bdc",
                "thermo_params.intake_duration_deg",
                "thermo_params.exhaust_open_offset_from_expansion_tdc",
                "thermo_params.exhaust_duration_deg",
                "thermo_params.spark_timing_deg_from_compression_tdc",
            ],
            "required_response_keys": [
                "efficiency",
                "diag.thermo_solver_status",
                "diag.thermo_model_version",
                "diag.valve_timing.intake_open_deg",
                "diag.valve_timing.intake_close_deg",
                "diag.valve_timing.exhaust_open_deg",
                "diag.valve_timing.exhaust_close_deg",
                "diag.valve_timing.overlap_deg",
                "diag.mixture_preparation.delivered_vapor_fraction",
                "diag.mixture_preparation.wall_film_fraction",
                "diag.mixture_preparation.mixture_inhomogeneity",
                "diag.ignition_stage.spark_timing_deg_from_compression_tdc",
                "diag.ignition_stage.preignition_margin",
                "diag.ignition_stage.ignitability_margin",
                "diag.ignition_stage.soc_deg",
                "diag.ignition_stage.ca10_deg",
            ],
        },
        {
            "edge_id": "edge.gear.forward",
            "required_request_keys": ["ctx.fidelity", "i_req_profile", "gear_params"],
            "required_response_keys": ["loss_total", "diag.hertz_stress_max"],
        },
        {
            "edge_id": "edge.machining.forward",
            "required_request_keys": ["ctx.machining_mode", "ctx.tolerance_threshold_mm"],
            "required_response_keys": ["tooling_cost", "tol_penalty"],
        },
        {
            "edge_id": "edge.realworld.forward",
            "required_request_keys": ["tribology_scuff_method", "strict_tribology_data"],
            "required_response_keys": [
                "lambda_min",
                "scuff_margin_flash_C",
                "scuff_margin_integral_C",
                "micropitting_safety",
                "material_temp_margin_C",
            ],
        },
        {
            "edge_id": "edge.constraints.combine",
            "required_request_keys": ["thermo_G", "gear_G", "constraint_phase"],
            "required_response_keys": ["G", "diag.constraints"],
        },
        {
            "edge_id": "edge.objectives.assemble",
            "required_request_keys": ["eta_comb", "eta_exp", "eta_gear", "life_damage_total"],
            "required_response_keys": ["F", "diag.objectives.names"],
        },
        {
            "edge_id": "edge.surrogate.predict",
            "required_request_keys": ["n_candidates"],
            "required_response_keys": ["predictions", "uncertainty"],
        },
        {
            "edge_id": "edge.solver.refine",
            "required_request_keys": ["candidate_id", "backend", "fidelity"],
            "required_response_keys": ["solver_success", "solver_backend"],
        },
        {
            "edge_id": "edge.truth.evaluate",
            "required_request_keys": ["candidate_id", "fidelity"],
            "required_response_keys": ["objective", "diag"],
        },
        {
            "edge_id": "edge.lifetime.extract",
            "required_request_keys": ["diag.realworld.life_damage"],
            "required_response_keys": [
                "life_damage_total",
                "life_damage_status",
                "life_damage_input_mode",
            ],
        },
    ],
    "critical_real_key_paths_stage_a_to_c": list(CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C),
}

_EDGE_SPEC = {
    str(item["edge_id"]): item for item in CONNECTION_CONTRACT_V1.get("required_edges", [])
}
_ACTIVE_TRACER: ContextVar[ContractTracer | None] = ContextVar(
    "larrak2_active_contract_tracer", default=None
)


def expected_engine_mode(*, edge_id: str, fidelity: int) -> str:
    if edge_id not in _EDGE_SPEC:
        raise KeyError(f"Unknown edge_id '{edge_id}'")
    if int(fidelity) == 0:
        return "placeholder"
    if int(fidelity) == 1:
        return "hybrid"
    if int(fidelity) == 2:
        return "production"
    raise ValueError(f"Unsupported fidelity '{fidelity}'")


def flatten_key_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            key_str = str(key)
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            paths.add(next_prefix)
            paths |= flatten_key_paths(item, next_prefix)
        return paths
    if isinstance(value, (list, tuple)):
        list_prefix = f"{prefix}[]" if prefix else "[]"
        paths.add(list_prefix)
        if value:
            paths |= flatten_key_paths(value[0], list_prefix)
        return paths
    if prefix:
        paths.add(prefix)
    return paths


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class _EdgeStats:
    seen: int = 0
    ok: int = 0
    error: int = 0
    routing_violations: int = 0
    modes_seen: set[str] | None = None
    missing_request_keys: set[str] | None = None
    missing_response_keys: set[str] | None = None

    def ensure_sets(self) -> None:
        if self.modes_seen is None:
            self.modes_seen = set()
        if self.missing_request_keys is None:
            self.missing_request_keys = set()
        if self.missing_response_keys is None:
            self.missing_response_keys = set()


class ContractTracer:
    """Edge-level contract tracer with deterministic JSONL + summary output."""

    def __init__(
        self,
        *,
        trace_path: str | Path,
        summary_path: str | Path | None = None,
        fidelity: int | None = None,
        enforce_routing: bool = False,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.summary_path = (
            Path(summary_path)
            if summary_path is not None
            else self.trace_path.parent / "contract_summary.json"
        )
        self.fidelity = fidelity
        self.enforce_routing = bool(enforce_routing)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        if self.trace_path.exists():
            self.trace_path.unlink()
        self._events = 0
        self._edge_stats: dict[str, _EdgeStats] = {edge: _EdgeStats() for edge in EDGE_IDS}
        self._routing_violations: list[dict[str, Any]] = []
        self._fout = self.trace_path.open("a", encoding="utf-8")
        self._closed = False

    def emit(
        self,
        *,
        edge_id: str,
        engine_mode: str,
        status: str,
        request_payload: dict[str, Any] | None = None,
        response_payload: dict[str, Any] | None = None,
        error_signature: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if edge_id not in _EDGE_SPEC:
            raise KeyError(f"Unknown edge_id '{edge_id}'")
        mode = str(engine_mode).strip().lower()
        if mode not in ENGINE_MODES:
            raise ValueError(f"engine_mode must be one of {ENGINE_MODES}, got {engine_mode!r}")
        status_norm = str(status).strip().lower()
        if status_norm not in {"ok", "error"}:
            raise ValueError(f"status must be 'ok' or 'error', got {status!r}")

        request_payload = dict(request_payload or {})
        response_payload = dict(response_payload or {})
        request_keys = flatten_key_paths(request_payload)
        response_keys = flatten_key_paths(response_payload)
        spec = _EDGE_SPEC[edge_id]
        missing_request = sorted(set(spec.get("required_request_keys", [])) - request_keys)
        missing_response = sorted(set(spec.get("required_response_keys", [])) - response_keys)

        expected = None
        routing_violation = False
        if self.fidelity is not None:
            expected = expected_engine_mode(edge_id=edge_id, fidelity=int(self.fidelity))
            routing_violation = expected != mode

        self._events += 1
        event_idx = int(self._events)
        event: dict[str, Any] = {
            "event_index": event_idx,
            "timestamp": time.time(),
            "contract_version": CONTRACT_VERSION,
            "edge_id": edge_id,
            "fidelity": self.fidelity,
            "engine_mode": mode,
            "expected_engine_mode": expected,
            "routing_violation": routing_violation,
            "status": status_norm,
            "error_signature": str(error_signature or ""),
            "request_keys": sorted(request_keys),
            "response_keys": sorted(response_keys),
            "missing_required_request_keys": missing_request,
            "missing_required_response_keys": missing_response,
            "request_payload": _json_safe(request_payload),
            "response_payload": _json_safe(response_payload),
            "metadata": _json_safe(metadata or {}),
        }
        self._fout.write(json.dumps(event, sort_keys=True) + "\n")
        self._fout.flush()

        stats = self._edge_stats[edge_id]
        stats.ensure_sets()
        stats.seen += 1
        if status_norm == "ok":
            stats.ok += 1
        else:
            stats.error += 1
        stats.modes_seen.add(mode)
        stats.missing_request_keys.update(missing_request)
        stats.missing_response_keys.update(missing_response)
        if routing_violation:
            stats.routing_violations += 1
            violation = {
                "event_index": event_idx,
                "edge_id": edge_id,
                "fidelity": self.fidelity,
                "expected_engine_mode": expected,
                "observed_engine_mode": mode,
            }
            self._routing_violations.append(violation)
            if self.enforce_routing:
                raise RuntimeError(
                    "Contract routing violation: "
                    f"edge={edge_id}, fidelity={self.fidelity}, expected={expected}, observed={mode}"
                )

    def summary_payload(self) -> dict[str, Any]:
        edge_payload: dict[str, Any] = {}
        for edge_id in EDGE_IDS:
            stats = self._edge_stats[edge_id]
            stats.ensure_sets()
            edge_payload[edge_id] = {
                "seen": int(stats.seen),
                "ok": int(stats.ok),
                "error": int(stats.error),
                "modes_seen": sorted(stats.modes_seen),
                "routing_violations": int(stats.routing_violations),
                "missing_required_request_keys": sorted(stats.missing_request_keys),
                "missing_required_response_keys": sorted(stats.missing_response_keys),
            }

        missing_edges = sorted([edge for edge in EDGE_IDS if edge_payload[edge]["seen"] <= 0])
        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "contract_version": CONTRACT_VERSION,
            "fidelity": self.fidelity,
            "trace_file": str(self.trace_path),
            "summary_file": str(self.summary_path),
            "required_edges": list(EDGE_IDS),
            "missing_edges": missing_edges,
            "total_events": int(self._events),
            "routing_violations": list(self._routing_violations),
            "edges": edge_payload,
            "critical_real_key_paths_stage_a_to_c": list(CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C),
        }

    def close(self) -> None:
        if self._closed:
            return
        summary = self.summary_payload()
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._fout.close()
        self._closed = True


def get_active_contract_tracer() -> ContractTracer | None:
    return _ACTIVE_TRACER.get()


def activate_contract_tracer(tracer: ContractTracer | None) -> Token[ContractTracer | None]:
    return _ACTIVE_TRACER.set(tracer)


def deactivate_contract_tracer(token: Token[ContractTracer | None]) -> None:
    _ACTIVE_TRACER.reset(token)


@contextmanager
def active_contract_tracer(
    *,
    trace_path: str | Path,
    summary_path: str | Path | None = None,
    fidelity: int | None = None,
    enforce_routing: bool = False,
) -> ContractTracer:
    tracer = ContractTracer(
        trace_path=trace_path,
        summary_path=summary_path,
        fidelity=fidelity,
        enforce_routing=enforce_routing,
    )
    token = activate_contract_tracer(tracer)
    try:
        yield tracer
    finally:
        deactivate_contract_tracer(token)
        tracer.close()


def log_contract_edge(
    *,
    edge_id: str,
    engine_mode: str,
    status: str,
    request_payload: dict[str, Any] | None = None,
    response_payload: dict[str, Any] | None = None,
    error_signature: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    tracer = get_active_contract_tracer()
    if tracer is None:
        return
    tracer.emit(
        edge_id=edge_id,
        engine_mode=engine_mode,
        status=status,
        request_payload=request_payload,
        response_payload=response_payload,
        error_signature=error_signature,
        metadata=metadata,
    )


__all__ = [
    "CONNECTION_CONTRACT_V1",
    "CONTRACT_VERSION",
    "CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C",
    "EDGE_IDS",
    "ContractTracer",
    "activate_contract_tracer",
    "active_contract_tracer",
    "deactivate_contract_tracer",
    "expected_engine_mode",
    "flatten_key_paths",
    "get_active_contract_tracer",
    "log_contract_edge",
]
