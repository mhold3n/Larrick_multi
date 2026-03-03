"""Unit tests for architecture contract spec and tracer behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.architecture.contracts import (
    CONNECTION_CONTRACT_V1,
    CONTRACT_VERSION,
    EDGE_IDS,
    ContractTracer,
)


def test_contract_spec_integrity() -> None:
    assert CONNECTION_CONTRACT_V1["contract_version"] == CONTRACT_VERSION
    required_edges = CONNECTION_CONTRACT_V1.get("required_edges", [])
    edge_ids = [str(item.get("edge_id", "")) for item in required_edges]
    assert sorted(edge_ids) == sorted(EDGE_IDS)
    for item in required_edges:
        assert isinstance(item.get("required_request_keys", []), list)
        assert isinstance(item.get("required_response_keys", []), list)


def test_contract_tracer_deterministic_error_capture(tmp_path: Path) -> None:
    trace_path = tmp_path / "contract_trace.jsonl"
    summary_path = tmp_path / "contract_summary.json"
    tracer = ContractTracer(
        trace_path=trace_path,
        summary_path=summary_path,
        fidelity=0,
        enforce_routing=False,
    )
    tracer.emit(
        edge_id="edge.decode_candidate",
        engine_mode="placeholder",
        status="ok",
        request_payload={"x": [0.1, 0.2], "ctx": {"fidelity": 0}},
        response_payload={"candidate": {"decoded": True}},
    )
    tracer.emit(
        edge_id="edge.truth.evaluate",
        engine_mode="placeholder",
        status="error",
        request_payload={"candidate_id": "abc", "fidelity": 0},
        response_payload={},
        error_signature="sim_failed",
    )
    tracer.close()

    rows = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(rows) == 2
    assert rows[0]["event_index"] == 1
    assert rows[1]["event_index"] == 2
    assert rows[1]["status"] == "error"
    assert rows[1]["error_signature"] == "sim_failed"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["contract_version"] == CONTRACT_VERSION
    assert summary["total_events"] == 2
    assert summary["edges"]["edge.decode_candidate"]["seen"] == 1
    assert summary["edges"]["edge.truth.evaluate"]["error"] == 1


def test_contract_tracer_routing_enforcement(tmp_path: Path) -> None:
    tracer = ContractTracer(
        trace_path=tmp_path / "trace.jsonl",
        summary_path=tmp_path / "summary.json",
        fidelity=0,
        enforce_routing=True,
    )
    with pytest.raises(RuntimeError, match="Contract routing violation"):
        tracer.emit(
            edge_id="edge.decode_candidate",
            engine_mode="production",
            status="ok",
            request_payload={"x": [0.0], "ctx": {"fidelity": 0}},
            response_payload={"candidate": {"decoded": True}},
        )
    tracer.close()
