"""Architecture-tracing helpers used by runtime and optimization."""

from .contracts import (
    CONNECTION_CONTRACT_V1,
    CONTRACT_VERSION,
    CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C,
    EDGE_IDS,
    ContractTracer,
    activate_contract_tracer,
    active_contract_tracer,
    deactivate_contract_tracer,
    expected_engine_mode,
    flatten_key_paths,
    get_active_contract_tracer,
    log_contract_edge,
)

__all__ = [
    "CONNECTION_CONTRACT_V1",
    "CONTRACT_VERSION",
    "CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C",
    "EDGE_IDS",
    "ContractTracer",
    "active_contract_tracer",
    "activate_contract_tracer",
    "deactivate_contract_tracer",
    "expected_engine_mode",
    "flatten_key_paths",
    "get_active_contract_tracer",
    "log_contract_edge",
]
