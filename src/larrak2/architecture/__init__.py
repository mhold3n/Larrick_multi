"""Architecture contract primitives for orchestration/interface readiness."""

from .contracts import (
    CONNECTION_CONTRACT_V1,
    CONTRACT_VERSION,
    CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C,
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
    "CONTRACT_VERSION",
    "CONNECTION_CONTRACT_V1",
    "CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C",
    "ContractTracer",
    "activate_contract_tracer",
    "active_contract_tracer",
    "deactivate_contract_tracer",
    "expected_engine_mode",
    "flatten_key_paths",
    "get_active_contract_tracer",
    "log_contract_edge",
]
