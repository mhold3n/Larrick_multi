"""Thermo validation gates: physics invariants, benchmark agreement, trend checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import load_anchor_manifest


@dataclass
class ThermoValidationReport:
    """Validation report emitted by the two-zone thermo solver."""

    mass_residual: float
    energy_residual: float
    non_negative_states_ok: bool
    monotonic_burn_ok: bool
    choked_branch_continuity_error: float
    benchmark_status: str
    benchmark_ok: bool
    in_validated_envelope: bool
    trend_checks: dict[str, bool] = field(default_factory=dict)
    nn_disagreement: dict[str, float] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def passed(self, *, mass_tol: float, energy_tol: float, branch_tol: float) -> bool:
        trend_ok = all(bool(v) for v in self.trend_checks.values()) if self.trend_checks else True
        return bool(
            self.mass_residual <= float(mass_tol)
            and self.energy_residual <= float(energy_tol)
            and self.non_negative_states_ok
            and self.monotonic_burn_ok
            and self.choked_branch_continuity_error <= float(branch_tol)
            and self.benchmark_ok
            and trend_ok
        )


class ThermoValidationError(RuntimeError):
    """Structured thermo validation failure with machine-readable payload."""

    def __init__(self, message: str, *, payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.payload = dict(payload or {})


def in_validated_envelope(
    *,
    rpm: float,
    torque: float,
    manifest: dict[str, Any],
) -> bool:
    env = manifest.get("validated_envelope", {})
    return bool(
        float(env.get("rpm_min", -np.inf)) <= float(rpm) <= float(env.get("rpm_max", np.inf))
        and float(env.get("torque_min", -np.inf))
        <= float(torque)
        <= float(env.get("torque_max", np.inf))
    )


def compute_nn_disagreement(
    *,
    m_air_eq: float,
    m_air_nn: float,
    residual_eq: float,
    residual_nn: float,
    scavenging_eq: float,
    scavenging_nn: float,
) -> dict[str, float]:
    denom_air = max(abs(float(m_air_eq)), 1e-12)
    return {
        "delta_m_air": float((float(m_air_nn) - float(m_air_eq)) / denom_air),
        "delta_residual": float(float(residual_nn) - float(residual_eq)),
        "delta_scavenging_eff": float(float(scavenging_nn) - float(scavenging_eq)),
    }


def validate_benchmark_agreement(
    *,
    disagreement: dict[str, float],
    thresholds: dict[str, float],
    in_envelope: bool,
) -> tuple[bool, str, list[str]]:
    if not in_envelope:
        return True, "outside_validated_envelope", []

    msgs: list[str] = []
    ok = True
    if abs(float(disagreement.get("delta_m_air", 0.0))) > float(
        thresholds.get("delta_m_air_rel_max", 0.10)
    ):
        ok = False
        msgs.append("delta_m_air exceeds threshold")
    if abs(float(disagreement.get("delta_residual", 0.0))) > float(
        thresholds.get("delta_residual_abs_max", 0.05)
    ):
        ok = False
        msgs.append("delta_residual exceeds threshold")
    if abs(float(disagreement.get("delta_scavenging_eff", 0.0))) > float(
        thresholds.get("delta_scavenging_abs_max", 0.08)
    ):
        ok = False
        msgs.append("delta_scavenging_eff exceeds threshold")
    return ok, "pass" if ok else "failed", msgs


def evaluate_trend_checks(
    *,
    trapped_mass_base: float,
    trapped_mass_intake_up: float,
    scavenging_base: float,
    scavenging_backpressure_up: float,
    burn_cap_base: float,
    burn_cap_richer: float,
) -> dict[str, bool]:
    """Core trend sanity checks demanded by hard-first gate."""
    return {
        "intake_area_trend_ok": bool(
            float(trapped_mass_intake_up) >= float(trapped_mass_base) - 1e-9
        ),
        "backpressure_trend_ok": bool(
            float(scavenging_backpressure_up) <= float(scavenging_base) + 1e-9
        ),
        "rich_mixture_burn_cap_trend_ok": bool(
            float(burn_cap_richer) <= float(burn_cap_base) + 1e-9
        ),
    }


def build_validation_report(
    *,
    mass_residual: float,
    energy_residual: float,
    non_negative_states_ok: bool,
    monotonic_burn_ok: bool,
    choked_branch_continuity_error: float,
    benchmark_status: str,
    benchmark_ok: bool,
    in_validated_envelope_flag: bool,
    trend_checks: dict[str, bool],
    nn_disagreement: dict[str, float],
    messages: list[str] | None = None,
) -> ThermoValidationReport:
    return ThermoValidationReport(
        mass_residual=float(max(0.0, mass_residual)),
        energy_residual=float(max(0.0, energy_residual)),
        non_negative_states_ok=bool(non_negative_states_ok),
        monotonic_burn_ok=bool(monotonic_burn_ok),
        choked_branch_continuity_error=float(max(0.0, choked_branch_continuity_error)),
        benchmark_status=str(benchmark_status),
        benchmark_ok=bool(benchmark_ok),
        in_validated_envelope=bool(in_validated_envelope_flag),
        trend_checks=dict(trend_checks),
        nn_disagreement=dict(nn_disagreement),
        messages=list(messages or []),
    )


def load_validation_manifest(path: str | None) -> dict[str, Any]:
    """Load anchor manifest with defaults."""
    return load_anchor_manifest(path)
