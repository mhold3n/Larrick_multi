"""Hybrid correction bounds and envelope-gating checks."""

from __future__ import annotations

import numpy as np

from larrak2.core.encoding import decode_candidate, mid_bounds_candidate
from larrak2.core.types import EvalContext
from larrak2.thermo import two_zone


def test_hybrid_correction_is_bounded_by_beta() -> None:
    x_eq = 10.0
    x_nn_hi = 100.0
    x_nn_lo = 0.0
    k = 0.5
    beta = 0.15

    hi = two_zone._hybrid_correct(x_eq=x_eq, x_nn=x_nn_hi, k=k, beta=beta)
    lo = two_zone._hybrid_correct(x_eq=x_eq, x_nn=x_nn_lo, k=k, beta=beta)

    assert np.isclose(hi, x_eq * (1.0 + beta))
    assert np.isclose(lo, x_eq * (1.0 - beta))


def test_hybrid_correction_disabled_outside_validated_envelope(monkeypatch) -> None:
    candidate = decode_candidate(mid_bounds_candidate())

    def _fake_predict(**_: object) -> dict[str, float]:
        # Large disagreement should still not apply hybrid correction outside envelope.
        return {
            "m_air_trapped": 1.0e3,
            "residual_fraction": 0.99,
            "scavenging_efficiency": 0.01,
        }

    monkeypatch.setattr(two_zone, "_predict_openfoam_breathing", _fake_predict)

    ctx = EvalContext(
        rpm=3000.0,
        torque=450.0,  # outside envelope torque_max=400
        fidelity=2,
        seed=11,
    )
    result = two_zone.evaluate_two_zone_thermo(candidate.thermo, ctx)

    assert result.diag["in_validated_envelope"] is False
    assert result.diag["thermo_benchmark_status"] == "outside_validated_envelope"
    assert result.diag["thermo_hybrid_correction_active"] is False
    assert any(
        "hybrid correction disabled outside validated envelope" in str(msg)
        for msg in result.diag.get("thermo_validation_messages", [])
    )
