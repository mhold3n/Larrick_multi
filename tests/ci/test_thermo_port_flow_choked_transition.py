"""Choked/subcritical transition checks for port-flow equations."""

from __future__ import annotations

import numpy as np

from larrak2.thermo.port_flow import compressible_orifice_mass_flow, critical_pressure_ratio


def test_choked_transition_continuity() -> None:
    gamma = 1.35
    p_up = 2.0e5
    t_up = 380.0
    area = 3.5e-4
    cd = 0.8
    r = 287.0

    pr_crit = critical_pressure_ratio(gamma)

    mdot_left, _ = compressible_orifice_mass_flow(
        p_up,
        t_up,
        p_up * max(1e-6, pr_crit - 1e-5),
        area,
        cd=cd,
        gamma=gamma,
        r_specific=r,
    )
    mdot_right, _ = compressible_orifice_mass_flow(
        p_up,
        t_up,
        p_up * min(0.999999, pr_crit + 1e-5),
        area,
        cd=cd,
        gamma=gamma,
        r_specific=r,
    )

    rel = abs(mdot_left - mdot_right) / max(abs(mdot_left), abs(mdot_right), 1e-12)
    assert rel <= 2e-2
    assert np.isfinite(mdot_left)
    assert np.isfinite(mdot_right)
