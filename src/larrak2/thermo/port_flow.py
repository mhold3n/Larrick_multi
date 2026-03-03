"""Equation-based rotary-port gas flow utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortFlowResult:
    """Aggregated gas-exchange flow terms for one cycle."""

    m_in_total: float
    m_out_total: float
    m_net_total: float
    m_dot_in: np.ndarray
    m_dot_out: np.ndarray
    intake_area: np.ndarray
    exhaust_area: np.ndarray
    choked_fraction_intake: float
    choked_fraction_exhaust: float
    branch_continuity_error: float


def _mod_angle_deg(theta: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64)
    return np.mod(arr, 360.0)


def _window_mask(theta_deg: np.ndarray, open_deg: float, close_deg: float) -> np.ndarray:
    """Return True where theta is in [open, close] with wrap-aware behavior."""
    t = _mod_angle_deg(theta_deg)
    o = float(open_deg) % 360.0
    c = float(close_deg) % 360.0
    if np.isclose(o, c):
        return np.ones_like(t, dtype=bool)
    if o < c:
        return (t >= o) & (t <= c)
    return (t >= o) | (t <= c)


def _edge_distance(theta: np.ndarray, edge: float) -> np.ndarray:
    d = np.abs(theta - edge)
    return np.minimum(d, 360.0 - d)


def rotary_port_area_schedule(
    theta_deg: np.ndarray,
    *,
    open_deg: float,
    close_deg: float,
    max_area_m2: float,
    ramp_deg: float = 5.0,
) -> np.ndarray:
    """Smooth open/close schedule for effective port area."""
    theta = _mod_angle_deg(theta_deg)
    area = np.zeros_like(theta, dtype=np.float64)
    active = _window_mask(theta, open_deg, close_deg)
    area[active] = float(max(max_area_m2, 0.0))

    r = max(float(ramp_deg), 1e-6)
    d_open = _edge_distance(theta, float(open_deg) % 360.0)
    d_close = _edge_distance(theta, float(close_deg) % 360.0)
    edge_factor = np.minimum(1.0, np.minimum(d_open, d_close) / r)
    area *= edge_factor
    return area


def critical_pressure_ratio(gamma: float) -> float:
    g = max(float(gamma), 1.0001)
    return (2.0 / (g + 1.0)) ** (g / (g - 1.0))


def compressible_orifice_mass_flow(
    p_up: float,
    t_up: float,
    p_down: float,
    area: float,
    *,
    cd: float,
    gamma: float,
    r_specific: float,
) -> tuple[float, bool]:
    """Mass flow rate through compressible orifice (kg/s), positive from up->down."""
    a = max(float(area), 0.0)
    if a <= 0.0:
        return 0.0, False

    p_u = max(float(p_up), 1.0)
    p_d = max(float(p_down), 1.0)
    t_u = max(float(t_up), 120.0)
    g = max(float(gamma), 1.0001)
    r = max(float(r_specific), 1e-9)
    c_d = float(np.clip(cd, 0.0, 1.5))

    pr = p_d / p_u
    pr_crit = critical_pressure_ratio(g)

    prefactor = c_d * a * p_u / np.sqrt(r * t_u)

    if pr <= pr_crit:
        # Choked branch
        phi = np.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        return float(prefactor * phi), True

    # Subcritical branch
    term = max(0.0, pr ** (2.0 / g) - pr ** ((g + 1.0) / g))
    phi = np.sqrt(2.0 * g / (g - 1.0) * term)
    return float(prefactor * phi), False


def signed_orifice_mass_flow(
    p_a: float,
    t_a: float,
    p_b: float,
    area: float,
    *,
    cd: float,
    gamma: float,
    r_specific: float,
) -> tuple[float, bool]:
    """Signed mass flow with direction from high-pressure side to low-pressure side."""
    if p_a >= p_b:
        mdot, choked = compressible_orifice_mass_flow(
            p_a,
            t_a,
            p_b,
            area,
            cd=cd,
            gamma=gamma,
            r_specific=r_specific,
        )
        return mdot, choked

    mdot, choked = compressible_orifice_mass_flow(
        p_b,
        t_a,
        p_a,
        area,
        cd=cd,
        gamma=gamma,
        r_specific=r_specific,
    )
    return -mdot, choked


def compute_cycle_port_flows(
    theta_deg: np.ndarray,
    p_cyl: np.ndarray,
    t_cyl: np.ndarray,
    *,
    rpm: float,
    p_manifold_pa: float,
    p_back_pa: float,
    intake_open_deg: float,
    intake_close_deg: float,
    exhaust_open_deg: float,
    exhaust_close_deg: float,
    intake_port_area_m2: float,
    exhaust_port_area_m2: float,
    cd_intake: float,
    cd_exhaust: float,
    gamma: float,
    r_specific: float,
) -> PortFlowResult:
    """Compute intake and exhaust mass-flow traces over one cycle."""
    theta = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    p_cycle = np.asarray(p_cyl, dtype=np.float64).reshape(-1)
    t_cycle = np.asarray(t_cyl, dtype=np.float64).reshape(-1)
    if theta.size < 2:
        raise ValueError("theta_deg must have at least two points")
    if p_cycle.size != theta.size or t_cycle.size != theta.size:
        raise ValueError("theta, p_cyl, t_cyl length mismatch")

    intake_area = rotary_port_area_schedule(
        theta,
        open_deg=float(intake_open_deg),
        close_deg=float(intake_close_deg),
        max_area_m2=float(intake_port_area_m2),
    )
    exhaust_area = rotary_port_area_schedule(
        theta,
        open_deg=float(exhaust_open_deg),
        close_deg=float(exhaust_close_deg),
        max_area_m2=float(exhaust_port_area_m2),
    )

    m_dot_in = np.zeros(theta.size, dtype=np.float64)
    m_dot_out = np.zeros(theta.size, dtype=np.float64)
    choked_intake = 0
    choked_exhaust = 0

    for i in range(theta.size):
        mdot_int, int_choked = signed_orifice_mass_flow(
            float(p_manifold_pa),
            float(t_cycle[i]),
            float(p_cycle[i]),
            float(intake_area[i]),
            cd=float(cd_intake),
            gamma=float(gamma),
            r_specific=float(r_specific),
        )
        # intake positive into cylinder
        m_dot_in[i] = mdot_int
        if int_choked and abs(mdot_int) > 0.0:
            choked_intake += 1

        mdot_exh, exh_choked = signed_orifice_mass_flow(
            float(p_cycle[i]),
            float(t_cycle[i]),
            float(p_back_pa),
            float(exhaust_area[i]),
            cd=float(cd_exhaust),
            gamma=float(gamma),
            r_specific=float(r_specific),
        )
        # exhaust positive out of cylinder
        m_dot_out[i] = mdot_exh
        if exh_choked and abs(mdot_exh) > 0.0:
            choked_exhaust += 1

    dtheta = np.diff(theta)
    # Handle non-uniform grid and wrap final segment.
    dtheta = np.concatenate([dtheta, [max(1e-6, 360.0 - (theta[-1] - theta[0]))]])
    dt = dtheta / max(6.0 * float(rpm), 1e-9)

    m_in_total = float(np.sum(np.maximum(m_dot_in, 0.0) * dt))
    m_out_total = float(np.sum(np.maximum(m_dot_out, 0.0) * dt))

    # Continuity around critical pressure ratio: check left/right branch mismatch.
    pr_crit = critical_pressure_ratio(float(gamma))
    eps = 1e-5
    p_ref = max(float(p_manifold_pa), 1.0)
    t_ref = max(float(np.mean(t_cycle)), 120.0)
    test_area = max(float(np.max(intake_area)), 1e-10)
    mdot_left, _ = compressible_orifice_mass_flow(
        p_ref,
        t_ref,
        p_ref * max(1e-6, pr_crit - eps),
        test_area,
        cd=float(cd_intake),
        gamma=float(gamma),
        r_specific=float(r_specific),
    )
    mdot_right, _ = compressible_orifice_mass_flow(
        p_ref,
        t_ref,
        p_ref * min(0.999999, pr_crit + eps),
        test_area,
        cd=float(cd_intake),
        gamma=float(gamma),
        r_specific=float(r_specific),
    )
    denom = max(abs(mdot_left), abs(mdot_right), 1e-12)
    branch_continuity_error = float(abs(mdot_left - mdot_right) / denom)

    return PortFlowResult(
        m_in_total=m_in_total,
        m_out_total=m_out_total,
        m_net_total=float(m_in_total - m_out_total),
        m_dot_in=m_dot_in,
        m_dot_out=m_dot_out,
        intake_area=intake_area,
        exhaust_area=exhaust_area,
        choked_fraction_intake=float(choked_intake / max(theta.size, 1)),
        choked_fraction_exhaust=float(choked_exhaust / max(theta.size, 1)),
        branch_continuity_error=branch_continuity_error,
    )
