"""Equation-based gas-exchange / scavenging model for rotary-port breathing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .port_flow import PortFlowResult, compute_cycle_port_flows


@dataclass
class ScavengingResult:
    """Cycle-level scavenging metrics."""

    delivery_ratio: float
    trapping_efficiency: float
    scavenging_efficiency: float
    residual_fraction: float
    m_air_trapped: float
    m_in_total: float
    m_out_total: float
    branch_continuity_error: float
    flow_trace: PortFlowResult


def perfect_displacement_scavenging(
    delivery_ratio: float,
) -> ScavengingResult:
    """Idealized perfect-displacement scavenging envelope."""
    dr = np.clip(float(delivery_ratio), 0.0, 2.0)
    if dr <= 1.0:
        scav_eff = dr
        trap_eff = 1.0
    else:
        scav_eff = 1.0
        trap_eff = 1.0 / dr

    residual = 1.0 - scav_eff
    zeros = np.zeros(1, dtype=np.float64)
    dummy_trace = PortFlowResult(
        m_in_total=0.0,
        m_out_total=0.0,
        m_net_total=0.0,
        m_dot_in=zeros,
        m_dot_out=zeros,
        intake_area=zeros,
        exhaust_area=zeros,
        choked_fraction_intake=0.0,
        choked_fraction_exhaust=0.0,
        branch_continuity_error=0.0,
    )
    return ScavengingResult(
        delivery_ratio=float(dr),
        trapping_efficiency=float(trap_eff),
        scavenging_efficiency=float(scav_eff),
        residual_fraction=float(residual),
        m_air_trapped=0.0,
        m_in_total=0.0,
        m_out_total=0.0,
        branch_continuity_error=0.0,
        flow_trace=dummy_trace,
    )


def evaluate_rotary_scavenging(
    *,
    theta_deg: np.ndarray,
    p_cyl: np.ndarray,
    t_cyl: np.ndarray,
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
    m_initial: float,
) -> ScavengingResult:
    """Compute equation-based scavenging metrics from rotary-port flow equations."""
    flow = compute_cycle_port_flows(
        theta_deg,
        p_cyl,
        t_cyl,
        rpm=float(rpm),
        p_manifold_pa=float(p_manifold_pa),
        p_back_pa=float(p_back_pa),
        intake_open_deg=float(intake_open_deg),
        intake_close_deg=float(intake_close_deg),
        exhaust_open_deg=float(exhaust_open_deg),
        exhaust_close_deg=float(exhaust_close_deg),
        intake_port_area_m2=float(intake_port_area_m2),
        exhaust_port_area_m2=float(exhaust_port_area_m2),
        cd_intake=float(cd_intake),
        cd_exhaust=float(cd_exhaust),
        gamma=float(gamma),
        r_specific=float(r_specific),
    )

    m0 = max(float(m_initial), 1e-12)
    m_air_trapped = max(1e-12, m0 + flow.m_net_total)

    # Delivery ratio in scavenging literature form.
    delivery_ratio = float(np.clip(flow.m_in_total / m0, 0.0, 10.0))
    ideal = perfect_displacement_scavenging(delivery_ratio)

    # Blend idealized scavenging with physically computed exchange fraction.
    exchange_frac = np.clip(flow.m_out_total / m0, 0.0, 1.0)
    scavenging_efficiency = float(
        np.clip(0.5 * ideal.scavenging_efficiency + 0.5 * exchange_frac, 0.0, 1.0)
    )

    trapping_efficiency = (
        float(np.clip(flow.m_in_total / max(flow.m_in_total + flow.m_out_total, 1e-12), 0.0, 1.0))
        if flow.m_in_total > 0.0
        else 0.0
    )
    residual_fraction = float(np.clip(1.0 - scavenging_efficiency, 0.0, 1.0))

    return ScavengingResult(
        delivery_ratio=delivery_ratio,
        trapping_efficiency=trapping_efficiency,
        scavenging_efficiency=scavenging_efficiency,
        residual_fraction=residual_fraction,
        m_air_trapped=m_air_trapped,
        m_in_total=float(flow.m_in_total),
        m_out_total=float(flow.m_out_total),
        branch_continuity_error=float(flow.branch_continuity_error),
        flow_trace=flow,
    )
