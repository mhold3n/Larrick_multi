"""Tests for motion-law and valve-timing co-optimization plumbing."""

from __future__ import annotations

import numpy as np

from larrak2.core.constraints import THERMO_CONSTRAINTS_FID0
from larrak2.core.encoding import ThermoParams, decode_candidate, mid_bounds_candidate
from larrak2.core.types import BreathingConfig, EvalContext
from larrak2.thermo.constants import load_thermo_constants
from larrak2.thermo.ignition_stage import evaluate_ignition_stage
from larrak2.thermo.mixture_preparation import evaluate_mixture_preparation
from larrak2.thermo.motionlaw import eval_thermo
from larrak2.thermo.two_zone import _phase_driven_volume
from larrak2.thermo.valve_timing import derive_valve_timing, overlap_duration_deg


def test_derive_valve_timing_anchors_intake_to_resolved_bdc() -> None:
    theta = np.linspace(0.0, 359.0, 360, dtype=np.float64)
    V, _dV = _phase_driven_volume(
        theta,
        compression_duration=60.0,
        expansion_duration=270.0,
        v_clearance=1.0,
        v_displaced=2.0,
    )
    params = ThermoParams(
        compression_duration=60.0,
        expansion_duration=270.0,
        heat_release_center=15.0,
        heat_release_width=30.0,
        lambda_af=1.0,
        intake_open_offset_from_bdc=-5.0,
        intake_duration_deg=80.0,
        exhaust_open_offset_from_expansion_tdc=-4.0,
        exhaust_duration_deg=90.0,
    )

    derived = derive_valve_timing(
        params=params,
        theta_deg=theta,
        volume=V,
        breathing=BreathingConfig(),
    )

    assert derived.motion_events.intake_anchor_bdc_deg == 270.0
    assert derived.intake_open_deg == 265.0
    assert derived.intake_close_deg == 345.0
    assert derived.exhaust_open_deg == 356.0
    assert derived.exhaust_close_deg == 86.0
    assert derived.overlap_deg == overlap_duration_deg(265.0, 345.0, 356.0, 86.0)


def test_derive_valve_timing_override_mode_uses_breathing_edges() -> None:
    theta = np.linspace(0.0, 359.0, 360, dtype=np.float64)
    V = np.linspace(1.0, 2.0, 360, dtype=np.float64)
    V[200:220] = 2.0
    params = ThermoParams(
        compression_duration=60.0,
        expansion_duration=90.0,
        heat_release_center=15.0,
        heat_release_width=30.0,
        lambda_af=1.0,
        intake_open_offset_from_bdc=-5.0,
        intake_duration_deg=80.0,
        exhaust_open_offset_from_expansion_tdc=-4.0,
        exhaust_duration_deg=90.0,
    )
    breathing = BreathingConfig(
        intake_open_deg=320.0,
        intake_close_deg=40.0,
        exhaust_open_deg=350.0,
        exhaust_close_deg=70.0,
        valve_timing_mode="override",
    )

    derived = derive_valve_timing(
        params=params,
        theta_deg=theta,
        volume=V,
        breathing=breathing,
    )

    assert derived.timing_source == "breathing_override"
    assert derived.intake_open_deg == 320.0
    assert derived.intake_close_deg == 40.0
    assert derived.exhaust_open_deg == 350.0
    assert derived.exhaust_close_deg == 70.0
    assert derived.overlap_deg == 50.0


def test_eval_thermo_emits_valve_timing_diag_and_new_constraints() -> None:
    candidate = decode_candidate(mid_bounds_candidate())
    ctx = EvalContext(
        rpm=1800.0,
        torque=80.0,
        fidelity=0,
        breathing=BreathingConfig(fuel_name="ethanol"),
    )

    result = eval_thermo(candidate.thermo, ctx)

    assert result.G.shape == (len(THERMO_CONSTRAINTS_FID0),)
    valve_timing = result.diag["valve_timing"]
    assert valve_timing["intake_open_offset_from_bdc"] == candidate.thermo.intake_open_offset_from_bdc
    assert valve_timing["intake_duration_deg"] == candidate.thermo.intake_duration_deg
    assert "overlap_deg" in valve_timing
    assert "stable_combustion_thresholds" in result.diag
    assert result.diag["mixture_preparation"]["fuel_name"] == "ethanol"
    assert "ignitability_margin" in result.diag["ignition_stage"]
    assert (
        result.diag["ignition_stage"]["spark_timing_deg_from_compression_tdc"]
        == candidate.thermo.spark_timing_deg_from_compression_tdc
    )
    assert result.diag["chemistry_handoff"]["fuel_name"] == "ethanol"


def test_mixture_preparation_responds_to_fuel_choice() -> None:
    params = decode_candidate(mid_bounds_candidate()).thermo
    constants = load_thermo_constants()

    gasoline_ctx = EvalContext(
        rpm=2200.0,
        torque=90.0,
        fidelity=0,
        breathing=BreathingConfig(fuel_name="gasoline"),
    )
    methanol_ctx = EvalContext(
        rpm=2200.0,
        torque=90.0,
        fidelity=0,
        breathing=BreathingConfig(fuel_name="methanol"),
    )
    gasoline_mix = evaluate_mixture_preparation(
        params=params,
        ctx=gasoline_ctx,
        breathing=gasoline_ctx.breathing or BreathingConfig(),
        m_air_trapped_kg=9.0e-4,
        intake_close_deg=340.0,
        constants=constants,
    )
    methanol_mix = evaluate_mixture_preparation(
        params=params,
        ctx=methanol_ctx,
        breathing=methanol_ctx.breathing or BreathingConfig(),
        m_air_trapped_kg=9.0e-4,
        intake_close_deg=340.0,
        constants=constants,
    )

    assert methanol_mix.charge_temp_k < gasoline_mix.charge_temp_k
    assert methanol_mix.delivered_vapor_fraction >= gasoline_mix.delivered_vapor_fraction
    assert 0.0 <= gasoline_mix.wall_film_fraction <= 0.45


def test_ignition_stage_preignition_margin_decreases_with_earlier_ivc() -> None:
    theta = np.linspace(0.0, 359.0, 360, dtype=np.float64)
    volume, _dV = _phase_driven_volume(
        theta,
        compression_duration=60.0,
        expansion_duration=270.0,
        v_clearance=1.0,
        v_displaced=2.0,
    )
    constants = load_thermo_constants()
    ctx = EvalContext(
        rpm=2400.0,
        torque=90.0,
        fidelity=0,
        breathing=BreathingConfig(fuel_name="gasoline"),
    )

    early_ivc_params = ThermoParams(
        compression_duration=60.0,
        expansion_duration=270.0,
        heat_release_center=15.0,
        heat_release_width=30.0,
        lambda_af=1.0,
        intake_open_offset_from_bdc=-10.0,
        intake_duration_deg=35.0,
        exhaust_open_offset_from_expansion_tdc=-4.0,
        exhaust_duration_deg=90.0,
        spark_timing_deg_from_compression_tdc=-8.0,
    )
    late_ivc_params = ThermoParams(
        compression_duration=60.0,
        expansion_duration=270.0,
        heat_release_center=15.0,
        heat_release_width=30.0,
        lambda_af=1.0,
        intake_open_offset_from_bdc=-10.0,
        intake_duration_deg=85.0,
        exhaust_open_offset_from_expansion_tdc=-4.0,
        exhaust_duration_deg=90.0,
        spark_timing_deg_from_compression_tdc=-8.0,
    )
    early_timing = derive_valve_timing(
        params=early_ivc_params,
        theta_deg=theta,
        volume=volume,
        breathing=ctx.breathing,
    )
    late_timing = derive_valve_timing(
        params=late_ivc_params,
        theta_deg=theta,
        volume=volume,
        breathing=ctx.breathing,
    )
    early_mix = evaluate_mixture_preparation(
        params=early_ivc_params,
        ctx=ctx,
        breathing=BreathingConfig(
            fuel_name="gasoline",
            intake_open_deg=early_timing.intake_open_deg,
            intake_close_deg=early_timing.intake_close_deg,
        ),
        m_air_trapped_kg=9.0e-4,
        intake_close_deg=early_timing.intake_close_deg,
        constants=constants,
    )
    late_mix = evaluate_mixture_preparation(
        params=late_ivc_params,
        ctx=ctx,
        breathing=BreathingConfig(
            fuel_name="gasoline",
            intake_open_deg=late_timing.intake_open_deg,
            intake_close_deg=late_timing.intake_close_deg,
        ),
        m_air_trapped_kg=9.0e-4,
        intake_close_deg=late_timing.intake_close_deg,
        constants=constants,
    )
    early_ignition = evaluate_ignition_stage(
        params=early_ivc_params,
        ctx=ctx,
        theta_deg=theta,
        volume=volume,
        motion_events=early_timing.motion_events,
        mixture=early_mix,
        ivc_deg=early_timing.intake_close_deg,
        p_manifold_pa=101325.0,
        gamma_u=constants.gamma_u,
    )
    late_ignition = evaluate_ignition_stage(
        params=late_ivc_params,
        ctx=ctx,
        theta_deg=theta,
        volume=volume,
        motion_events=late_timing.motion_events,
        mixture=late_mix,
        ivc_deg=late_timing.intake_close_deg,
        p_manifold_pa=101325.0,
        gamma_u=constants.gamma_u,
    )

    assert early_ignition.preignition_margin < late_ignition.preignition_margin


def test_ignition_stage_ignitability_margin_improves_with_later_spark() -> None:
    theta = np.linspace(0.0, 359.0, 360, dtype=np.float64)
    volume, _dV = _phase_driven_volume(
        theta,
        compression_duration=60.0,
        expansion_duration=270.0,
        v_clearance=1.0,
        v_displaced=2.0,
    )
    constants = load_thermo_constants()
    ctx = EvalContext(
        rpm=2600.0,
        torque=100.0,
        fidelity=0,
        breathing=BreathingConfig(fuel_name="gasoline"),
    )
    base_params = ThermoParams(
        compression_duration=60.0,
        expansion_duration=270.0,
        heat_release_center=15.0,
        heat_release_width=30.0,
        lambda_af=1.0,
        intake_open_offset_from_bdc=-5.0,
        intake_duration_deg=75.0,
        exhaust_open_offset_from_expansion_tdc=-4.0,
        exhaust_duration_deg=90.0,
        spark_timing_deg_from_compression_tdc=-15.0,
    )
    late_spark_params = ThermoParams(
        **{
            **base_params.__dict__,
            "spark_timing_deg_from_compression_tdc": -4.0,
        }
    )
    timing = derive_valve_timing(
        params=base_params,
        theta_deg=theta,
        volume=volume,
        breathing=ctx.breathing,
    )
    mix = evaluate_mixture_preparation(
        params=base_params,
        ctx=ctx,
        breathing=BreathingConfig(
            fuel_name="gasoline",
            intake_open_deg=timing.intake_open_deg,
            intake_close_deg=timing.intake_close_deg,
        ),
        m_air_trapped_kg=9.0e-4,
        intake_close_deg=timing.intake_close_deg,
        constants=constants,
    )
    early_ignition = evaluate_ignition_stage(
        params=base_params,
        ctx=ctx,
        theta_deg=theta,
        volume=volume,
        motion_events=timing.motion_events,
        mixture=mix,
        ivc_deg=timing.intake_close_deg,
        p_manifold_pa=101325.0,
        gamma_u=constants.gamma_u,
    )
    late_ignition = evaluate_ignition_stage(
        params=late_spark_params,
        ctx=ctx,
        theta_deg=theta,
        volume=volume,
        motion_events=timing.motion_events,
        mixture=mix,
        ivc_deg=timing.intake_close_deg,
        p_manifold_pa=101325.0,
        gamma_u=constants.gamma_u,
    )

    assert late_ignition.ignitability_margin > early_ignition.ignitability_margin
