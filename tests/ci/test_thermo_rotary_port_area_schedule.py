"""Rotary port area schedule shape checks."""

from __future__ import annotations

import numpy as np

from larrak2.thermo.port_flow import rotary_port_area_schedule


def test_rotary_port_area_schedule_basic_behavior() -> None:
    theta = np.linspace(0.0, 359.0, 360)
    area = rotary_port_area_schedule(
        theta,
        open_deg=20.0,
        close_deg=120.0,
        max_area_m2=4.0e-4,
        ramp_deg=5.0,
    )

    assert area.shape == theta.shape
    assert np.all(area >= 0.0)
    assert float(np.max(area)) <= 4.0e-4 + 1e-12

    outside = (theta < 15.0) | (theta > 125.0)
    assert np.all(area[outside] <= 1e-12)

    inside = (theta >= 30.0) & (theta <= 110.0)
    assert np.mean(area[inside]) > 0.6 * 4.0e-4


def test_rotary_port_area_schedule_equal_open_close_means_closed() -> None:
    theta = np.linspace(0.0, 359.0, 360)
    area = rotary_port_area_schedule(
        theta,
        open_deg=0.0,
        close_deg=0.0,
        max_area_m2=4.0e-4,
        ramp_deg=5.0,
    )

    assert area.shape == theta.shape
    assert np.all(area <= 1e-12)
