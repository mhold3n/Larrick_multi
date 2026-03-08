"""Valve timing derivation from motion-law anchors."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..core.encoding import ThermoParams
from ..core.types import BreathingConfig
from .timing_profile import load_thermo_timing_profile


@dataclass(frozen=True)
class MotionLawEvents:
    expansion_start_tdc_deg: float
    intake_anchor_bdc_deg: float
    compression_start_bdc_deg: float
    compression_end_tdc_deg: float
    bdc_detection_method: str


@dataclass(frozen=True)
class DerivedValveTiming:
    intake_open_offset_from_bdc: float
    intake_duration_deg: float
    exhaust_open_offset_from_expansion_tdc: float
    exhaust_duration_deg: float
    intake_open_deg: float
    intake_close_deg: float
    exhaust_open_deg: float
    exhaust_close_deg: float
    overlap_deg: float
    timing_source: str
    timing_profile_id: str
    timing_profile_version: str
    timing_legacy_injected: bool
    motion_events: MotionLawEvents

    def as_dict(self) -> dict[str, float | str | bool | dict[str, float | str]]:
        return {
            "intake_open_offset_from_bdc": float(self.intake_open_offset_from_bdc),
            "intake_duration_deg": float(self.intake_duration_deg),
            "exhaust_open_offset_from_expansion_tdc": float(
                self.exhaust_open_offset_from_expansion_tdc
            ),
            "exhaust_duration_deg": float(self.exhaust_duration_deg),
            "intake_open_deg": float(self.intake_open_deg),
            "intake_close_deg": float(self.intake_close_deg),
            "exhaust_open_deg": float(self.exhaust_open_deg),
            "exhaust_close_deg": float(self.exhaust_close_deg),
            "overlap_deg": float(self.overlap_deg),
            "timing_source": str(self.timing_source),
            "timing_profile_id": str(self.timing_profile_id),
            "timing_profile_version": str(self.timing_profile_version),
            "timing_legacy_injected": bool(self.timing_legacy_injected),
            "motion_events": {
                "expansion_start_tdc_deg": float(self.motion_events.expansion_start_tdc_deg),
                "intake_anchor_bdc_deg": float(self.motion_events.intake_anchor_bdc_deg),
                "compression_start_bdc_deg": float(self.motion_events.compression_start_bdc_deg),
                "compression_end_tdc_deg": float(self.motion_events.compression_end_tdc_deg),
                "bdc_detection_method": str(self.motion_events.bdc_detection_method),
            },
        }


def _mod_deg(value: float) -> float:
    return float(np.mod(float(value), 360.0))


def _expand_interval(open_deg: float, close_deg: float) -> list[tuple[float, float]]:
    o = _mod_deg(open_deg)
    c = _mod_deg(close_deg)
    if np.isclose(o, c):
        return []
    if o < c:
        return [(o, c)]
    return [(o, 360.0), (0.0, c)]


def overlap_duration_deg(
    intake_open_deg: float,
    intake_close_deg: float,
    exhaust_open_deg: float,
    exhaust_close_deg: float,
) -> float:
    total = 0.0
    for lo_a, hi_a in _expand_interval(intake_open_deg, intake_close_deg):
        for lo_b, hi_b in _expand_interval(exhaust_open_deg, exhaust_close_deg):
            total += max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    return float(total)


def resolve_motion_law_events(theta_deg: np.ndarray, volume: np.ndarray) -> MotionLawEvents:
    """Resolve the candidate-specific motion-law anchor events from the volume trace."""
    theta = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    V = np.asarray(volume, dtype=np.float64).reshape(-1)
    if theta.size == 0 or V.size == 0 or theta.size != V.size:
        raise ValueError("theta_deg and volume must be non-empty arrays with matching length")

    vmax = float(np.max(V))
    mask = np.isclose(V, vmax, rtol=1e-6, atol=max(1e-12, abs(vmax) * 1e-6))
    max_indices = np.where(mask)[0]
    if max_indices.size == 0:
        raise RuntimeError("Unable to resolve BDC anchor from motion-law volume trace")

    first_idx = int(max_indices[0])
    last_idx = int(max_indices[-1])
    return MotionLawEvents(
        expansion_start_tdc_deg=0.0,
        intake_anchor_bdc_deg=float(theta[first_idx]),
        compression_start_bdc_deg=float(theta[last_idx]),
        compression_end_tdc_deg=360.0,
        bdc_detection_method="first_max_volume_sample",
    )


def derive_valve_timing(
    *,
    params: ThermoParams,
    theta_deg: np.ndarray,
    volume: np.ndarray,
    breathing: BreathingConfig | None,
    timing_profile_path: str | None = None,
) -> DerivedValveTiming:
    profile = load_thermo_timing_profile(timing_profile_path)
    mode = str(getattr(breathing, "valve_timing_mode", "candidate"))
    events = resolve_motion_law_events(theta_deg, volume)

    if mode == "override" and breathing is not None:
        intake_open = float(breathing.intake_open_deg)
        intake_close = float(breathing.intake_close_deg)
        exhaust_open = float(breathing.exhaust_open_deg)
        exhaust_close = float(breathing.exhaust_close_deg)
        overlap = overlap_duration_deg(intake_open, intake_close, exhaust_open, exhaust_close)
        intake_offset = float((_mod_deg(intake_open - events.intake_anchor_bdc_deg) + 180.0) % 360.0 - 180.0)
        exhaust_offset = float(
            (_mod_deg(exhaust_open - events.expansion_start_tdc_deg) + 180.0) % 360.0 - 180.0
        )
        intake_duration = float((_mod_deg(intake_close - intake_open) + 360.0) % 360.0)
        exhaust_duration = float((_mod_deg(exhaust_close - exhaust_open) + 360.0) % 360.0)
        timing_source = "breathing_override"
        legacy_injected = False
    else:
        intake_offset = float(params.intake_open_offset_from_bdc)
        intake_duration = float(params.intake_duration_deg)
        exhaust_offset = float(params.exhaust_open_offset_from_expansion_tdc)
        exhaust_duration = float(params.exhaust_duration_deg)
        intake_open = _mod_deg(events.intake_anchor_bdc_deg + intake_offset)
        intake_close = _mod_deg(intake_open + intake_duration)
        exhaust_open = _mod_deg(events.expansion_start_tdc_deg + exhaust_offset)
        exhaust_close = _mod_deg(exhaust_open + exhaust_duration)
        overlap = overlap_duration_deg(intake_open, intake_close, exhaust_open, exhaust_close)
        timing_source = str(getattr(params, "timing_source", "encoded_candidate"))
        legacy_injected = bool(getattr(params, "timing_legacy_injected", False))

    return DerivedValveTiming(
        intake_open_offset_from_bdc=float(intake_offset),
        intake_duration_deg=float(intake_duration),
        exhaust_open_offset_from_expansion_tdc=float(exhaust_offset),
        exhaust_duration_deg=float(exhaust_duration),
        intake_open_deg=float(intake_open),
        intake_close_deg=float(intake_close),
        exhaust_open_deg=float(exhaust_open),
        exhaust_close_deg=float(exhaust_close),
        overlap_deg=float(overlap),
        timing_source=timing_source,
        timing_profile_id=str(profile.profile_id),
        timing_profile_version=str(profile.profile_version),
        timing_legacy_injected=legacy_injected,
        motion_events=events,
    )


def breathing_with_derived_timing(
    breathing: BreathingConfig | None,
    derived: DerivedValveTiming,
) -> BreathingConfig:
    base = breathing or BreathingConfig()
    return replace(
        base,
        overlap_deg=float(derived.overlap_deg),
        intake_open_deg=float(derived.intake_open_deg),
        intake_close_deg=float(derived.intake_close_deg),
        exhaust_open_deg=float(derived.exhaust_open_deg),
        exhaust_close_deg=float(derived.exhaust_close_deg),
    )

