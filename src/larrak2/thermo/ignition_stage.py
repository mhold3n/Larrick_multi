"""Compression-to-ignition chemistry stage for hybrid thermo evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.encoding import ThermoParams
from ..core.types import EvalContext
from .chemistry_profile import fuel_profile_for_name, load_thermo_chemistry_profile
from .mixture_preparation import MixturePreparationResult
from .valve_timing import MotionLawEvents


@dataclass(frozen=True)
class IgnitionStageResult:
    spark_timing_deg_from_compression_tdc: float
    spark_absolute_deg: float
    ivc_deg: float
    preignition_integral: float
    preignition_margin: float
    ignitability_delay_target_s: float
    ignition_delay_spark_s: float
    ignitability_margin: float
    soc_deg: float
    ca10_deg: float
    burn_duration_deg: float
    chemistry_heat_release_center_deg: float
    chemistry_heat_release_width_deg: float
    fuel_name: str
    profile_id: str
    profile_version: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "spark_timing_deg_from_compression_tdc": float(
                self.spark_timing_deg_from_compression_tdc
            ),
            "spark_absolute_deg": float(self.spark_absolute_deg),
            "ivc_deg": float(self.ivc_deg),
            "preignition_integral": float(self.preignition_integral),
            "preignition_margin": float(self.preignition_margin),
            "ignitability_delay_target_s": float(self.ignitability_delay_target_s),
            "ignition_delay_spark_s": float(self.ignition_delay_spark_s),
            "ignitability_margin": float(self.ignitability_margin),
            "soc_deg": float(self.soc_deg),
            "ca10_deg": float(self.ca10_deg),
            "burn_duration_deg": float(self.burn_duration_deg),
            "chemistry_heat_release_center_deg": float(self.chemistry_heat_release_center_deg),
            "chemistry_heat_release_width_deg": float(self.chemistry_heat_release_width_deg),
            "fuel_name": str(self.fuel_name),
            "profile_id": str(self.profile_id),
            "profile_version": str(self.profile_version),
        }


def _interp_periodic(
    theta_deg: np.ndarray, values: np.ndarray, query_deg: np.ndarray
) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    query = np.asarray(query_deg, dtype=np.float64).reshape(-1)
    order = np.argsort(theta)
    theta_s = np.mod(theta[order], 360.0)
    vals_s = vals[order]
    theta_ext = np.concatenate([theta_s, [theta_s[0] + 360.0]])
    vals_ext = np.concatenate([vals_s, [vals_s[0]]])
    q = np.mod(query, 360.0)
    return np.interp(q, theta_ext, vals_ext)


def _ignition_delay_s(
    *,
    temp_k: float,
    pressure_pa: float,
    lambda_af: float,
    delivered_vapor_fraction: float,
    mixture_homogeneity: float,
    fuel: Any,
) -> float:
    t = max(float(temp_k), 250.0)
    p = max(float(pressure_pa), 1.0)
    tau = float(fuel.ignition_delay_ref_s)
    temp_power = float(np.clip(float(fuel.ignition_temp_activation_k) / 1500.0, 1.0, 8.0))
    tau *= (float(fuel.ignition_temp_ref_k) / t) ** temp_power
    pressure_power = float(max(-float(fuel.ignition_pressure_exponent), 0.0))
    tau *= (float(fuel.ignition_pressure_ref_pa) / p) ** pressure_power
    tau *= 1.0 + float(fuel.ignition_lambda_sensitivity) * abs(float(lambda_af) - 1.0)
    tau /= max(0.25, 0.25 + 0.75 * float(delivered_vapor_fraction))
    tau /= max(0.25, 0.25 + 0.75 * float(mixture_homogeneity))
    return float(np.clip(tau, 1.0e-5, 0.2))


def evaluate_ignition_stage(
    *,
    params: ThermoParams,
    ctx: EvalContext,
    theta_deg: np.ndarray,
    volume: np.ndarray,
    motion_events: MotionLawEvents,
    mixture: MixturePreparationResult,
    ivc_deg: float,
    p_manifold_pa: float,
    gamma_u: float,
) -> IgnitionStageResult:
    profile = load_thermo_chemistry_profile(getattr(ctx, "thermo_chemistry_profile_path", None))
    fuel = fuel_profile_for_name(mixture.fuel_name, profile_path=profile.path)

    spark_rel = float(
        getattr(
            params, "spark_timing_deg_from_compression_tdc", profile.spark_timing_legacy_default
        )
    )
    spark_abs = float(motion_events.compression_end_tdc_deg + spark_rel)
    spark_abs_wrapped = float(np.mod(spark_abs, 360.0))
    ivc_deg = float(np.mod(ivc_deg, 360.0))

    span_deg = spark_abs - ivc_deg
    if span_deg <= 0.0:
        span_deg += 360.0
    n_steps = max(8, int(np.ceil(span_deg / 2.0)) + 1)
    theta_samples = np.linspace(ivc_deg, spark_abs, n_steps, dtype=np.float64)
    V_samples = _interp_periodic(theta_deg, volume, theta_samples)
    V_ivc = max(float(V_samples[0]), 1e-12)
    compression_ratio = np.clip(V_ivc / np.maximum(V_samples, 1e-12), 1.0, None)
    p_samples = float(p_manifold_pa) * compression_ratio ** float(gamma_u)
    t_samples = float(mixture.charge_temp_k) * compression_ratio ** max(float(gamma_u) - 1.0, 0.0)

    dtheta = np.diff(theta_samples)
    dt = dtheta / max(6.0 * float(ctx.rpm), 1e-9)
    preignition_integral = 0.0
    for i in range(len(dt)):
        tau = _ignition_delay_s(
            temp_k=float(t_samples[i]),
            pressure_pa=float(p_samples[i]),
            lambda_af=float(params.lambda_af),
            delivered_vapor_fraction=float(mixture.delivered_vapor_fraction),
            mixture_homogeneity=float(mixture.mixture_homogeneity),
            fuel=fuel,
        )
        preignition_integral += float(dt[i]) / max(tau, 1e-9)

    preignition_margin = float(1.0 - preignition_integral)
    tau_spark = _ignition_delay_s(
        temp_k=float(t_samples[-1]),
        pressure_pa=float(p_samples[-1]),
        lambda_af=float(params.lambda_af),
        delivered_vapor_fraction=float(mixture.delivered_vapor_fraction),
        mixture_homogeneity=float(mixture.mixture_homogeneity),
        fuel=fuel,
    )
    compression_window_deg = float(motion_events.compression_end_tdc_deg - ivc_deg)
    if compression_window_deg <= 0.0:
        compression_window_deg += 360.0
    compression_completion = float(
        np.clip((spark_abs - ivc_deg) / max(compression_window_deg, 1e-9), 0.0, 1.0)
    )
    tau_spark /= 0.6 + 0.8 * compression_completion
    ign_target = float(fuel.ignitability_target_delay_s) * (
        1.0 + 0.3 * float(mixture.mixture_inhomogeneity) + 0.1 * abs(float(params.lambda_af) - 1.0)
    )
    ignitability_margin = float(ign_target / max(tau_spark, 1e-9) - 1.0)

    spark_delay_s = tau_spark * float(fuel.spark_assist_factor)
    soc_abs = spark_abs + spark_delay_s * 6.0 * float(ctx.rpm)
    ca10_lag_deg = max(
        2.0,
        0.18
        * float(fuel.burn_duration_base_deg)
        * (
            1.0
            + 0.8 * float(mixture.mixture_inhomogeneity)
            + 0.4 * float(mixture.wall_film_fraction)
            + 0.25 * abs(float(params.lambda_af) - 1.0)
        ),
    )
    ca10_abs = soc_abs + ca10_lag_deg

    chemistry_burn_duration = float(fuel.burn_duration_base_deg) * (
        1.0
        + float(fuel.burn_duration_lambda_sensitivity) * abs(float(params.lambda_af) - 1.0)
        + float(fuel.burn_duration_inhomogeneity_sensitivity) * float(mixture.mixture_inhomogeneity)
        + 0.35 * float(mixture.wall_film_fraction)
        - 0.15 * float(mixture.delivered_vapor_fraction)
    )
    chemistry_burn_duration = float(
        np.clip(
            chemistry_burn_duration,
            float(profile.wiebe_handoff.burn_duration_min_deg),
            float(profile.wiebe_handoff.burn_duration_max_deg),
        )
    )
    chemistry_center = float(np.mod(soc_abs + 0.5 * chemistry_burn_duration, 360.0))

    return IgnitionStageResult(
        spark_timing_deg_from_compression_tdc=spark_rel,
        spark_absolute_deg=spark_abs_wrapped,
        ivc_deg=float(np.mod(ivc_deg, 360.0)),
        preignition_integral=float(preignition_integral),
        preignition_margin=preignition_margin,
        ignitability_delay_target_s=float(ign_target),
        ignition_delay_spark_s=float(tau_spark),
        ignitability_margin=ignitability_margin,
        soc_deg=float(np.mod(soc_abs, 360.0)),
        ca10_deg=float(np.mod(ca10_abs, 360.0)),
        burn_duration_deg=chemistry_burn_duration,
        chemistry_heat_release_center_deg=chemistry_center,
        chemistry_heat_release_width_deg=chemistry_burn_duration,
        fuel_name=str(fuel.fuel_name),
        profile_id=str(profile.profile_id),
        profile_version=str(profile.profile_version),
    )


__all__ = ["IgnitionStageResult", "evaluate_ignition_stage"]
