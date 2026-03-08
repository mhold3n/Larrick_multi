"""Upstream mixture-preparation model for shower-injector plenum/port fueling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.encoding import ThermoParams
from ..core.types import BreathingConfig, EvalContext
from .chemistry_profile import fuel_profile_for_name, load_thermo_chemistry_profile


@dataclass(frozen=True)
class MixturePreparationResult:
    delivered_vapor_fraction: float
    wall_film_fraction: float
    mixture_inhomogeneity: float
    mixture_homogeneity: float
    residence_time_s: float
    plenum_residence_time_s: float
    charge_temp_k: float
    delivered_fuel_vapor_mass_kg: float
    wall_film_mass_kg: float
    fuel_name: str
    profile_id: str
    profile_version: str
    injector_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "delivered_vapor_fraction": float(self.delivered_vapor_fraction),
            "wall_film_fraction": float(self.wall_film_fraction),
            "mixture_inhomogeneity": float(self.mixture_inhomogeneity),
            "mixture_homogeneity": float(self.mixture_homogeneity),
            "residence_time_s": float(self.residence_time_s),
            "plenum_residence_time_s": float(self.plenum_residence_time_s),
            "charge_temp_k": float(self.charge_temp_k),
            "delivered_fuel_vapor_mass_kg": float(self.delivered_fuel_vapor_mass_kg),
            "wall_film_mass_kg": float(self.wall_film_mass_kg),
            "fuel_name": str(self.fuel_name),
            "profile_id": str(self.profile_id),
            "profile_version": str(self.profile_version),
            "injector_type": str(self.injector_type),
        }


def evaluate_mixture_preparation(
    *,
    params: ThermoParams,
    ctx: EvalContext,
    breathing: BreathingConfig,
    m_air_trapped_kg: float,
    intake_close_deg: float,
    constants: Any,
) -> MixturePreparationResult:
    profile = load_thermo_chemistry_profile(getattr(ctx, "thermo_chemistry_profile_path", None))
    fuel = fuel_profile_for_name(
        getattr(breathing, "fuel_name", "gasoline"), profile_path=profile.path
    )
    hw = profile.hardware

    rpm = max(float(ctx.rpm), 1.0)
    intake_open_deg = float(getattr(breathing, "intake_open_deg", 0.0))
    intake_duration_deg = float((float(intake_close_deg) - intake_open_deg + 360.0) % 360.0)
    intake_duration_s = intake_duration_deg / max(6.0 * rpm, 1e-9)
    plenum_residence_time_s = float(hw.plenum_volume_factor) * intake_duration_s
    residence_time_s = intake_duration_s + plenum_residence_time_s

    m_air = max(float(m_air_trapped_kg), 1e-12)
    lam = max(float(params.lambda_af), 1e-6)
    m_fuel = m_air / max(float(fuel.afr_stoich) * lam, 1e-12)

    p_ref = max(float(getattr(breathing, "p_manifold_Pa", 101325.0)), 1.0)
    t_ref = max(float(getattr(constants, "t_intake_k", 300.0)), 120.0)
    tau = float(hw.evaporation_tau_ref_s) * float(fuel.evaporation_tau_scale)
    tau *= (t_ref / 350.0) ** float(hw.evaporation_temp_exponent)
    tau *= (p_ref / 101325.0) ** float(hw.evaporation_pressure_exponent)
    tau = float(np.clip(tau, 1e-5, 0.25))

    equilibrium_vapor_fraction = 1.0 - np.exp(-residence_time_s / max(tau, 1e-9))
    equilibrium_vapor_fraction = float(np.clip(equilibrium_vapor_fraction, 0.0, 1.0))

    wall_film_fraction = float(
        np.clip(
            hw.wall_film_base_fraction
            * fuel.wall_wetting_factor
            * max(0.15, 1.0 - equilibrium_vapor_fraction)
            * (1.0 + 0.25 * max(0.0, abs(float(params.lambda_af) - 1.0))),
            0.0,
            0.45,
        )
    )
    delivered_vapor_fraction = float(
        np.clip(equilibrium_vapor_fraction * (1.0 - wall_film_fraction), 0.0, 1.0)
    )

    base_homogeneity = float(hw.mixture_homogeneity_base) * float(fuel.mixture_homogeneity_factor)
    mixing_gain = float(hw.mixing_length_factor) * float(np.clip(residence_time_s / 0.03, 0.0, 1.0))
    lambda_penalty = float(hw.mixture_inhomogeneity_lambda_gain) * abs(
        float(params.lambda_af) - 1.0
    )
    rpm_penalty = float(hw.mixture_inhomogeneity_rpm_gain) * float(np.clip(rpm / 7000.0, 0.0, 1.0))
    mixture_homogeneity = float(
        np.clip(
            base_homogeneity
            + mixing_gain
            - lambda_penalty
            - rpm_penalty
            - 0.7 * wall_film_fraction,
            0.0,
            1.0,
        )
    )
    mixture_inhomogeneity = float(np.clip(1.0 - mixture_homogeneity, 0.0, 1.0))

    fuel_air_ratio = m_fuel / max(m_air + m_fuel, 1e-12)
    charge_temp_drop = (
        float(hw.charge_cooling_gain_k)
        * float(fuel.charge_cooling_factor)
        * delivered_vapor_fraction
        * float(np.clip(8.0 * fuel_air_ratio, 0.0, 1.0))
    )
    charge_temp_k = float(np.clip(t_ref - charge_temp_drop, 230.0, t_ref + 20.0))

    delivered_fuel_vapor_mass_kg = float(m_fuel * delivered_vapor_fraction)
    wall_film_mass_kg = float(m_fuel * wall_film_fraction)

    return MixturePreparationResult(
        delivered_vapor_fraction=delivered_vapor_fraction,
        wall_film_fraction=wall_film_fraction,
        mixture_inhomogeneity=mixture_inhomogeneity,
        mixture_homogeneity=mixture_homogeneity,
        residence_time_s=residence_time_s,
        plenum_residence_time_s=plenum_residence_time_s,
        charge_temp_k=charge_temp_k,
        delivered_fuel_vapor_mass_kg=delivered_fuel_vapor_mass_kg,
        wall_film_mass_kg=wall_film_mass_kg,
        fuel_name=str(fuel.fuel_name),
        profile_id=str(profile.profile_id),
        profile_version=str(profile.profile_version),
        injector_type=str(hw.injector_type),
    )


__all__ = ["MixturePreparationResult", "evaluate_mixture_preparation"]
