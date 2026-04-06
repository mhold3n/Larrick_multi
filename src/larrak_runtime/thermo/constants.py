"""Thermo constant packs and citation-aware loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_THERMO_CONSTANTS_PATH = Path("data/thermo/literature_constants_v1.json")
DEFAULT_THERMO_CONSTANTS_MANIFEST_PATH = Path("data/thermo/literature_constants_manifest_v1.json")
DEFAULT_THERMO_ANCHOR_MANIFEST_PATH = Path("data/thermo/anchor_manifest_v1.json")


@dataclass(frozen=True)
class ThermoConstants:
    """Numerical constant set used by two-zone thermo model."""

    version: str
    gamma_u: float
    gamma_b: float
    r_u: float
    r_b: float
    cv_u: float
    cv_b: float
    cd_intake: float
    cd_exhaust: float
    fuel_lhv: float
    afr_stoich: float
    o2_required_per_fuel: float
    o2_mass_fraction_air: float
    t_intake_k: float
    t_residual_k: float
    wall_temp_k: float
    wiebe_a1: float
    wiebe_m1: float
    wiebe_a2: float
    wiebe_m2: float
    wiebe_split: float
    vapor_tau_ref_s: float
    vapor_temp_exponent: float
    vapor_pressure_exponent: float
    h_wall_u: float
    h_wall_b: float
    p_limit_bar: float
    efficiency_upper_bound: float
    nn_correction_k: float
    nn_correction_beta: float


def _as_path(raw: str | Path | None, default_path: Path) -> Path:
    if raw is None:
        return default_path
    text = str(raw).strip()
    if not text:
        return default_path
    return Path(text)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required thermo file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _extract_constant(constants_payload: dict[str, Any], symbol: str) -> float:
    if symbol not in constants_payload:
        raise ValueError(f"Missing required thermo constant '{symbol}'")
    rec = constants_payload[symbol]
    if not isinstance(rec, dict):
        raise ValueError(f"Constant record for '{symbol}' must be an object")
    for required in ("value", "units", "source", "valid_range"):
        if required not in rec:
            raise ValueError(f"Constant '{symbol}' missing required field '{required}'")

    value = float(rec["value"])
    if not np.isfinite(value):
        raise ValueError(f"Constant '{symbol}' has non-finite value")

    v_range = rec["valid_range"]
    if not (isinstance(v_range, list) and len(v_range) == 2):
        raise ValueError(f"Constant '{symbol}' valid_range must be [min, max]")
    v_min = float(v_range[0])
    v_max = float(v_range[1])
    if not (v_min <= value <= v_max):
        raise ValueError(f"Constant '{symbol}'={value} out of range [{v_min}, {v_max}]")
    return value


def load_thermo_constants(
    *,
    constants_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> ThermoConstants:
    """Load and validate thermo constants + citation manifest."""
    constants_file = _as_path(constants_path, DEFAULT_THERMO_CONSTANTS_PATH)
    manifest_file = _as_path(manifest_path, DEFAULT_THERMO_CONSTANTS_MANIFEST_PATH)

    constants_payload = _load_json(constants_file)
    manifest_payload = _load_json(manifest_file)

    if "citations" not in manifest_payload or not isinstance(manifest_payload["citations"], list):
        raise ValueError(f"Thermo constants manifest missing citations list: {manifest_file}")

    version = str(constants_payload.get("version", "")).strip()
    if not version:
        raise ValueError(f"Thermo constants file missing version: {constants_file}")

    records = constants_payload.get("constants")
    if not isinstance(records, dict):
        raise ValueError("Thermo constants file must contain object field 'constants'")

    return ThermoConstants(
        version=version,
        gamma_u=_extract_constant(records, "gamma_u"),
        gamma_b=_extract_constant(records, "gamma_b"),
        r_u=_extract_constant(records, "r_u"),
        r_b=_extract_constant(records, "r_b"),
        cv_u=_extract_constant(records, "cv_u"),
        cv_b=_extract_constant(records, "cv_b"),
        cd_intake=_extract_constant(records, "cd_intake"),
        cd_exhaust=_extract_constant(records, "cd_exhaust"),
        fuel_lhv=_extract_constant(records, "fuel_lhv"),
        afr_stoich=_extract_constant(records, "afr_stoich"),
        o2_required_per_fuel=_extract_constant(records, "o2_required_per_fuel"),
        o2_mass_fraction_air=_extract_constant(records, "o2_mass_fraction_air"),
        t_intake_k=_extract_constant(records, "t_intake_k"),
        t_residual_k=_extract_constant(records, "t_residual_k"),
        wall_temp_k=_extract_constant(records, "wall_temp_k"),
        wiebe_a1=_extract_constant(records, "wiebe_a1"),
        wiebe_m1=_extract_constant(records, "wiebe_m1"),
        wiebe_a2=_extract_constant(records, "wiebe_a2"),
        wiebe_m2=_extract_constant(records, "wiebe_m2"),
        wiebe_split=_extract_constant(records, "wiebe_split"),
        vapor_tau_ref_s=_extract_constant(records, "vapor_tau_ref_s"),
        vapor_temp_exponent=_extract_constant(records, "vapor_temp_exponent"),
        vapor_pressure_exponent=_extract_constant(records, "vapor_pressure_exponent"),
        h_wall_u=_extract_constant(records, "h_wall_u"),
        h_wall_b=_extract_constant(records, "h_wall_b"),
        p_limit_bar=_extract_constant(records, "p_limit_bar"),
        efficiency_upper_bound=_extract_constant(records, "efficiency_upper_bound"),
        nn_correction_k=_extract_constant(records, "nn_correction_k"),
        nn_correction_beta=_extract_constant(records, "nn_correction_beta"),
    )


def load_anchor_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load benchmark anchor manifest; returns defaults if missing optional keys."""
    manifest_file = _as_path(path, DEFAULT_THERMO_ANCHOR_MANIFEST_PATH)
    payload = _load_json(manifest_file)

    envelope = payload.get("validated_envelope", {})
    thresholds = payload.get("thresholds", {})
    anchors = payload.get("anchors", [])

    if not isinstance(envelope, dict):
        raise ValueError("validated_envelope must be an object")
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be an object")
    if not isinstance(anchors, list):
        raise ValueError("anchors must be a list")

    anchors_validated: list[dict[str, Any]] = []
    for idx, rec in enumerate(anchors):
        if not isinstance(rec, dict):
            raise ValueError(f"Anchor at index {idx} must be an object")
        if "rpm" not in rec or "torque" not in rec:
            raise ValueError(f"Anchor at index {idx} must contain 'rpm' and 'torque'")
        rpm = float(rec["rpm"])
        torque = float(rec["torque"])
        if not np.isfinite(rpm) or not np.isfinite(torque):
            raise ValueError(f"Anchor at index {idx} has non-finite rpm/torque")
        if rpm <= 0.0:
            raise ValueError(f"Anchor at index {idx} has non-positive rpm={rpm}")
        if torque < 0.0:
            raise ValueError(f"Anchor at index {idx} has negative torque={torque}")
        anchors_validated.append(
            {
                "rpm": rpm,
                "torque": torque,
                "label": str(rec.get("label", "")).strip(),
                "source": str(rec.get("source", "")).strip(),
                "provenance": rec.get("provenance", {}),
            }
        )
    return {
        "version": str(payload.get("version", "")),
        "validated_envelope": {
            "rpm_min": float(envelope.get("rpm_min", 0.0)),
            "rpm_max": float(envelope.get("rpm_max", 1e9)),
            "torque_min": float(envelope.get("torque_min", 0.0)),
            "torque_max": float(envelope.get("torque_max", 1e9)),
        },
        "thresholds": {
            "delta_m_air_rel_max": float(thresholds.get("delta_m_air_rel_max", 0.10)),
            "delta_residual_abs_max": float(thresholds.get("delta_residual_abs_max", 0.05)),
            "delta_scavenging_abs_max": float(thresholds.get("delta_scavenging_abs_max", 0.08)),
        },
        "anchors": anchors_validated,
    }
