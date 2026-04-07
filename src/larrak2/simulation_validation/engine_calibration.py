"""Bridge integrated engine runs to live spray/reacting validation targets."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from larrak_runtime.core.evaluator import evaluate_candidate
from larrak_runtime.core.types import BreathingConfig, EvalContext
from larrak_runtime.thermo.validation import ThermoValidationError

from ..orchestration.adapters.simulation_adapter import (
    candidate_openfoam_handoff_bundle,
    candidate_openfoam_params,
)
from .cantera_mechanisms import convert_chemkin_to_yaml

DEFAULT_ENGINE_PACKAGE_MANIFEST = Path(
    "mechanisms/openfoam/v2512/chem323_reduced/package_manifest.json"
)
ENGINE_TRACKED_SPECIES = ("IC8H18", "O2", "N2", "CO2", "H2O")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _wrap_cycle_angle_deg(angle_deg: float) -> float:
    return float(((float(angle_deg) + 180.0) % 360.0) - 180.0)


def _normalize_species_mole_fractions(
    composition: dict[str, Any],
    *,
    floor: float = 0.0,
) -> dict[str, float]:
    normalized = {
        str(species): max(float(value), float(floor))
        for species, value in dict(composition or {}).items()
        if float(value) > 0.0
    }
    total = sum(normalized.values())
    if total <= 0.0:
        return {}
    return {species: value / total for species, value in normalized.items()}


def _blend_species_mole_fractions(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    primary_weight: float,
) -> dict[str, float]:
    alpha = _clamp(float(primary_weight), 0.0, 1.0)
    merged: dict[str, float] = {}
    for species in set(primary) | set(secondary):
        merged[str(species)] = alpha * float(primary.get(species, 0.0)) + (1.0 - alpha) * float(
            secondary.get(species, 0.0)
        )
    return _normalize_species_mole_fractions(merged)


def _build_eval_context_from_engine_params(
    tuned_params: dict[str, Any],
    *,
    rpm: float,
    torque: float,
    fidelity: int,
) -> EvalContext:
    breathing = BreathingConfig(
        bore_mm=float(tuned_params.get("bore_mm", 80.0)),
        stroke_mm=float(tuned_params.get("stroke_mm", 90.0)),
        intake_port_area_m2=float(tuned_params.get("intake_port_area_m2", 4.0e-4)),
        exhaust_port_area_m2=float(tuned_params.get("exhaust_port_area_m2", 4.0e-4)),
        p_manifold_Pa=float(tuned_params.get("p_manifold_Pa", 101325.0)),
        p_back_Pa=float(tuned_params.get("p_back_Pa", 101325.0)),
        overlap_deg=float(tuned_params.get("overlap_deg", 0.0)),
        intake_open_deg=float(tuned_params.get("intake_open_deg", -125.0)),
        intake_close_deg=float(tuned_params.get("intake_close_deg", -85.0)),
        exhaust_open_deg=float(tuned_params.get("exhaust_open_deg", 35.0)),
        exhaust_close_deg=float(tuned_params.get("exhaust_close_deg", 85.0)),
        fuel_name="gasoline",
        valve_timing_mode="override",
    )
    return EvalContext(
        rpm=float(rpm),
        torque=float(torque),
        fidelity=int(fidelity),
        breathing=breathing,
        surrogate_validation_mode="warn",
        thermo_symbolic_mode="warn",
        strict_data=False,
    )


def _measured_targets(regime_cfg: dict[str, Any]) -> dict[str, float]:
    simulation_data = dict(regime_cfg.get("simulation_data", {}) or {})
    targets: dict[str, float] = {}
    for key, value in simulation_data.items():
        if not key.endswith("_measured"):
            continue
        targets[key[: -len("_measured")]] = float(value)
    return targets


def _latest_log_summary(engine_case_dir: str | Path) -> dict[str, float]:
    root = Path(engine_case_dir)
    summaries = sorted(
        root.glob("logSummary.*.dat"),
        key=lambda path: float(path.name[len("logSummary.") : -len(".dat")]),
    )
    if not summaries:
        return {}
    latest = summaries[-1]
    rows = [
        line.strip()
        for line in latest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        return {}
    crank, pressure, temperature, velocity = rows[-1].split()
    return {
        "time_s": float(latest.name[len("logSummary.") : -len(".dat")]),
        "crank_angle_deg": float(crank),
        "mean_pressure_Pa": float(pressure),
        "mean_temperature_K": float(temperature),
        "mean_velocity_magnitude_m_s": float(velocity),
    }


def build_engine_calibration_report(
    *,
    suite_config_path: str | Path,
    engine_case_dir: str | Path,
) -> dict[str, Any]:
    suite_cfg = _load_json(suite_config_path)
    spray_targets = _measured_targets(dict(suite_cfg.get("regimes", {}).get("spray", {}) or {}))
    reacting_targets = _measured_targets(
        dict(suite_cfg.get("regimes", {}).get("reacting_flow", {}) or {})
    )
    engine_summary = _latest_log_summary(engine_case_dir)
    engine_metrics_path = Path(engine_case_dir) / "openfoam_metrics.json"
    engine_metrics = _load_json(engine_metrics_path) if engine_metrics_path.exists() else {}

    spray_angle = float(spray_targets.get("vapor_spreading_angle_deg_sprayG", 90.0))
    homogeneity = min(1.0, max(0.05, spray_angle / 120.0))
    reacting_temperature = float(
        reacting_targets.get("gas_temperature_K_iso_octane_reacting", 980.0)
    )
    reacting_velocity = float(reacting_targets.get("bulk_velocity_m_s_iso_octane_reacting", 15.0))

    report = {
        "suite_id": str(suite_cfg.get("suite_id", "")),
        "engine_case_dir": str(Path(engine_case_dir)),
        "engine_latest_summary": engine_summary,
        "engine_metrics": engine_metrics,
        "spray_targets": spray_targets,
        "reacting_targets": reacting_targets,
        "recommended_engine_seed_updates": {
            "handoff_velocity_m_s": reacting_velocity,
            "T_residual_K": reacting_temperature,
            "mixture_homogeneity_index": homogeneity,
            "vapor_fraction": homogeneity,
            "spray_penetration_target_mm": float(
                spray_targets.get("liquid_penetration_max_mm_sprayG", 0.0)
            ),
            "spray_smd_target_um": float(spray_targets.get("droplet_smd_um_sprayG_z15mm", 0.0)),
        },
        "gap_report": {},
    }

    gap_report: dict[str, float] = {}
    if engine_summary:
        gap_report["mean_temperature_to_reacting_target_K"] = float(
            engine_summary.get("mean_temperature_K", 0.0) - reacting_temperature
        )
        gap_report["mean_velocity_to_reacting_target_m_s"] = float(
            engine_summary.get("mean_velocity_magnitude_m_s", 0.0) - reacting_velocity
        )
        gap_report["mean_pressure_to_reacting_target_Pa"] = float(
            engine_summary.get("mean_pressure_Pa", 0.0)
            - float(
                suite_cfg.get("regimes", {})
                .get("reacting_flow", {})
                .get("case_spec", {})
                .get("operating_point", {})
                .get("pressure_bar", 15.0)
            )
            * 1.0e5
        )
    if "residual_fraction" in engine_metrics:
        gap_report["residual_fraction_to_homogeneity_proxy"] = float(
            engine_metrics["residual_fraction"] - (1.0 - homogeneity)
        )
    report["gap_report"] = gap_report
    return report


def propose_engine_tuning_params(
    *,
    calibration_report: dict[str, Any],
    baseline_params: dict[str, Any],
) -> dict[str, Any]:
    """Return a deterministic tuned parameter set from a bridge report."""
    tuned = dict(baseline_params)
    engine_summary = dict(calibration_report.get("engine_latest_summary", {}) or {})
    recommended = dict(calibration_report.get("recommended_engine_seed_updates", {}) or {})
    reacting_targets = dict(calibration_report.get("reacting_targets", {}) or {})
    engine_metrics = dict(calibration_report.get("engine_metrics", {}) or {})

    target_pressure = float(
        reacting_targets.get(
            "pressure_Pa_target",
            reacting_targets.get("gas_pressure_Pa_iso_octane_reacting", 15.0e5),
        )
    )
    current_pressure = max(float(engine_summary.get("mean_pressure_Pa", 1.0e5)), 1.0e5)
    current_temperature = float(
        engine_summary.get("mean_temperature_K", tuned.get("T_intake_K", 300.0))
    )
    current_velocity = max(
        0.0,
        float(
            engine_summary.get(
                "mean_velocity_magnitude_m_s",
                tuned.get("handoff_velocity_m_s", recommended.get("handoff_velocity_m_s", 0.0)),
            )
        ),
    )
    target_temperature = float(
        reacting_targets.get("gas_temperature_K_iso_octane_reacting", 1400.0)
    )
    target_velocity = float(recommended.get("handoff_velocity_m_s", 44.0))
    target_residual = 1.0 - float(recommended.get("mixture_homogeneity_index", 0.75))
    current_residual = float(
        engine_metrics.get("residual_fraction", tuned.get("residual_fraction_seed", 0.08))
    )

    pressure_gap = max(0.0, target_pressure - current_pressure)
    temperature_gap = max(0.0, target_temperature - current_temperature)
    velocity_gap = current_velocity - target_velocity

    mean_pressure_target = _clamp(
        current_pressure + 0.06 * pressure_gap,
        1.25e5,
        2.25e5,
    )

    # When the engine is already far too fast, reduce the inlet-outlet split and
    # let exhaust back-pressure carry most of the mean-pressure increase.
    if velocity_gap > 100.0:
        pressure_split = _clamp(mean_pressure_target * 0.01, 2.0e3, 6.0e3)
        tuned["p_manifold_Pa"] = _clamp(mean_pressure_target - 0.5 * pressure_split, 1.1e5, 2.1e5)
        tuned["p_back_Pa"] = _clamp(
            mean_pressure_target + 0.5 * pressure_split, tuned["p_manifold_Pa"], 2.25e5
        )
        tuned["handoff_velocity_m_s"] = _clamp(
            min(float(tuned.get("handoff_velocity_m_s", target_velocity)), target_velocity * 0.25),
            0.0,
            12.0,
        )
        tuned["intake_port_area_m2"] = max(float(tuned.get("intake_port_area_m2", 4.0e-4)), 4.5e-4)
        tuned["exhaust_port_area_m2"] = max(
            float(tuned.get("exhaust_port_area_m2", 4.0e-4)), 4.5e-4
        )
        tuned["intake_open_deg"] = -125.0
        tuned["intake_close_deg"] = -85.0
        tuned["exhaust_open_deg"] = 35.0
        tuned["exhaust_close_deg"] = 85.0
    else:
        pressure_split = _clamp(mean_pressure_target * 0.04, 5.0e3, 2.0e4)
        tuned["p_manifold_Pa"] = _clamp(mean_pressure_target + 0.5 * pressure_split, 1.1e5, 2.25e5)
        tuned["p_back_Pa"] = _clamp(
            mean_pressure_target - 0.5 * pressure_split, 1.0e5, tuned["p_manifold_Pa"]
        )
        tuned["handoff_velocity_m_s"] = target_velocity
        tuned["intake_port_area_m2"] = float(tuned.get("intake_port_area_m2", 4.0e-4))
        tuned["exhaust_port_area_m2"] = float(tuned.get("exhaust_port_area_m2", 4.0e-4))
        tuned.setdefault("intake_open_deg", -150.0)
        tuned.setdefault("intake_close_deg", -60.0)
        tuned.setdefault("exhaust_open_deg", 20.0)
        tuned.setdefault("exhaust_close_deg", 110.0)

    tuned["T_intake_K"] = _clamp(
        max(float(tuned.get("T_intake_K", 300.0)), current_temperature + 0.05 * temperature_gap),
        325.0,
        420.0,
    )
    tuned["T_residual_K"] = _clamp(
        max(float(tuned.get("T_residual_K", 900.0)), current_temperature + 0.30 * temperature_gap),
        tuned["T_intake_K"] + 150.0,
        1200.0,
    )
    tuned["engine_wall_temperature_K"] = _clamp(
        max(
            float(tuned.get("engine_wall_temperature_K", 600.0)),
            current_temperature + 0.18 * temperature_gap,
        ),
        500.0,
        600.0,
    )
    tuned["engine_max_temperature_K"] = max(
        float(tuned.get("engine_max_temperature_K", 1350.0)),
        float(tuned["T_residual_K"]) + 200.0,
        1350.0,
    )
    tuned["residual_fraction_seed"] = _clamp(
        max(
            float(tuned.get("residual_fraction_seed", 0.08)),
            current_residual + 0.5 * max(target_residual - current_residual, 0.0),
        ),
        0.10,
        0.20,
    )
    tuned.setdefault("engine_proof_mode", "full_cycle_breathing")
    tuned.setdefault("deltaT", 1.0e-4)

    return tuned


def compare_engine_calibration_reports(
    baseline_report: dict[str, Any],
    tuned_report: dict[str, Any],
) -> dict[str, float]:
    """Report whether absolute gaps shrank between two calibration reports."""
    baseline_gaps = dict(baseline_report.get("gap_report", {}) or {})
    tuned_gaps = dict(tuned_report.get("gap_report", {}) or {})
    comparison: dict[str, float] = {}
    for key, baseline_value in baseline_gaps.items():
        if key not in tuned_gaps:
            continue
        comparison[key] = abs(float(baseline_value)) - abs(float(tuned_gaps[key]))
    return comparison


def _load_cantera_module() -> Any:
    try:
        return importlib.import_module("cantera")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cantera runtime is required for Cantera-derived engine initialization. "
            "Install the optional combustion extra with `pip install .[combustion]`."
        ) from exc


def _resolve_cantera_mechanism_from_package_manifest(
    package_manifest_path: str | Path = DEFAULT_ENGINE_PACKAGE_MANIFEST,
) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(package_manifest_path)
    manifest = _load_json(manifest_path)
    generated_yaml = Path(str(manifest.get("generated_yaml_path", "")).strip())
    if generated_yaml and generated_yaml.exists():
        return manifest, generated_yaml

    raw_files = [Path(str(item)) for item in list(manifest.get("source_raw_files", []) or [])]
    if len(raw_files) < 2:
        raise FileNotFoundError(
            f"Package manifest '{manifest_path}' does not include enough source_raw_files to rebuild Cantera YAML"
        )

    mechanism_file = next(
        (
            path
            for path in raw_files
            if path.suffix.lower() in {".inp", ".txt"} and "chem" in path.name.lower()
        ),
        raw_files[0],
    )
    thermo_file = next((path for path in raw_files if "therm" in path.name.lower()), None)
    transport_file = next((path for path in raw_files if "transport" in path.name.lower()), None)
    if thermo_file is None:
        raise FileNotFoundError(
            f"Package manifest '{manifest_path}' is missing a thermo source file"
        )
    if not generated_yaml:
        generated_yaml = Path("outputs/validation_runtime/mechanisms/chem323_reduced.yaml")
    convert_chemkin_to_yaml(
        input_file=mechanism_file,
        thermo_file=thermo_file,
        transport_file=transport_file,
        output_file=generated_yaml,
        sanitizer_profile=str(manifest.get("sanitizer_profile", "")),
        quiet=True,
    )
    return manifest, generated_yaml


def _derive_cantera_handoff_from_seed(
    *,
    seed_bundle: dict[str, Any],
    diagnostic_seed: dict[str, Any],
    package_manifest_path: str | Path,
    max_integration_time_s: float,
    preignition_fraction_of_ignition_delay: float,
    max_species: int,
    min_species_mole_fraction: float,
    base_pressure_Pa: float,
    base_temperature_K: float,
    initial_temperature_boost_K: float = 0.0,
    bundle_id: str = "preignition_cantera_seed_v1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, mechanism_path = _resolve_cantera_mechanism_from_package_manifest(
        package_manifest_path
    )
    ct = _load_cantera_module()

    composition = _normalize_species_mole_fractions(
        dict(seed_bundle.get("species_mole_fractions", {}) or {})
    )
    gas = ct.Solution(str(mechanism_path), transport_model=None)
    gas.TPX = (
        max(float(seed_bundle["temperature_K"]) + float(initial_temperature_boost_K), 250.0),
        max(float(seed_bundle["pressure_Pa"]), 1.0),
        composition,
    )
    reactor = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reactor])

    history: list[dict[str, Any]] = []
    last_time = 0.0
    last_pressure = float(reactor.thermo.P)
    max_pressure_rate = float("-inf")
    ignition_time_s: float | None = None
    while float(net.time) < float(max_integration_time_s):
        net.step()
        current_time = float(net.time)
        current_pressure = float(reactor.thermo.P)
        current_temperature = float(reactor.thermo.T)
        dt = max(current_time - last_time, 1.0e-12)
        pressure_rate = (current_pressure - last_pressure) / dt
        if pressure_rate > max_pressure_rate:
            max_pressure_rate = pressure_rate
            ignition_time_s = current_time
        history.append(
            {
                "time_s": current_time,
                "pressure_Pa": current_pressure,
                "temperature_K": current_temperature,
                "velocity_m_s": 0.0,
            }
        )
        last_time = current_time
        last_pressure = current_pressure

    if not history:
        raise RuntimeError("Cantera pre-ignition derivation produced no reactor history")

    if ignition_time_s is None or ignition_time_s <= 0.0 or max_pressure_rate <= 0.0:
        target_time_s = float(max_integration_time_s)
    else:
        target_time_s = min(
            float(max_integration_time_s),
            max(1.0e-7, float(ignition_time_s) * float(preignition_fraction_of_ignition_delay)),
        )

    sample = min(history, key=lambda item: abs(float(item["time_s"]) - target_time_s))
    gas.TP = float(sample["temperature_K"]), float(sample["pressure_Pa"])
    gas.X = composition
    if sample["time_s"] != history[-1]["time_s"]:
        gas.TPX = (
            max(float(seed_bundle["temperature_K"]) + float(initial_temperature_boost_K), 250.0),
            max(float(seed_bundle["pressure_Pa"]), 1.0),
            composition,
        )
        reactor = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([reactor])
        while float(net.time) < float(sample["time_s"]):
            net.step()
        gas = reactor.thermo

    mole_fractions = {
        gas.species_name(i): float(gas.X[i])
        for i in range(int(gas.n_species))
        if float(gas.X[i]) >= float(min_species_mole_fraction)
    }
    ranked = sorted(mole_fractions.items(), key=lambda item: item[1], reverse=True)
    reduced_species = dict(ranked[: int(max_species)])
    for species in ("IC8H18", "O2", "N2", "CO2", "H2O", "OH", "CO", "HO2", "H2", "CH2O"):
        if species in mole_fractions:
            reduced_species[species] = mole_fractions[species]
    reduced_species = _normalize_species_mole_fractions(reduced_species)

    total_mass_kg = float(seed_bundle.get("total_mass_kg", 4.0e-4))
    total_energy_J = total_mass_kg * max(float(getattr(gas, "cp_mass", 1000.0)) * float(gas.T), 1.0)
    bundle = {
        **seed_bundle,
        "bundle_id": str(bundle_id),
        "pressure_Pa": float(gas.P),
        "temperature_K": float(gas.T),
        "species_mole_fractions": reduced_species,
        "total_mass_kg": total_mass_kg,
        "total_energy_J": total_energy_J,
    }
    diagnostic = {
        "package_manifest_path": str(package_manifest_path),
        "package_id": str(manifest.get("package_id", "")),
        "package_hash": str(manifest.get("package_hash", "")),
        "cantera_mechanism_path": str(mechanism_path),
        "base_pressure_Pa": float(base_pressure_Pa),
        "base_temperature_K": float(base_temperature_K),
        "seed_bundle": dict(seed_bundle),
        "diagnostic_seed": dict(diagnostic_seed),
        "sample_time_s": float(sample["time_s"]),
        "target_time_s": float(target_time_s),
        "estimated_ignition_time_s": None if ignition_time_s is None else float(ignition_time_s),
        "max_pressure_rate_Pa_s": float(max_pressure_rate),
        "initial_temperature_boost_K": float(initial_temperature_boost_K),
        "final_time_s": float(history[-1]["time_s"]),
        "final_pressure_Pa": float(history[-1]["pressure_Pa"]),
        "final_temperature_K": float(history[-1]["temperature_K"]),
        "history_tail": history[-10:],
    }
    return bundle, diagnostic


def derive_cantera_preignition_handoff_bundle(
    *,
    calibration_report: dict[str, Any],
    tuned_params: dict[str, Any],
    package_manifest_path: str | Path = DEFAULT_ENGINE_PACKAGE_MANIFEST,
    max_integration_time_s: float = 2.0e-4,
    preignition_fraction_of_ignition_delay: float = 0.25,
    max_species: int = 24,
    min_species_mole_fraction: float = 1.0e-8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive a closed-valve pre-ignition bundle from a Cantera reactor evolution."""
    proposed_seed = propose_preignition_handoff_bundle(
        calibration_report=calibration_report,
        tuned_params=tuned_params,
    )
    engine_summary = dict(calibration_report.get("engine_latest_summary", {}) or {})
    engine_metrics = dict(calibration_report.get("engine_metrics", {}) or {})
    composition = dict(proposed_seed.get("species_mole_fractions", {}) or {})
    base_temperature = max(
        float(
            engine_metrics.get(
                "mass_weighted_temperature_K",
                engine_summary.get("mean_temperature_K", proposed_seed["temperature_K"]),
            )
        ),
        float(tuned_params.get("T_intake_K", proposed_seed["temperature_K"])),
    )
    base_pressure = max(
        float(engine_summary.get("mean_pressure_Pa", proposed_seed["pressure_Pa"])),
        float(tuned_params.get("p_manifold_Pa", proposed_seed["pressure_Pa"])),
    )
    ct = _load_cantera_module()
    _, mechanism_path = _resolve_cantera_mechanism_from_package_manifest(package_manifest_path)
    gas = ct.Solution(str(mechanism_path), transport_model=None)
    gas.TPX = (base_temperature, base_pressure, composition)
    target_pressure = float(proposed_seed["pressure_Pa"])
    gas.SP = gas.s, target_pressure
    compressed_temperature = float(gas.T)
    compressed_pressure = float(gas.P)
    seed = {
        **proposed_seed,
        "pressure_Pa": compressed_pressure,
        "temperature_K": compressed_temperature,
    }
    return _derive_cantera_handoff_from_seed(
        seed_bundle=seed,
        diagnostic_seed={"seed_origin": "compressed_calibration_report_v1"},
        package_manifest_path=package_manifest_path,
        max_integration_time_s=max_integration_time_s,
        preignition_fraction_of_ignition_delay=preignition_fraction_of_ignition_delay,
        max_species=max_species,
        min_species_mole_fraction=min_species_mole_fraction,
        base_pressure_Pa=base_pressure,
        base_temperature_K=base_temperature,
    )


def derive_two_zone_cantera_preignition_handoff_bundle(
    *,
    candidate_x: list[float] | tuple[float, ...] | np.ndarray,
    calibration_report: dict[str, Any],
    tuned_params: dict[str, Any],
    candidate_id: str = "truth_00",
    rpm: float | None = None,
    torque: float | None = None,
    fidelity: int = 1,
    package_manifest_path: str | Path = DEFAULT_ENGINE_PACKAGE_MANIFEST,
    pre_spark_deg: float = 2.0,
    max_integration_time_s: float | None = None,
    preignition_fraction_of_ignition_delay: float = 0.15,
    spark_temperature_boost_K: float | None = None,
    max_species: int = 24,
    min_species_mole_fraction: float = 1.0e-8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive a closed-valve Cantera seed from a real two-zone ignition-stage evaluation."""
    x = np.asarray(candidate_x, dtype=np.float64).reshape(-1)
    resolved_rpm = float(rpm if rpm is not None else tuned_params.get("rpm", 1800.0))
    resolved_torque = float(torque if torque is not None else tuned_params.get("torque", 80.0))
    ctx = _build_eval_context_from_engine_params(
        tuned_params,
        rpm=resolved_rpm,
        torque=resolved_torque,
        fidelity=fidelity,
    )
    candidate = {"id": str(candidate_id), "x": x}
    eval_diag: dict[str, Any]
    try:
        eval_result = evaluate_candidate(x, ctx)
        eval_diag = dict(eval_result.diag or {})
    except ThermoValidationError as exc:
        thermo_payload = dict(getattr(exc, "payload", {}) or {})
        eval_diag = {"thermo": thermo_payload}
    thermo_diag = dict(eval_diag.get("thermo", {}) or {})
    ignition_stage = dict(thermo_diag.get("ignition_stage", {}) or {})
    mixture_preparation = dict(thermo_diag.get("mixture_preparation", {}) or {})
    if not ignition_stage:
        raise RuntimeError("Two-zone thermo evaluation did not produce ignition_stage diagnostics")

    openfoam_params = candidate_openfoam_params(candidate, ctx, eval_diag=eval_diag)
    base_bundle = candidate_openfoam_handoff_bundle(
        candidate,
        ctx,
        openfoam_params=openfoam_params,
        eval_diag=eval_diag,
    )
    proposed_seed = propose_preignition_handoff_bundle(
        calibration_report=calibration_report,
        tuned_params=tuned_params,
    )

    spark_absolute_deg = float(
        ignition_stage.get("spark_absolute_deg", proposed_seed["cycle_coordinate_deg"])
    )
    cycle_coordinate_deg = _wrap_cycle_angle_deg(spark_absolute_deg - float(pre_spark_deg))
    engine_end_angle_deg = float(tuned_params.get("engine_end_angle_deg", 20.0))
    window_span_deg = max(4.0, abs(engine_end_angle_deg - cycle_coordinate_deg))
    window_time_s = window_span_deg / 360.0 * 60.0 / max(resolved_rpm, 1.0)
    integration_limit_s = float(
        max_integration_time_s
        if max_integration_time_s is not None
        else min(
            max(float(ignition_stage.get("ignition_delay_spark_s", 1.0e-3)) * 0.2, 2.5e-4),
            max(window_time_s * 0.35, 2.5e-4),
            1.5e-3,
        )
    )

    primary_weight = _clamp(float(proposed_seed.get("residual_fraction", 0.2)), 0.2, 0.7)
    base_composition = _blend_species_mole_fractions(
        dict(proposed_seed.get("species_mole_fractions", {}) or {}),
        dict(base_bundle.get("species_mole_fractions", {}) or {}),
        primary_weight=primary_weight,
    )
    if "H2O" not in base_composition:
        base_composition = _normalize_species_mole_fractions(
            {**base_composition, "H2O": max(0.5 * float(base_composition.get("CO2", 0.0)), 1.0e-4)}
        )

    ignitability_margin = float(ignition_stage.get("ignitability_margin", 0.0))
    computed_boost = _clamp(max(-ignitability_margin, 0.0) * 200.0, 90.0, 240.0)
    seed = {
        **base_bundle,
        **proposed_seed,
        "bundle_id": "preignition_two_zone_seed_v1",
        "pressure_Pa": float(proposed_seed["pressure_Pa"]),
        "temperature_K": float(proposed_seed["temperature_K"]),
        "species_mole_fractions": base_composition,
        "cycle_coordinate_deg": cycle_coordinate_deg,
        "stage_marker": "pre_ignition_two_zone_cantera",
        "vapor_fraction": _clamp(
            float(
                mixture_preparation.get(
                    "delivered_vapor_fraction",
                    proposed_seed.get("vapor_fraction", 1.0),
                )
            ),
            0.0,
            1.0,
        ),
        "mixture_homogeneity_index": _clamp(
            float(
                mixture_preparation.get(
                    "mixture_homogeneity",
                    proposed_seed.get("mixture_homogeneity_index", 0.8),
                )
            ),
            0.0,
            1.0,
        ),
        # Closed-valve handoff should start nearly quiescent; the reacting target
        # velocity is a downstream observable, not the initial seed state.
        "velocity_m_s": _clamp(
            float(
                base_bundle.get(
                    "velocity_m_s",
                    tuned_params.get("handoff_velocity_m_s", 0.0),
                )
            ),
            0.0,
            5.0,
        ),
        "total_mass_kg": float(
            calibration_report.get("engine_metrics", {}).get(
                "trapped_mass",
                base_bundle.get("total_mass_kg", 4.0e-4),
            )
        ),
        "residual_fraction": float(
            proposed_seed.get("residual_fraction", base_bundle.get("residual_fraction", 0.1))
        ),
    }
    seed["total_energy_J"] = float(
        seed["total_mass_kg"] * max(float(seed["temperature_K"]), 1.0) * 1000.0
    )
    bundle, diagnostic = _derive_cantera_handoff_from_seed(
        seed_bundle=seed,
        diagnostic_seed={
            "seed_origin": "two_zone_ignition_stage_v1",
            "candidate_id": str(candidate_id),
            "candidate_x": x.tolist(),
            "eval_context": {
                "rpm": resolved_rpm,
                "torque": resolved_torque,
                "fidelity": int(fidelity),
            },
            "openfoam_params": dict(openfoam_params),
            "base_bundle": dict(base_bundle),
            "proposed_seed": dict(proposed_seed),
            "ignition_stage": ignition_stage,
            "mixture_preparation": mixture_preparation,
            "cycle_coordinate_deg": cycle_coordinate_deg,
            "integration_limit_s": integration_limit_s,
        },
        package_manifest_path=package_manifest_path,
        max_integration_time_s=integration_limit_s,
        preignition_fraction_of_ignition_delay=preignition_fraction_of_ignition_delay,
        max_species=max_species,
        min_species_mole_fraction=min_species_mole_fraction,
        base_pressure_Pa=float(
            calibration_report.get("engine_latest_summary", {}).get(
                "mean_pressure_Pa",
                seed["pressure_Pa"],
            )
        ),
        base_temperature_K=float(
            calibration_report.get("engine_metrics", {}).get(
                "mass_weighted_temperature_K",
                calibration_report.get("engine_latest_summary", {}).get(
                    "mean_temperature_K",
                    seed["temperature_K"],
                ),
            )
        ),
        initial_temperature_boost_K=float(
            spark_temperature_boost_K if spark_temperature_boost_K is not None else computed_boost
        ),
        bundle_id="preignition_two_zone_cantera_seed_v1",
    )
    full_species = dict(bundle.get("species_mole_fractions", {}) or {})
    projected_species = _normalize_species_mole_fractions(
        {species: full_species.get(species, 0.0) for species in ENGINE_TRACKED_SPECIES}
    )
    bundle["species_mole_fractions"] = projected_species
    diagnostic["engine_projected_species_mole_fractions"] = projected_species
    diagnostic["full_cantera_species_mole_fractions"] = full_species
    return bundle, diagnostic


def propose_preignition_handoff_bundle(
    *,
    calibration_report: dict[str, Any],
    tuned_params: dict[str, Any],
) -> dict[str, Any]:
    """Return a compressed, closed-valve handoff seed for reacting calibration."""
    engine_summary = dict(calibration_report.get("engine_latest_summary", {}) or {})
    engine_metrics = dict(calibration_report.get("engine_metrics", {}) or {})

    base_pressure = max(
        float(engine_summary.get("mean_pressure_Pa", tuned_params.get("p_manifold_Pa", 2.0e5))),
        2.0e5,
    )
    base_temperature = max(
        float(
            engine_metrics.get(
                "mass_weighted_temperature_K",
                engine_summary.get("mean_temperature_K", tuned_params.get("T_intake_K", 420.0)),
            )
        ),
        float(tuned_params.get("T_intake_K", 420.0)),
    )
    residual_fraction = _clamp(
        max(
            float(engine_metrics.get("residual_fraction", 0.0)),
            float(tuned_params.get("residual_fraction_seed", 0.12)),
        ),
        0.18,
        0.30,
    )

    return {
        "bundle_id": "preignition_seed_v1",
        "mechanism_id": "chem323_reduced_v2512",
        "fuel_name": "iso-octane",
        "pressure_Pa": _clamp(base_pressure * 4.5, 7.5e5, 1.0e6),
        "temperature_K": _clamp(base_temperature * 2.4, 800.0, 950.0),
        "residual_fraction": residual_fraction,
        "velocity_m_s": 0.0,
        "cycle_coordinate_deg": -10.0,
        "stage_marker": "pre_ignition_closed_valve",
        "mixture_homogeneity_index": 0.8,
        "vapor_fraction": 1.0,
        "species_mole_fractions": {
            "IC8H18": 0.014,
            "O2": 0.205,
            "N2": 0.743,
            "CO2": 0.020,
            "H2O": 0.018,
        },
    }
