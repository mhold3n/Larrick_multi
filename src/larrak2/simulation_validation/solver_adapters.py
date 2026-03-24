"""Runtime adapter layer for populating validation simulation_data."""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from larrak2.adapters.openfoam import OpenFoamRunner

from .cantera_mechanisms import convert_chemkin_to_yaml
from .handoff import build_handoff_state_chain, compute_handoff_conservation
from .models import ValidationCaseSpec, ValidationDatasetManifest, ValidationRunManifest


@dataclass
class ResolvedSimulationInputs:
    """Resolved runtime inputs for a single regime validation run."""

    simulation_data: dict[str, Any]
    solver_artifacts: dict[str, str] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


def _adapter_config(
    case_spec: ValidationCaseSpec,
    simulation_data: dict[str, Any],
) -> dict[str, Any]:
    case_cfg = dict(case_spec.solver_config or {})
    case_adapter = case_cfg.get("simulation_adapter", {})
    data_adapter = simulation_data.get("simulation_adapter", {})
    if isinstance(data_adapter, dict) and data_adapter:
        merged = dict(case_adapter) if isinstance(case_adapter, dict) else {}
        merged.update(data_adapter)
        return merged
    return dict(case_adapter) if isinstance(case_adapter, dict) else {}


def _read_json_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at '{path}'")
    return payload


def _merge_metric_payload(
    base: dict[str, Any],
    payload: dict[str, Any],
    *,
    metric_ids: list[str],
) -> None:
    for metric_id in metric_ids:
        if metric_id in payload and metric_id not in base:
            base[metric_id] = payload[metric_id]
        measured_key = f"{metric_id}_measured"
        if measured_key in payload and measured_key not in base:
            base[measured_key] = payload[measured_key]

    for key in (
        "mechanism_provenance",
        "metric_mechanism_provenance",
        "spray_provenance",
        "reacting_flow_provenance",
        "full_handoff_provenance",
        "handoff_states",
        "handoff_bundle",
        "chemistry_cache_metadata",
    ):
        if key in payload and key not in base:
            base[key] = payload[key]


def _required_metric_ids(dataset: ValidationDatasetManifest) -> list[str]:
    return [spec.metric_id for spec in dataset.metrics if spec.required]


def _missing_metric_ids(
    simulation_data: dict[str, Any],
    metric_ids: list[str],
) -> list[str]:
    return [metric_id for metric_id in metric_ids if metric_id not in simulation_data]


def _merge_chemistry_offline_cache(
    *,
    cache_path: str,
    dataset: ValidationDatasetManifest,
    simulation_data: dict[str, Any],
    mechanism_file: str,
    adapter_cfg: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    path = Path(cache_path)
    if not path.exists():
        return dict(simulation_data), False
    payload = _read_json_mapping(path)
    merged = dict(simulation_data)
    _merge_metric_payload(
        merged,
        payload,
        metric_ids=[spec.metric_id for spec in dataset.metrics],
    )
    merged["mechanism_provenance"] = {
        **dict(merged.get("mechanism_provenance", {}) or {}),
        "backend": "offline_cache",
        "offline_results_path": cache_path,
        "fuel_name": str(adapter_cfg.get("fuel_name", dataset.fuel_family)),
        "mechanism_file": mechanism_file,
        "mechanism_format": str(adapter_cfg.get("mechanism_format", "")),
    }
    cache_complete = not _missing_metric_ids(merged, _required_metric_ids(dataset))
    return merged, cache_complete


def _write_chemistry_offline_cache(
    *,
    cache_path: str,
    dataset: ValidationDatasetManifest,
    case_spec: ValidationCaseSpec,
    simulation_data: dict[str, Any],
    mechanism_file: str,
    cantera_mechanism_file: str,
    adapter_cfg: dict[str, Any],
) -> str:
    payload: dict[str, Any] = {
        spec.metric_id: simulation_data[spec.metric_id]
        for spec in dataset.metrics
        if spec.metric_id in simulation_data
    }
    payload["mechanism_provenance"] = {
        **dict(simulation_data.get("mechanism_provenance", {}) or {}),
        "backend": "native_cantera",
        "mechanism_file": mechanism_file,
        "cantera_mechanism_file": cantera_mechanism_file,
        "fuel_name": str(adapter_cfg.get("fuel_name", dataset.fuel_family)),
    }
    metric_mechanism_provenance = dict(simulation_data.get("metric_mechanism_provenance", {}) or {})
    if metric_mechanism_provenance:
        payload["metric_mechanism_provenance"] = metric_mechanism_provenance
    payload["chemistry_cache_metadata"] = {
        "cache_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": dataset.dataset_id,
        "case_id": case_spec.case_id,
        "fuel_family": dataset.fuel_family,
        "metric_ids": [spec.metric_id for spec in dataset.metrics],
    }
    output_path = Path(cache_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(output_path)


def _load_cantera():
    if importlib.util.find_spec("cantera") is None:
        raise RuntimeError(
            "Cantera runtime is required for this combustion validation path. "
            "Install the optional combustion extra with `pip install .[combustion]`."
        )
    return importlib.import_module("cantera")


def _prepare_cantera_mechanism(
    mechanism_file: str,
    adapter_cfg: dict[str, Any],
) -> str:
    """Return a Cantera-loadable mechanism path, converting CHEMKIN if needed."""
    mechanism_path = Path(mechanism_file)
    mechanism_format = str(adapter_cfg.get("mechanism_format", "")).strip().lower()
    if (
        mechanism_path.suffix.lower() in {".yaml", ".yml", ".cti", ".xml"}
        and mechanism_format != "chemkin"
    ):
        return mechanism_file
    if mechanism_format not in {"", "chemkin"} and mechanism_path.suffix.lower() not in {
        ".inp",
        ".txt",
    }:
        return mechanism_file

    thermo_file = str(adapter_cfg.get("thermo_file", "")).strip()
    if not thermo_file:
        raise ValueError(
            f"CHEMKIN mechanism '{mechanism_file}' requires thermo_file for ck2yaml conversion"
        )
    thermo_path = Path(thermo_file)
    if not thermo_path.exists():
        raise FileNotFoundError(f"CHEMKIN thermo file not found: {thermo_file}")

    transport_file = str(adapter_cfg.get("transport_file", "")).strip()
    if transport_file and not Path(transport_file).exists():
        raise FileNotFoundError(f"CHEMKIN transport file not found: {transport_file}")

    out_name = str(adapter_cfg.get("generated_yaml_path", "")).strip()
    if not out_name:
        out_name = str(
            Path("outputs") / "validation_runtime" / "mechanisms" / f"{mechanism_path.stem}.yaml"
        )
    out_path = Path(out_name)
    convert_chemkin_to_yaml(
        input_file=mechanism_path,
        thermo_file=thermo_path,
        transport_file=Path(transport_file) if transport_file else None,
        output_file=out_path,
        phase_name=str(adapter_cfg.get("phase_name", "gas")),
        permissive=bool(adapter_cfg.get("permissive", False)),
        quiet=True,
        sanitizer_profile=str(adapter_cfg.get("sanitizer_profile", "")),
    )
    return str(out_path)


def _metric_cantera_adapter_cfg(
    adapter_cfg: dict[str, Any],
    metric_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Merge metric-local Cantera overrides onto the regime adapter config."""
    merged = dict(adapter_cfg)
    for key in (
        "mechanism_file",
        "mechanism_format",
        "thermo_file",
        "transport_file",
        "generated_yaml_path",
        "sanitizer_profile",
        "phase_name",
        "permissive",
        "fuel_name",
        "oxidizer",
    ):
        if key in metric_cfg:
            merged[key] = metric_cfg[key]
    return merged


def _pressure_pa(metric_cfg: dict[str, Any]) -> float:
    if "pressure_Pa" in metric_cfg:
        return float(metric_cfg["pressure_Pa"])
    if "pressure_bar" in metric_cfg:
        return float(metric_cfg["pressure_bar"]) * 1.0e5
    return 101325.0


def _units_scale_from_seconds(units: str) -> float:
    normalized = str(units).strip().lower()
    if normalized == "ms":
        return 1.0e3
    if normalized == "us":
        return 1.0e6
    return 1.0


def _compute_cantera_ignition_delay(
    ct: Any,
    *,
    mechanism_file: str,
    fuel: str,
    oxidizer: dict[str, float],
    metric_cfg: dict[str, Any],
    units: str,
) -> float:
    transport_model = metric_cfg.get("transport_model")
    if transport_model is None:
        gas = ct.Solution(mechanism_file, transport_model=None)
    else:
        gas = ct.Solution(mechanism_file, transport_model=transport_model)
    gas.TP = float(metric_cfg["temperature_K"]), _pressure_pa(metric_cfg)
    gas.set_equivalence_ratio(float(metric_cfg["equivalence_ratio"]), fuel, oxidizer)

    reactor_type = str(metric_cfg.get("reactor_type", "constant_volume_adiabatic"))
    if reactor_type == "constant_pressure":
        reactor = ct.IdealGasConstPressureReactor(gas)
    else:
        reactor = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reactor])

    end_time_s = float(metric_cfg.get("end_time_s", 0.25))
    last_time = 0.0
    last_pressure = float(reactor.thermo.P)
    max_rate = float("-inf")
    ignition_time_s = 0.0
    while net.time < end_time_s:
        net.step()
        current_time = float(net.time)
        current_pressure = float(reactor.thermo.P)
        dt = max(current_time - last_time, 1.0e-12)
        rate = (current_pressure - last_pressure) / dt
        if rate > max_rate:
            max_rate = rate
            ignition_time_s = current_time
        last_time = current_time
        last_pressure = current_pressure

    return ignition_time_s * _units_scale_from_seconds(units)


def _compute_cantera_flame_speed(
    ct: Any,
    *,
    mechanism_file: str,
    fuel: str,
    oxidizer: dict[str, float],
    metric_cfg: dict[str, Any],
) -> float:
    transport_model = metric_cfg.get("transport_model", "mixture-averaged")
    if transport_model is None:
        gas = ct.Solution(mechanism_file, transport_model=None)
    else:
        gas = ct.Solution(mechanism_file, transport_model=str(transport_model))
    gas.TP = float(metric_cfg["unburned_temperature_K"]), _pressure_pa(metric_cfg)
    gas.set_equivalence_ratio(float(metric_cfg["equivalence_ratio"]), fuel, oxidizer)

    width_m = float(metric_cfg.get("width_m", 0.04))
    grid_points = int(metric_cfg.get("grid_points", 0))
    if grid_points >= 2:
        grid = [i * width_m / max(grid_points - 1, 1) for i in range(grid_points)]
        flame = ct.FreeFlame(gas, grid=grid)
    else:
        flame = ct.FreeFlame(gas, width=width_m)

    if transport_model is not None:
        flame.transport_model = str(transport_model)
    max_grid_points = int(metric_cfg.get("max_grid_points", 0) or 0)
    if max_grid_points > 0:
        flame.set_max_grid_points(flame.flame, max_grid_points)
    flame.set_refine_criteria(
        ratio=float(metric_cfg.get("refine_ratio", 20.0)),
        slope=float(metric_cfg.get("refine_slope", 0.5)),
        curve=float(metric_cfg.get("refine_curve", 0.5)),
        prune=float(metric_cfg.get("refine_prune", 0.1)),
    )

    staged_energy = list(metric_cfg.get("staged_energy", []) or [])
    auto = bool(metric_cfg.get("auto", not staged_energy))
    loglevel = int(metric_cfg.get("loglevel", 0))
    refine_grid = bool(metric_cfg.get("refine_grid", True))
    restore_path = str(
        metric_cfg.get("restore_solution_path") or metric_cfg.get("solution_profile_path") or ""
    ).strip()
    solution_name = str(metric_cfg.get("solution_name", "solution")).strip() or "solution"
    if restore_path and Path(restore_path).exists():
        flame.restore(restore_path, name=solution_name, loglevel=loglevel)
    if staged_energy:
        for energy_enabled in staged_energy:
            flame.energy_enabled = bool(energy_enabled)
            flame.solve(loglevel=loglevel, auto=auto, refine_grid=refine_grid)
    else:
        flame.solve(loglevel=loglevel, auto=auto, refine_grid=refine_grid)
    save_path = str(
        metric_cfg.get("save_solution_path") or metric_cfg.get("solution_profile_path") or ""
    ).strip()
    if save_path:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        flame.save(
            str(save_file),
            name=solution_name,
            description="Cached Cantera flame-speed solution",
            overwrite=True,
        )
    return float(flame.velocity[0])


def _resolve_chemistry_inputs(
    dataset: ValidationDatasetManifest,
    case_spec: ValidationCaseSpec,
    simulation_data: dict[str, Any],
    adapter_cfg: dict[str, Any],
) -> ResolvedSimulationInputs:
    merged = dict(simulation_data)
    messages: list[str] = []
    metric_ids = [spec.metric_id for spec in dataset.metrics]
    required_metric_ids = _required_metric_ids(dataset)
    backend = str(adapter_cfg.get("backend", "fixture")).strip().lower()
    fixture_path = str(adapter_cfg.get("fixture_results_path", "")).strip()
    offline_results_path = str(adapter_cfg.get("offline_results_path", "")).strip()
    offline_results_only = bool(adapter_cfg.get("offline_results_only", False))
    refresh_offline_results = bool(adapter_cfg.get("refresh_offline_results", False))
    mechanism_file = str(
        adapter_cfg.get("mechanism_file") or case_spec.solver_config.get("mechanism_file") or ""
    ).strip()

    def _fixture_fallback(message: str) -> ResolvedSimulationInputs:
        if not fixture_path:
            raise RuntimeError(message)
        payload = _read_json_mapping(fixture_path)
        _merge_metric_payload(merged, payload, metric_ids=metric_ids)
        merged["mechanism_provenance"] = {
            **dict(merged.get("mechanism_provenance", {}) or {}),
            "backend": "fixture_fallback",
            "fixture_results_path": fixture_path,
            "fuel_name": str(adapter_cfg.get("fuel_name", dataset.fuel_family)),
            "mechanism_file": mechanism_file,
            "mechanism_format": str(adapter_cfg.get("mechanism_format", "")),
        }
        return ResolvedSimulationInputs(
            simulation_data=merged,
            solver_artifacts={"chemistry_fixture_results": str(Path(fixture_path))},
            messages=[message],
        )

    if offline_results_path and not refresh_offline_results:
        n_before = sum(1 for metric_id in metric_ids if metric_id in merged)
        merged, cache_complete = _merge_chemistry_offline_cache(
            cache_path=offline_results_path,
            dataset=dataset,
            simulation_data=merged,
            mechanism_file=mechanism_file,
            adapter_cfg=adapter_cfg,
        )
        if cache_complete:
            return ResolvedSimulationInputs(
                simulation_data=merged,
                solver_artifacts={"chemistry_offline_results": str(Path(offline_results_path))},
                messages=[f"Loaded chemistry metrics from offline cache '{offline_results_path}'"],
            )
        n_after = sum(1 for metric_id in metric_ids if metric_id in merged)
        if n_after > n_before:
            messages.append(f"Loaded partial chemistry offline cache from '{offline_results_path}'")

    if offline_results_path and offline_results_only:
        if fixture_path:
            return _fixture_fallback(
                "Offline chemistry cache missing or incomplete; chemistry adapter fell back to "
                "fixture results"
            )
        raise RuntimeError(
            f"Offline chemistry results were required but not available at '{offline_results_path}'"
        )

    if backend == "fixture":
        if not fixture_path:
            raise ValueError("Chemistry fixture adapter requires fixture_results_path")
        payload = _read_json_mapping(fixture_path)
        _merge_metric_payload(merged, payload, metric_ids=metric_ids)
        merged.setdefault(
            "mechanism_provenance",
            {
                "backend": "fixture",
                "fixture_results_path": fixture_path,
                "fuel_name": str(adapter_cfg.get("fuel_name", dataset.fuel_family)),
            },
        )
        return ResolvedSimulationInputs(
            simulation_data=merged,
            solver_artifacts={"chemistry_fixture_results": str(Path(fixture_path))},
            messages=[f"Loaded chemistry metrics from fixture '{fixture_path}'"],
        )

    if backend not in {"native_cantera", "auto"}:
        raise ValueError(f"Unsupported chemistry adapter backend '{backend}'")

    cantera_missing = importlib.util.find_spec("cantera") is None
    if cantera_missing and backend == "auto" and fixture_path:
        return _fixture_fallback(
            "Cantera runtime not available; chemistry adapter fell back to fixture results"
        )

    ct = _load_cantera()
    if not mechanism_file:
        raise ValueError("Chemistry Cantera adapter requires mechanism_file")
    cantera_mechanism_file = _prepare_cantera_mechanism(mechanism_file, adapter_cfg)
    fuel = str(adapter_cfg.get("fuel_name", dataset.fuel_family))
    oxidizer = dict(adapter_cfg.get("oxidizer", {"O2": 0.21, "N2": 0.79}) or {})
    metric_cfgs = dict(adapter_cfg.get("metrics", {}) or {})
    prepared_metric_contexts: dict[str, tuple[str, str, str, dict[str, float], str]] = {}
    metric_mechanism_provenance = dict(merged.get("metric_mechanism_provenance", {}) or {})

    try:

        def _checkpoint_cache() -> None:
            if offline_results_path:
                _write_chemistry_offline_cache(
                    cache_path=offline_results_path,
                    dataset=dataset,
                    case_spec=case_spec,
                    simulation_data=merged,
                    mechanism_file=mechanism_file,
                    cantera_mechanism_file=cantera_mechanism_file,
                    adapter_cfg=adapter_cfg,
                )

        for spec in dataset.metrics:
            if spec.metric_id in merged:
                continue
            metric_cfg = dict(metric_cfgs.get(spec.metric_id, {}) or {})
            method = str(metric_cfg.get("method", "")).strip().lower()
            if method == "fixed_value":
                merged[spec.metric_id] = float(metric_cfg["value"])
                _checkpoint_cache()
                continue
            if method == "species_fraction":
                species_source = dict(metric_cfg.get("composition", {}) or {})
                if not species_source:
                    raise ValueError(
                        f"Chemistry species_fraction metric '{spec.metric_id}' requires composition"
                    )
                species_name = str(metric_cfg.get("species", "")).strip()
                if not species_name:
                    raise ValueError(
                        f"Chemistry species_fraction metric '{spec.metric_id}' requires species"
                    )
                merged[spec.metric_id] = float(species_source.get(species_name, 0.0))
                _checkpoint_cache()
                continue
            if method == "ignition_delay":
                metric_adapter_cfg = _metric_cantera_adapter_cfg(adapter_cfg, metric_cfg)
                metric_mechanism_file = str(
                    metric_adapter_cfg.get("mechanism_file") or mechanism_file
                ).strip()
                if not metric_mechanism_file:
                    raise ValueError(
                        f"Chemistry ignition metric '{spec.metric_id}' requires mechanism_file"
                    )
                metric_context_key = json.dumps(
                    {
                        "mechanism_file": metric_mechanism_file,
                        "mechanism_format": str(metric_adapter_cfg.get("mechanism_format", "")),
                        "thermo_file": str(metric_adapter_cfg.get("thermo_file", "")),
                        "transport_file": str(metric_adapter_cfg.get("transport_file", "")),
                        "generated_yaml_path": str(
                            metric_adapter_cfg.get("generated_yaml_path", "")
                        ),
                        "sanitizer_profile": str(metric_adapter_cfg.get("sanitizer_profile", "")),
                        "fuel_name": str(metric_adapter_cfg.get("fuel_name", fuel)),
                        "oxidizer": dict(metric_adapter_cfg.get("oxidizer", oxidizer) or {}),
                    },
                    sort_keys=True,
                )
                if metric_context_key not in prepared_metric_contexts:
                    prepared_metric_contexts[metric_context_key] = (
                        metric_mechanism_file,
                        _prepare_cantera_mechanism(metric_mechanism_file, metric_adapter_cfg),
                        str(metric_adapter_cfg.get("fuel_name", fuel)),
                        dict(metric_adapter_cfg.get("oxidizer", oxidizer) or {}),
                        str(metric_adapter_cfg.get("mechanism_format", "")),
                    )
                (
                    metric_mechanism_source,
                    metric_cantera_mechanism_file,
                    metric_fuel,
                    metric_oxidizer,
                    metric_mechanism_format,
                ) = prepared_metric_contexts[metric_context_key]
                merged[spec.metric_id] = _compute_cantera_ignition_delay(
                    ct,
                    mechanism_file=metric_cantera_mechanism_file,
                    fuel=metric_fuel,
                    oxidizer=metric_oxidizer,
                    metric_cfg=metric_cfg,
                    units=spec.units,
                )
                metric_mechanism_provenance[spec.metric_id] = {
                    "backend": "native_cantera",
                    "mechanism_file": metric_mechanism_source,
                    "cantera_mechanism_file": metric_cantera_mechanism_file,
                    "mechanism_format": metric_mechanism_format,
                    "fuel_name": metric_fuel,
                }
                merged["metric_mechanism_provenance"] = metric_mechanism_provenance
                _checkpoint_cache()
                continue
            if method == "flame_speed":
                metric_adapter_cfg = _metric_cantera_adapter_cfg(adapter_cfg, metric_cfg)
                metric_mechanism_file = str(
                    metric_adapter_cfg.get("mechanism_file") or mechanism_file
                ).strip()
                if not metric_mechanism_file:
                    raise ValueError(
                        f"Chemistry flame-speed metric '{spec.metric_id}' requires mechanism_file"
                    )
                metric_context_key = json.dumps(
                    {
                        "mechanism_file": metric_mechanism_file,
                        "mechanism_format": str(metric_adapter_cfg.get("mechanism_format", "")),
                        "thermo_file": str(metric_adapter_cfg.get("thermo_file", "")),
                        "transport_file": str(metric_adapter_cfg.get("transport_file", "")),
                        "generated_yaml_path": str(
                            metric_adapter_cfg.get("generated_yaml_path", "")
                        ),
                        "sanitizer_profile": str(metric_adapter_cfg.get("sanitizer_profile", "")),
                        "fuel_name": str(metric_adapter_cfg.get("fuel_name", fuel)),
                        "oxidizer": dict(metric_adapter_cfg.get("oxidizer", oxidizer) or {}),
                    },
                    sort_keys=True,
                )
                if metric_context_key not in prepared_metric_contexts:
                    prepared_metric_contexts[metric_context_key] = (
                        metric_mechanism_file,
                        _prepare_cantera_mechanism(metric_mechanism_file, metric_adapter_cfg),
                        str(metric_adapter_cfg.get("fuel_name", fuel)),
                        dict(metric_adapter_cfg.get("oxidizer", oxidizer) or {}),
                        str(metric_adapter_cfg.get("mechanism_format", "")),
                    )
                (
                    metric_mechanism_source,
                    metric_cantera_mechanism_file,
                    metric_fuel,
                    metric_oxidizer,
                    metric_mechanism_format,
                ) = prepared_metric_contexts[metric_context_key]
                merged[spec.metric_id] = _compute_cantera_flame_speed(
                    ct,
                    mechanism_file=metric_cantera_mechanism_file,
                    fuel=metric_fuel,
                    oxidizer=metric_oxidizer,
                    metric_cfg=metric_cfg,
                )
                metric_mechanism_provenance[spec.metric_id] = {
                    "backend": "native_cantera",
                    "mechanism_file": metric_mechanism_source,
                    "cantera_mechanism_file": metric_cantera_mechanism_file,
                    "mechanism_format": metric_mechanism_format,
                    "fuel_name": metric_fuel,
                }
                merged["metric_mechanism_provenance"] = metric_mechanism_provenance
                _checkpoint_cache()
                continue
            if not spec.required:
                messages.append(
                    f"Skipping optional chemistry metric '{spec.metric_id}' "
                    "because no adapter method was configured"
                )
                continue
            raise ValueError(
                f"Chemistry adapter has no method for metric '{spec.metric_id}'. "
                "Provide simulation_data directly or declare simulation_adapter.metrics."
            )
    except Exception as exc:
        if backend == "auto" and fixture_path:
            return _fixture_fallback(
                "Cantera chemistry execution failed; chemistry adapter fell back to "
                f"fixture results ({type(exc).__name__}: {exc})"
            )
        raise

    cache_messages: list[str] = []
    cache_artifacts: dict[str, str] = {}
    if offline_results_path and not _missing_metric_ids(merged, required_metric_ids):
        written_path = _write_chemistry_offline_cache(
            cache_path=offline_results_path,
            dataset=dataset,
            case_spec=case_spec,
            simulation_data=merged,
            mechanism_file=mechanism_file,
            cantera_mechanism_file=cantera_mechanism_file,
            adapter_cfg=adapter_cfg,
        )
        cache_artifacts["chemistry_offline_results"] = written_path
        cache_messages.append(f"Wrote chemistry offline cache to '{written_path}'")

    merged.setdefault(
        "mechanism_provenance",
        {
            "backend": "native_cantera",
            "mechanism_file": mechanism_file,
            "cantera_mechanism_file": cantera_mechanism_file,
            "fuel_name": fuel,
        },
    )
    return ResolvedSimulationInputs(
        simulation_data=merged,
        solver_artifacts={
            "mechanism_file": mechanism_file,
            "cantera_mechanism_file": cantera_mechanism_file,
            **cache_artifacts,
        },
        messages=[
            f"Computed chemistry metrics with Cantera using '{mechanism_file}'",
            *cache_messages,
            *messages,
        ],
    )


def _resolve_openfoam_case(
    *,
    regime_name: str,
    dataset: ValidationDatasetManifest,
    case_spec: ValidationCaseSpec,
    simulation_data: dict[str, Any],
    adapter_cfg: dict[str, Any],
    provenance_key: str,
) -> ResolvedSimulationInputs:
    merged = dict(simulation_data)
    metric_ids = [spec.metric_id for spec in dataset.metrics]
    backend = str(adapter_cfg.get("backend", "fixture")).strip().lower()

    if backend == "fixture":
        fixture_path = str(adapter_cfg.get("fixture_results_path", "")).strip()
        if not fixture_path:
            raise ValueError(f"{regime_name} fixture adapter requires fixture_results_path")
        payload = _read_json_mapping(fixture_path)
        _merge_metric_payload(merged, payload, metric_ids=metric_ids)
        merged.setdefault(
            provenance_key,
            {
                "backend": "fixture",
                "fixture_results_path": fixture_path,
            },
        )
        return ResolvedSimulationInputs(
            simulation_data=merged,
            solver_artifacts={f"{regime_name}_fixture_results": str(Path(fixture_path))},
            messages=[f"Loaded {regime_name} metrics from fixture '{fixture_path}'"],
        )

    if backend not in {"openfoam_case", "docker_openfoam"}:
        raise ValueError(f"Unsupported {regime_name} adapter backend '{backend}'")

    template_dir = str(adapter_cfg.get("template_dir", "")).strip()
    if not template_dir:
        raise ValueError(f"{regime_name} OpenFOAM adapter requires template_dir")
    run_dir = Path(
        str(adapter_cfg.get("run_dir", "")).strip()
        or Path("outputs") / "validation_runtime" / case_spec.case_id
    )
    params = dict(adapter_cfg.get("params", {}) or {})
    if not params:
        params.update(
            {
                k: v
                for k, v in dict(case_spec.operating_point or {}).items()
                if isinstance(v, (int, float))
            }
        )
    runner = OpenFoamRunner(
        template_dir=template_dir,
        solver_cmd=str(adapter_cfg.get("solver_cmd", "pisoFoam")),
        backend="docker",
        docker_image=str(adapter_cfg.get("docker_image", "")).strip() or None,
    )
    runner.setup_case(run_dir, params)

    should_run_solver = bool(adapter_cfg.get("force_run", False))
    if should_run_solver:
        success = runner.run(
            run_dir,
            timeout_s=int(adapter_cfg.get("timeout_s", 1800)),
        )
        if not success:
            raise RuntimeError(f"{regime_name} OpenFOAM case failed for template '{template_dir}'")

    payload = runner.parse_results(run_dir)
    _merge_metric_payload(merged, payload, metric_ids=metric_ids)
    merged.setdefault(
        provenance_key,
        {
            "backend": backend,
            "template_dir": str(Path(template_dir)),
            "run_dir": str(run_dir),
            "solver_cmd": str(adapter_cfg.get("solver_cmd", "pisoFoam")),
            "force_run": should_run_solver,
        },
    )
    return ResolvedSimulationInputs(
        simulation_data=merged,
        solver_artifacts={
            "template_dir": str(Path(template_dir)),
            "run_dir": str(run_dir),
        },
        messages=[
            (
                f"Resolved {regime_name} metrics from OpenFOAM case '{template_dir}'"
                + (" after solver execution" if should_run_solver else " using case artifacts")
            )
        ],
    )


def _resolve_fixture_case(
    *,
    regime_name: str,
    dataset: ValidationDatasetManifest,
    simulation_data: dict[str, Any],
    adapter_cfg: dict[str, Any],
    provenance_key: str,
) -> ResolvedSimulationInputs:
    merged = dict(simulation_data)
    fixture_path = str(adapter_cfg.get("fixture_results_path", "")).strip()
    if not fixture_path:
        raise ValueError(f"{regime_name} fixture adapter requires fixture_results_path")
    payload = _read_json_mapping(fixture_path)
    _merge_metric_payload(
        merged,
        payload,
        metric_ids=[spec.metric_id for spec in dataset.metrics],
    )
    merged.setdefault(
        provenance_key,
        {
            "backend": "fixture",
            "fixture_results_path": fixture_path,
        },
    )
    return ResolvedSimulationInputs(
        simulation_data=merged,
        solver_artifacts={f"{regime_name}_fixture_results": str(Path(fixture_path))},
        messages=[f"Loaded {regime_name} metrics from fixture '{fixture_path}'"],
    )


def _resolve_full_handoff_inputs(
    dataset: ValidationDatasetManifest,
    case_spec: ValidationCaseSpec,
    simulation_data: dict[str, Any],
    adapter_cfg: dict[str, Any],
    *,
    prior_simulation_data: dict[str, dict[str, Any]],
) -> ResolvedSimulationInputs:
    merged = dict(simulation_data)
    chemistry_data = dict(prior_simulation_data.get("chemistry", {}) or {})
    spray_data = dict(prior_simulation_data.get("spray", {}) or {})
    reacting_data = dict(prior_simulation_data.get("reacting_flow", {}) or {})
    closed_cylinder_data = dict(prior_simulation_data.get("closed_cylinder", {}) or {})

    base_state = dict(adapter_cfg.get("base_state", {}) or {})
    state_chain = build_handoff_state_chain(
        base_state=base_state,
        state_overrides=list(adapter_cfg.get("states", []) or []),
        chemistry_data=chemistry_data,
        spray_data=spray_data,
        reacting_data=reacting_data,
        closed_cylinder_data=closed_cylinder_data,
        case_spec=case_spec,
    )
    validation_errors: list[str] = []
    for bundle in state_chain:
        validation_errors.extend(bundle.validate())
    if validation_errors:
        raise ValueError(
            "Reduced-state handoff bundle validation failed: " + "; ".join(validation_errors)
        )

    tolerances = dict(adapter_cfg.get("conservation_tolerances", {}) or {})
    conservation = compute_handoff_conservation(
        state_chain,
        mass_tolerance=float(tolerances.get("mass", 1.0e-6)),
        energy_tolerance=float(tolerances.get("energy", 1.0e-3)),
    )
    merged.update(conservation)
    merged["handoff_bundle"] = state_chain[0].to_dict() if state_chain else {}
    merged.setdefault(
        "full_handoff_provenance",
        {
            "backend": "derived_contract",
            "bundle_id": state_chain[0].bundle_id if state_chain else "",
            "mechanism_id": state_chain[0].mechanism_id if state_chain else "",
            "fuel_name": state_chain[0].fuel_name if state_chain else "",
        },
    )
    return ResolvedSimulationInputs(
        simulation_data=merged,
        solver_artifacts={
            "handoff_bundle_id": state_chain[0].bundle_id if state_chain else "",
            "handoff_stage_count": str(len(state_chain)),
        },
        messages=["Built reduced-state handoff contract from upstream regime outputs"],
    )


def resolve_simulation_inputs(
    regime_name: str,
    dataset: ValidationDatasetManifest,
    case_spec: ValidationCaseSpec,
    simulation_data: dict[str, Any],
    *,
    prior_results: dict[str, ValidationRunManifest] | None = None,
    prior_simulation_data: dict[str, dict[str, Any]] | None = None,
) -> ResolvedSimulationInputs:
    """Resolve missing runtime simulation data for a validation regime."""
    _ = prior_results
    adapter_cfg = _adapter_config(case_spec, simulation_data)
    if not adapter_cfg:
        return ResolvedSimulationInputs(simulation_data=dict(simulation_data))

    if regime_name == "chemistry":
        return _resolve_chemistry_inputs(dataset, case_spec, simulation_data, adapter_cfg)
    if regime_name == "spray":
        return _resolve_openfoam_case(
            regime_name=regime_name,
            dataset=dataset,
            case_spec=case_spec,
            simulation_data=simulation_data,
            adapter_cfg=adapter_cfg,
            provenance_key="spray_provenance",
        )
    if regime_name == "reacting_flow":
        return _resolve_openfoam_case(
            regime_name=regime_name,
            dataset=dataset,
            case_spec=case_spec,
            simulation_data=simulation_data,
            adapter_cfg=adapter_cfg,
            provenance_key="reacting_flow_provenance",
        )
    if regime_name == "full_handoff":
        return _resolve_full_handoff_inputs(
            dataset,
            case_spec,
            simulation_data,
            adapter_cfg,
            prior_simulation_data=dict(prior_simulation_data or {}),
        )
    if regime_name == "closed_cylinder":
        return _resolve_fixture_case(
            regime_name=regime_name,
            dataset=dataset,
            simulation_data=simulation_data,
            adapter_cfg=adapter_cfg,
            provenance_key="closed_cylinder_provenance",
        )

    return ResolvedSimulationInputs(simulation_data=dict(simulation_data))
