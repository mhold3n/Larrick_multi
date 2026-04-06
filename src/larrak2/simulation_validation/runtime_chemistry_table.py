"""Build and stage offline Cantera chemistry lookup tables for engine runtime use."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .openfoam_chemistry_package import _format_number, _format_word

RUNTIME_CHEMISTRY_TABLE_SCHEMA_VERSION = 2
DEFAULT_STATE_SPECIES = ("IC8H18", "O2", "CO2", "H2O", "OH", "CO", "HO2", "H2", "CH2O")
DEFAULT_BALANCE_SPECIES = "N2"
DEFAULT_INTERPOLATION_METHOD = "local_rbf"
DEFAULT_FALLBACK_POLICY = "fullReducedKinetics"
DEFAULT_JACOBIAN_MODE = "full_species"
DEFAULT_JACOBIAN_STORAGE = "csr"
DEFAULT_TRANSFORMED_STATE_VARIABLES = ("Pressure", "OH", "CO", "HO2", "H2", "CH2O")


def _load_cantera():
    if importlib.util.find_spec("cantera") is None:
        raise RuntimeError(
            "Cantera runtime is required for offline chemistry tables. "
            "Install the optional combustion extra with `pip install .[combustion]`."
        )
    return importlib.import_module("cantera")


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at '{path}'")
    return payload


def _resolve_repo_relative_path(raw_path: str | Path, *, repo_root: Path) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo_root / path).resolve()


def _load_config(config_path: str | Path) -> dict[str, Any]:
    payload = _load_json(config_path)
    table_cfg = dict(payload.get("runtime_chemistry_table", payload) or {})
    if not table_cfg:
        raise ValueError("Runtime chemistry table config is empty")
    return table_cfg


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_package_manifest(table_cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    package_dir = Path(str(table_cfg.get("package_dir", "")).strip())
    if package_dir:
        manifest_path = package_dir / "package_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"OpenFOAM chemistry package manifest not found: {manifest_path}"
            )
        return package_dir, _load_json(manifest_path)
    raise ValueError("Runtime chemistry table config must define package_dir")


def _resolve_yaml_path(package_manifest: dict[str, Any], *, repo_root: Path) -> Path:
    yaml_path = Path(str(package_manifest.get("generated_yaml_path", "")).strip())
    if not yaml_path:
        raise ValueError("Package manifest must define generated_yaml_path")
    if not yaml_path.is_absolute():
        yaml_path = repo_root / yaml_path
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Cantera YAML referenced by package manifest not found: {yaml_path}"
        )
    return yaml_path


def _default_axis_margin_fraction(axis_name: str) -> float:
    if axis_name == "Temperature":
        return 0.1
    if axis_name == "Pressure":
        return 0.2
    return 0.15


def _default_axis_min_span(axis_name: str) -> float:
    if axis_name == "Temperature":
        return 100.0
    if axis_name == "Pressure":
        return 1.0e5
    return 1.0e-6


def _state_axes(
    table_cfg: dict[str, Any],
    *,
    state_species: list[str],
    seed_points: list[dict[str, float]] | None = None,
) -> tuple[list[str], list[list[float]]]:
    axes_cfg = dict(table_cfg.get("state_axes", {}) or {})
    axis_order = ["Temperature", "Pressure", *state_species]
    strategy = str(table_cfg.get("state_axis_strategy", "explicit")).strip() or "explicit"
    if strategy == "explicit":
        axes: list[list[float]] = []
        for axis_name in axis_order:
            raw_values = list(axes_cfg.get(axis_name, []) or [])
            if len(raw_values) < 2:
                raise ValueError(f"State axis '{axis_name}' must define at least two points")
            values = sorted({float(value) for value in raw_values})
            axes.append(values)
        return axis_order, axes
    if strategy != "seed_corridor":
        raise ValueError(f"Unsupported state_axis_strategy '{strategy}'")

    seed_points = list(seed_points or [])
    quantiles = sorted(
        {
            min(1.0, max(0.0, float(value)))
            for value in list(table_cfg.get("state_axis_quantiles", [0.0, 0.5, 1.0]) or [])
        }
    )
    if len(quantiles) < 2:
        quantiles = [0.0, 0.5, 1.0]
    margin_fraction_cfg = dict(table_cfg.get("state_axis_margin_fraction", {}) or {})
    margin_absolute_cfg = dict(table_cfg.get("state_axis_margin_absolute", {}) or {})
    min_span_cfg = dict(table_cfg.get("state_axis_min_span", {}) or {})

    axes: list[list[float]] = []
    for axis_name in axis_order:
        explicit_values = sorted(
            {float(value) for value in list(axes_cfg.get(axis_name, []) or [])}
        )
        values = [
            float(point[axis_name])
            for point in seed_points
            if axis_name in point and math.isfinite(float(point[axis_name]))
        ]
        values.extend(explicit_values)
        if not values:
            raise ValueError(
                f"Unable to infer state axis '{axis_name}' from seed corridor; "
                "provide explicit state_axes or seed artifacts with this variable"
            )
        arr = np.asarray(values, dtype=float)
        center = float(np.quantile(arr, 0.5))
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        span = hi - lo
        min_span = float(min_span_cfg.get(axis_name, _default_axis_min_span(axis_name)))
        if span < min_span:
            lo = center - 0.5 * min_span
            hi = center + 0.5 * min_span
            span = hi - lo
        margin = max(
            float(margin_absolute_cfg.get(axis_name, 0.0)),
            span
            * float(margin_fraction_cfg.get(axis_name, _default_axis_margin_fraction(axis_name))),
        )
        lo -= margin
        hi += margin
        if axis_name == "Temperature":
            lo = max(200.0, lo)
        elif axis_name == "Pressure":
            lo = max(1.0, lo)
        else:
            lo = max(0.0, lo)
            hi = min(1.0, hi)
        if hi <= lo:
            hi = lo + max(min_span, 1.0e-8)
        axis_values = sorted(
            {
                float(lo),
                min(max(center, lo), hi),
                float(hi),
                *explicit_values,
                *[float(np.quantile(arr, q)) for q in quantiles[1:-1]],
            }
        )
        if len(axis_values) < 2:
            axis_values = [float(lo), float(hi)]
        axes.append(axis_values)
    return axis_order, axes


def _mole_fractions_to_mass_fractions(
    species_payload: dict[str, float],
    *,
    species_names: list[str],
    molecular_weights: np.ndarray,
) -> dict[str, float]:
    mole = np.zeros(len(species_names), dtype=float)
    index_by_species = {name: index for index, name in enumerate(species_names)}
    for species_name, value in species_payload.items():
        index = index_by_species.get(species_name)
        if index is None:
            continue
        mole[index] = max(float(value), 0.0)
    total_mole = float(np.sum(mole))
    if total_mole <= 0.0:
        return {}
    mole /= total_mole
    mass = mole * molecular_weights
    total_mass = float(np.sum(mass))
    if total_mass <= 0.0:
        return {}
    mass /= total_mass
    return {
        species_name: float(mass[index])
        for index, species_name in enumerate(species_names)
        if mass[index] > 0.0
    }


def _resolve_transformed_state_variables(
    table_cfg: dict[str, Any],
    *,
    axis_order: list[str],
) -> list[str]:
    requested = [
        str(item).strip()
        for item in list(
            table_cfg.get("transformed_state_variables", DEFAULT_TRANSFORMED_STATE_VARIABLES) or []
        )
        if str(item).strip()
    ]
    return [name for name in requested if name in axis_order]


def _state_transform_floor_map(
    table_cfg: dict[str, Any],
    *,
    axis_order: list[str],
    transformed_state_variables: list[str],
) -> dict[str, float]:
    configured = dict(table_cfg.get("state_transform_floors", {}) or {})
    floors: dict[str, float] = {}
    transformed_set = set(transformed_state_variables)
    for axis_name in axis_order:
        if axis_name not in transformed_set:
            floors[axis_name] = 0.0
            continue
        if axis_name == "Pressure":
            floors[axis_name] = float(configured.get(axis_name, 1.0))
        else:
            floors[axis_name] = float(configured.get(axis_name, 1.0e-30))
    return floors


def _transform_state_matrix(
    states: np.ndarray,
    *,
    axis_order: list[str],
    transformed_state_variables: list[str],
    state_transform_floors: dict[str, float],
) -> np.ndarray:
    transformed = np.asarray(states, dtype=float).copy()
    transformed_set = set(transformed_state_variables)
    for index, axis_name in enumerate(axis_order):
        if axis_name not in transformed_set:
            continue
        floor = max(float(state_transform_floors.get(axis_name, 1.0e-30)), 1.0e-300)
        transformed[:, index] = np.log10(np.maximum(transformed[:, index], floor))
    return transformed


def _transform_state_vector(
    state: dict[str, float] | np.ndarray,
    *,
    axis_order: list[str],
    transformed_state_variables: list[str],
    state_transform_floors: dict[str, float],
) -> np.ndarray:
    if isinstance(state, np.ndarray):
        values = np.asarray(state, dtype=float).copy()
    else:
        values = np.asarray([float(state[name]) for name in axis_order], dtype=float)
    transformed = _transform_state_matrix(
        values[None, :],
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
    )
    return transformed[0]


def _parse_openfoam_internal_scalar_field(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    uniform_match = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if uniform_match:
        return np.asarray([float(uniform_match.group(1))], dtype=float)
    nonuniform_match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        flags=re.S,
    )
    if nonuniform_match is None:
        raise ValueError(f"Could not parse internalField from OpenFOAM scalar field '{path}'")
    values = [float(token) for token in nonuniform_match.group(2).split()]
    expected_count = int(nonuniform_match.group(1))
    if len(values) != expected_count:
        raise ValueError(
            f"OpenFOAM field '{path}' declared {expected_count} values but contained {len(values)}"
        )
    return np.asarray(values, dtype=float)


def _seed_field_time_dirs(table_cfg: dict[str, Any], *, repo_root: Path) -> list[Path]:
    return _selected_field_time_dirs(
        list(table_cfg.get("seed_field_case_dirs", []) or []),
        repo_root=repo_root,
        max_time_dirs=max(int(table_cfg.get("seed_field_max_time_dirs", 3)), 1),
    )


def _selected_field_time_dirs(
    raw_paths: list[Any],
    *,
    repo_root: Path,
    max_time_dirs: int,
) -> list[Path]:
    selected_dirs: list[Path] = []
    max_time_dirs = max(int(max_time_dirs), 1)
    for raw_path in list(raw_paths or []):
        if not str(raw_path).strip():
            continue
        candidate = _resolve_repo_relative_path(raw_path, repo_root=repo_root)
        if not candidate.exists():
            continue
        if (candidate / "T").exists() and (candidate / "p").exists():
            selected_dirs.append(candidate)
            continue
        numeric_dirs: list[tuple[float, Path]] = []
        for child in candidate.iterdir():
            if not child.is_dir():
                continue
            try:
                numeric_dirs.append((float(child.name), child))
            except ValueError:
                continue
        numeric_dirs.sort(key=lambda item: item[0])
        selected_dirs.extend([path for _, path in numeric_dirs[-max_time_dirs:]])
    return selected_dirs


def _latest_logsummary_rows(case_dir: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    summaries = sorted(
        case_dir.glob("logSummary.*.dat"),
        key=lambda path: float(path.name[len("logSummary.") : -len(".dat")]),
    )
    for summary in summaries:
        payload = [
            line.strip()
            for line in summary.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not payload:
            continue
        crank, pressure, temperature, velocity = payload[-1].split()
        rows.append(
            {
                "time_s": float(summary.name[len("logSummary.") : -len(".dat")]),
                "crank_angle_deg": float(crank),
                "mean_pressure_Pa": float(pressure),
                "mean_temperature_K": float(temperature),
                "mean_velocity_magnitude_m_s": float(velocity),
            }
        )
    return rows


def _resolve_numeric_time_dir(case_dir: Path, target_time_s: float) -> Path | None:
    numeric_dirs: list[tuple[float, Path]] = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            numeric_dirs.append((float(child.name), child))
        except ValueError:
            continue
    if not numeric_dirs:
        return None
    numeric_time, path = min(numeric_dirs, key=lambda item: abs(item[0] - float(target_time_s)))
    tolerance = max(1.0e-12, abs(float(target_time_s)) * 1.0e-8)
    if abs(numeric_time - float(target_time_s)) > tolerance:
        return None
    return path


def _sample_field_points_from_time_dir(
    time_dir: Path,
    *,
    axis_order: list[str],
    max_cells_per_time_dir: int,
) -> list[dict[str, float]]:
    field_names = ["T", "p", *axis_order[2:]]
    field_arrays: dict[str, np.ndarray] = {}
    for field_name in field_names:
        field_arrays[field_name] = _parse_openfoam_internal_scalar_field(time_dir / field_name)
    field_size = len(field_arrays["T"])
    if field_size == 0:
        return []
    if any(len(values) not in {1, field_size} for values in field_arrays.values()):
        return []
    if field_size == 1:
        sample_indices = [0]
    else:
        selected_indices: set[int] = set()
        for values in field_arrays.values():
            if len(values) <= 1:
                continue
            selected_indices.add(int(np.argmin(values)))
            selected_indices.add(int(np.argmax(values)))
        if len(selected_indices) < max_cells_per_time_dir:
            evenly_spaced = np.linspace(0, field_size - 1, num=max_cells_per_time_dir, dtype=int)
            selected_indices.update(int(index) for index in evenly_spaced.tolist())
        sample_indices = sorted(selected_indices)[:max_cells_per_time_dir]
    points: list[dict[str, float]] = []
    for cell_index in sample_indices:
        point = {
            "Temperature": float(
                field_arrays["T"][0 if len(field_arrays["T"]) == 1 else cell_index]
            ),
            "Pressure": float(field_arrays["p"][0 if len(field_arrays["p"]) == 1 else cell_index]),
        }
        for species_name in axis_order[2:]:
            values = field_arrays[species_name]
            point[species_name] = float(values[0 if len(values) == 1 else cell_index])
        points.append(point)
    return points


def _collect_authority_window_points(
    table_cfg: dict[str, Any],
    *,
    axis_order: list[str],
    repo_root: Path,
) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    points: list[dict[str, float]] = []
    windows_meta: list[dict[str, Any]] = []
    authority_windows = list(table_cfg.get("authority_windows", []) or [])
    for raw_window in authority_windows:
        if not isinstance(raw_window, dict):
            continue
        raw_case_dir = str(raw_window.get("case_dir", "")).strip()
        if not raw_case_dir:
            windows_meta.append({"case_dir": "", "status": "missing_case_dir"})
            continue
        case_dir = _resolve_repo_relative_path(raw_case_dir, repo_root=repo_root)
        if not case_dir.exists():
            windows_meta.append(
                {
                    "case_dir": str(case_dir),
                    "status": "missing_case_dir",
                }
            )
            continue
        angle_min = raw_window.get("angle_min_deg")
        angle_max = raw_window.get("angle_max_deg")
        max_time_dirs = max(int(raw_window.get("max_time_dirs", 16)), 1)
        max_cells_per_time_dir = max(int(raw_window.get("max_cells_per_time_dir", 64)), 1)
        rows = _latest_logsummary_rows(case_dir)
        selected_rows = []
        for row in rows:
            crank_angle = float(row["crank_angle_deg"])
            if angle_min is not None and crank_angle < float(angle_min):
                continue
            if angle_max is not None and crank_angle > float(angle_max):
                continue
            selected_rows.append(row)
        if len(selected_rows) > max_time_dirs:
            indices = np.linspace(0, len(selected_rows) - 1, num=max_time_dirs, dtype=int)
            selected_rows = [selected_rows[int(index)] for index in indices.tolist()]
        sampled_time_dirs = 0
        for row in selected_rows:
            time_dir = _resolve_numeric_time_dir(case_dir, float(row["time_s"]))
            if time_dir is None:
                continue
            try:
                sampled_points = _sample_field_points_from_time_dir(
                    time_dir,
                    axis_order=axis_order,
                    max_cells_per_time_dir=max_cells_per_time_dir,
                )
            except (FileNotFoundError, ValueError):
                continue
            points.extend(sampled_points)
            sampled_time_dirs += 1
        windows_meta.append(
            {
                "case_dir": str(case_dir),
                "angle_min_deg": angle_min,
                "angle_max_deg": angle_max,
                "selected_logsummary_count": len(selected_rows),
                "sampled_time_dir_count": sampled_time_dirs,
                "max_cells_per_time_dir": max_cells_per_time_dir,
                "status": "ok" if sampled_time_dirs > 0 else "no_matching_time_dirs",
            }
        )
    unique: dict[tuple[float, ...], dict[str, float]] = {}
    for point in points:
        unique[_point_key(point, axis_order)] = point
    return list(unique.values()), windows_meta


def _authority_scan(
    authority_points: list[dict[str, float]],
    *,
    axis_order: list[str],
    axes: list[list[float]],
) -> dict[str, Any]:
    miss_counts: dict[str, int] = {}
    max_out_of_bound: dict[str, float] = {}
    missed_points: list[dict[str, float]] = []
    for point in authority_points:
        point_missed = False
        for axis_name, axis_values in zip(axis_order, axes, strict=True):
            value = float(point[axis_name])
            low = float(axis_values[0])
            high = float(axis_values[-1])
            if low <= value <= high:
                continue
            point_missed = True
            miss_counts[axis_name] = int(miss_counts.get(axis_name, 0)) + 1
            if value < low:
                excess = low - value
            else:
                excess = value - high
            max_out_of_bound[axis_name] = max(
                float(max_out_of_bound.get(axis_name, 0.0)), float(excess)
            )
        if point_missed:
            missed_points.append(dict(point))
    return {
        "authority_pass": not missed_points,
        "authority_miss_counts_by_variable": {
            key: int(value) for key, value in sorted(miss_counts.items())
        },
        "authority_max_out_of_bound_by_variable": {
            key: float(value) for key, value in sorted(max_out_of_bound.items())
        },
        "authority_point_count": int(len(authority_points)),
        "authority_missed_point_count": int(len(missed_points)),
        "missed_points": missed_points,
    }


def _build_mass_fraction_state(
    *,
    state_species: list[str],
    balance_species: str,
    point_values: dict[str, float],
) -> dict[str, float]:
    assigned = 0.0
    state: dict[str, float] = {}
    for species_name in state_species:
        value = max(float(point_values[species_name]), 0.0)
        state[species_name] = value
        assigned += value
    if assigned > 1.0 + 1.0e-10:
        raise ValueError(
            f"Lookup state assigns total species mass fraction {assigned:.6f} > 1.0 "
            f"for manifold species {state_species}"
        )
    state[balance_species] = max(0.0, 1.0 - assigned)
    return state


def _center_indices(axes: list[list[float]]) -> list[int]:
    return [len(axis_values) // 2 for axis_values in axes]


def _sparse_index_points(lengths: list[int], centers: list[int], max_level: int) -> list[list[int]]:
    points: list[list[int]] = []

    def _recurse(dim: int, remaining: int, prefix: list[int]) -> None:
        if dim >= len(lengths):
            points.append(list(prefix))
            return
        center = centers[dim]
        candidates = sorted(range(lengths[dim]), key=lambda index: (abs(index - center), index))
        for index in candidates:
            cost = abs(index - center)
            if cost > remaining:
                continue
            prefix.append(index)
            _recurse(dim + 1, remaining - cost, prefix)
            prefix.pop()

    _recurse(0, int(max_level), [])
    return points


def _point_key(axis_values: dict[str, float], axis_order: list[str]) -> tuple[float, ...]:
    return tuple(float(axis_values[name]) for name in axis_order)


def _sparse_points_from_axes(
    axis_order: list[str],
    axes: list[list[float]],
    *,
    sparse_level: int,
) -> list[dict[str, float]]:
    centers = _center_indices(axes)
    indices = _sparse_index_points(
        [len(axis_values) for axis_values in axes], centers, int(sparse_level)
    )
    seen: set[tuple[float, ...]] = set()
    points: list[dict[str, float]] = []
    for index_set in indices:
        point = {
            axis_name: float(axis_values[index])
            for axis_name, axis_values, index in zip(axis_order, axes, index_set, strict=True)
        }
        key = _point_key(point, axis_order)
        if key in seen:
            continue
        seen.add(key)
        points.append(point)
    return points


def _refined_axes(axes: list[list[float]]) -> list[list[float]]:
    refined: list[list[float]] = []
    for axis_values in axes:
        enriched = set(axis_values)
        for left, right in zip(axis_values, axis_values[1:]):
            enriched.add(0.5 * (float(left) + float(right)))
        refined.append(sorted(float(value) for value in enriched))
    return refined


def _clip_point_to_axes(
    point: dict[str, float],
    *,
    axis_order: list[str],
    axes: list[list[float]],
) -> dict[str, float]:
    clipped: dict[str, float] = {}
    for axis_name, axis_values in zip(axis_order, axes, strict=True):
        clipped[axis_name] = float(
            min(max(float(point[axis_name]), axis_values[0]), axis_values[-1])
        )
    return clipped


def _center_axis_defaults(axis_order: list[str], axes: list[list[float]]) -> dict[str, float]:
    centers = _center_indices(axes)
    return {
        axis_name: float(axis_values[index])
        for axis_name, axis_values, index in zip(axis_order, axes, centers, strict=True)
    }


def _artifact_state_point(
    payload: dict[str, Any],
    *,
    axis_order: list[str],
    state_key: str,
) -> dict[str, float] | None:
    state = dict(payload.get(state_key, {}) or {})
    if not state:
        return None
    point: dict[str, float] = {}
    for axis_name in axis_order:
        if axis_name not in state:
            return None
        point[axis_name] = float(state[axis_name])
    return point


def _rounded_transformed_state_signature(
    point: dict[str, float],
    *,
    axis_order: list[str],
    transformed_state_variables: list[str],
    state_transform_floors: dict[str, float],
) -> tuple[float, ...]:
    transformed = _transform_state_vector(
        point,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
    )
    signature: list[float] = []
    for axis_name, value in zip(axis_order, transformed.tolist(), strict=True):
        if axis_name == "Temperature":
            signature.append(round(float(value), 1))
        elif axis_name == "Pressure":
            signature.append(round(float(value), 3))
        else:
            signature.append(round(float(value), 3))
    return tuple(signature)


def _load_current_window_qdot_targets(
    table_cfg: dict[str, Any],
    *,
    axis_order: list[str],
    transformed_state_variables: list[str],
    state_transform_floors: dict[str, float],
    repo_root: Path,
) -> list[dict[str, Any]]:
    target_limit = max(int(table_cfg.get("current_window_qdot_target_limit", 1)), 1)
    stage_filter = {
        str(item).strip()
        for item in list(table_cfg.get("current_window_qdot_stage_names", []) or [])
        if str(item).strip()
    }
    deduped: OrderedDict[tuple[str, tuple[float, ...]], dict[str, Any]] = OrderedDict()
    for raw_path in list(table_cfg.get("seed_qdot_miss_artifacts", []) or []):
        if not str(raw_path).strip():
            continue
        path = _resolve_repo_relative_path(raw_path, repo_root=repo_root)
        if not path.exists():
            continue
        payload = _load_json(path)
        if str(payload.get("reject_variable", "")).strip() != "Qdot":
            continue
        point = _artifact_state_point(payload, axis_order=axis_order, state_key="reject_state")
        if point is None:
            continue
        stage_name = str(payload.get("stage_name", "")).strip()
        if stage_filter and stage_name not in stage_filter:
            continue
        signature = _rounded_transformed_state_signature(
            point,
            axis_order=axis_order,
            transformed_state_variables=transformed_state_variables,
            state_transform_floors=state_transform_floors,
        )
        key = (stage_name, signature)
        if key in deduped:
            deduped.pop(key)
        deduped[key] = {
            "stage_name": stage_name,
            "point_state": point,
            "reject_excess": float(payload.get("reject_excess", 0.0) or 0.0),
            "source_path": str(path),
        }
    if not deduped:
        return []
    return list(deduped.values())[-target_limit:]


def _load_species_miss_targets(
    table_cfg: dict[str, Any],
    *,
    axis_order: list[str],
    transformed_state_variables: list[str],
    state_transform_floors: dict[str, float],
    repo_root: Path,
) -> list[dict[str, Any]]:
    target_limit = max(int(table_cfg.get("current_window_diag_target_limit", 4)), 1)
    tracked_species = {
        str(item).strip()
        for item in list(table_cfg.get("state_species", []) or [])
        if str(item).strip()
    }
    stage_filter = {
        str(item).strip()
        for item in list(table_cfg.get("current_window_diag_stage_names", []) or [])
        if str(item).strip()
    }
    deduped: OrderedDict[tuple[str, str, tuple[float, ...]], dict[str, Any]] = OrderedDict()
    for raw_path in list(table_cfg.get("seed_species_miss_artifacts", []) or []):
        if not str(raw_path).strip():
            continue
        path = _resolve_repo_relative_path(raw_path, repo_root=repo_root)
        if not path.exists():
            continue
        payload = _load_json(path)
        reject_variable = str(payload.get("reject_variable", "")).strip()
        failure_class = str(payload.get("failure_class", "")).strip()
        if (
            not reject_variable.endswith("_diag")
            and reject_variable not in tracked_species
            and not (
                failure_class == "same_sign_overshoot"
                and reject_variable
                and reject_variable != "Qdot"
            )
        ):
            continue
        point = _artifact_state_point(payload, axis_order=axis_order, state_key="reject_state")
        if point is None:
            continue
        stage_name = str(payload.get("stage_name", "")).strip()
        if stage_filter and stage_name not in stage_filter:
            continue
        signature = _rounded_transformed_state_signature(
            point,
            axis_order=axis_order,
            transformed_state_variables=transformed_state_variables,
            state_transform_floors=state_transform_floors,
        )
        key = (stage_name, reject_variable, signature)
        if key in deduped:
            deduped.pop(key)
        deduped[key] = {
            "stage_name": stage_name,
            "reject_variable": reject_variable,
            "point_state": point,
            "reject_excess": float(payload.get("reject_excess", 0.0) or 0.0),
            "source_path": str(path),
        }
    if not deduped:
        return []
    return list(deduped.values())[-target_limit:]


def _coverage_corpus_row_from_item(
    item: dict[str, Any],
    *,
    axis_order: list[str],
    source_path: Path,
) -> dict[str, Any] | None:
    point = _artifact_state_point(item, axis_order=axis_order, state_key="raw_state")
    if point is None:
        return None
    return {
        "point_state": point,
        "query_count": int(item.get("query_count", 0) or 0),
        "table_hit_count": int(item.get("table_hit_count", 0) or 0),
        "coverage_reject_count": int(item.get("coverage_reject_count", 0) or 0),
        "trust_reject_count": int(item.get("trust_reject_count", 0) or 0),
        "worst_reject_variable": str(item.get("worst_reject_variable", "")).strip(),
        "worst_reject_excess": float(item.get("worst_reject_excess", 0.0) or 0.0),
        "nearest_sample_distance_min": float(item.get("nearest_sample_distance_min", 0.0) or 0.0),
        "stage_names": [
            str(stage_name).strip()
            for stage_name in list(item.get("stage_names", []) or [])
            if str(stage_name).strip()
        ],
        "source_path": str(source_path),
        "high_fidelity_trust_reject": bool(item.get("high_fidelity_trust_reject", False)),
    }


def _load_coverage_corpus_rows(
    table_cfg: dict[str, Any],
    *,
    axis_order: list[str],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path_count = 0
    row_count = 0
    for raw_path in list(table_cfg.get("coverage_corpora", []) or []):
        if not str(raw_path).strip():
            continue
        path = _resolve_repo_relative_path(raw_path, repo_root=repo_root)
        if not path.exists():
            continue
        payload = _load_json(path)
        bundled: list[Any] = []
        bundled.extend(list(payload.get("rows", []) or []))
        bundled.extend(list(payload.get("high_fidelity_rows", []) or []))
        path_count += 1
        row_count += len(bundled)
        for item in bundled:
            if not isinstance(item, dict):
                continue
            row = _coverage_corpus_row_from_item(item, axis_order=axis_order, source_path=path)
            if row is None:
                continue
            rows.append(row)
    return rows, {
        "coverage_corpus_path_count": int(path_count),
        "coverage_corpus_row_count": int(row_count),
    }


def _load_field_support_rows(
    table_cfg: dict[str, Any],
    *,
    axis_order: list[str],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_case_dirs = list(table_cfg.get("current_window_support_case_dirs", []) or [])
    if not raw_case_dirs:
        return [], {
            "field_support_case_dir_count": 0,
            "field_support_candidate_count": 0,
        }

    max_time_dirs = max(int(table_cfg.get("seed_field_max_time_dirs", 3)), 1)
    max_cells_per_time_dir = max(int(table_cfg.get("seed_field_max_cells_per_time_dir", 24)), 1)
    rows_by_key: OrderedDict[tuple[float, ...], dict[str, Any]] = OrderedDict()
    case_dir_count = 0

    for raw_case_dir in raw_case_dirs:
        if not str(raw_case_dir).strip():
            continue
        case_dir = _resolve_repo_relative_path(raw_case_dir, repo_root=repo_root)
        if not case_dir.exists():
            continue
        case_dir_count += 1
        for time_dir in _selected_field_time_dirs(
            [raw_case_dir],
            repo_root=repo_root,
            max_time_dirs=max_time_dirs,
        ):
            try:
                points = _sample_field_points_from_time_dir(
                    time_dir,
                    axis_order=axis_order,
                    max_cells_per_time_dir=max_cells_per_time_dir,
                )
            except (FileNotFoundError, ValueError):
                continue
            for point in points:
                rows_by_key[_point_key(point, axis_order)] = {
                    "point_state": point,
                    "stage_names": [],
                    "source_path": str(case_dir),
                    "time_dir": str(time_dir),
                }

    return list(rows_by_key.values()), {
        "field_support_case_dir_count": int(case_dir_count),
        "field_support_candidate_count": int(len(rows_by_key)),
    }


def _support_points_for_targets(
    targets: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    *,
    axis_order: list[str],
    transformed_state_variables: list[str],
    state_transform_floors: dict[str, float],
    per_target_limit: int,
    min_transformed_distance: float = 0.0,
) -> list[dict[str, float]]:
    if not targets or not coverage_rows or per_target_limit <= 0:
        return []
    floor_distance = max(float(min_transformed_distance), 0.0)
    selected: OrderedDict[tuple[float, ...], dict[str, float]] = OrderedDict()
    transformed_rows = [
        (
            row,
            _transform_state_vector(
                row["point_state"],
                axis_order=axis_order,
                transformed_state_variables=transformed_state_variables,
                state_transform_floors=state_transform_floors,
            ),
        )
        for row in coverage_rows
    ]
    for target in targets:
        target_stage = str(target.get("stage_name", "")).strip()
        target_state = _transform_state_vector(
            dict(target["point_state"]),
            axis_order=axis_order,
            transformed_state_variables=transformed_state_variables,
            state_transform_floors=state_transform_floors,
        )
        ranked: list[tuple[float, int, int, int, float, dict[str, Any]]] = []
        for row, transformed_row in transformed_rows:
            row_stages = set(row.get("stage_names", []) or [])
            if target_stage and row_stages and target_stage not in row_stages:
                continue
            distance = float(np.linalg.norm(transformed_row - target_state))
            if floor_distance > 0.0 and distance < floor_distance:
                continue
            ranked.append(
                (
                    distance,
                    -int(row.get("high_fidelity_trust_reject", False)),
                    -int(row.get("trust_reject_count", 0)),
                    -int(row.get("query_count", 0)),
                    -float(row.get("worst_reject_excess", 0.0)),
                    row,
                )
            )
        ranked.sort(key=lambda item: item[:5])
        for _, _, _, _, _, row in ranked[:per_target_limit]:
            selected[_point_key(row["point_state"], axis_order)] = dict(row["point_state"])
    return list(selected.values())


def _extract_seed_points(
    table_cfg: dict[str, Any],
    *,
    axis_order: list[str],
    gas: Any,
    repo_root: Path,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    seeds: list[dict[str, float]] = []
    axes_cfg = dict(table_cfg.get("state_axes", {}) or {})
    seed_defaults = dict(table_cfg.get("seed_defaults", {}) or {})
    center_defaults: dict[str, float] = {}
    for axis_name in axis_order:
        raw_values = list(axes_cfg.get(axis_name, []) or [])
        if raw_values:
            values = sorted(float(value) for value in raw_values)
            center_defaults[axis_name] = values[len(values) // 2]
        elif axis_name == "Temperature":
            center_defaults[axis_name] = float(seed_defaults.get(axis_name, 1000.0))
        elif axis_name == "Pressure":
            center_defaults[axis_name] = float(seed_defaults.get(axis_name, 1.0e6))
        else:
            center_defaults[axis_name] = float(seed_defaults.get(axis_name, 0.0))

    explicit_points = list(table_cfg.get("seed_points", []) or [])
    for item in explicit_points:
        if not isinstance(item, dict):
            continue
        point = dict(center_defaults)
        for axis_name in axis_order:
            if axis_name in item:
                point[axis_name] = float(item[axis_name])
        seeds.append(point)

    species_names = list(getattr(gas, "species_names", []))
    molecular_weights = np.asarray(getattr(gas, "molecular_weights", np.zeros(0)), dtype=float)

    handoff_defaults = dict(center_defaults)

    for handoff_path in list(table_cfg.get("seed_handoff_artifacts", []) or []):
        if not str(handoff_path).strip():
            continue
        path = _resolve_repo_relative_path(handoff_path, repo_root=repo_root)
        if not path.exists():
            continue
        payload = _load_json(path)
        bundle = dict(payload.get("handoff_bundle", payload) or {})
        if not bundle:
            continue
        species = dict(bundle.get("species_mass_fractions", {}) or {})
        if not species:
            species = _mole_fractions_to_mass_fractions(
                dict(bundle.get("species_mole_fractions", {}) or {}),
                species_names=species_names,
                molecular_weights=molecular_weights,
            )
        point = dict(center_defaults)
        point["Temperature"] = float(bundle.get("temperature_K", center_defaults["Temperature"]))
        point["Pressure"] = float(bundle.get("pressure_Pa", center_defaults["Pressure"]))
        for species_name in axis_order[2:]:
            if species_name in species:
                point[species_name] = float(species[species_name])
                handoff_defaults[species_name] = float(species[species_name])
        handoff_defaults["Temperature"] = point["Temperature"]
        handoff_defaults["Pressure"] = point["Pressure"]
        seeds.append(point)

    seed_trace_defaults = dict(handoff_defaults)
    seed_trace_defaults.update(dict(table_cfg.get("seed_trace_species_defaults", {}) or {}))
    seed_trace_max_points = max(int(table_cfg.get("seed_trace_max_points", 12)), 1)
    for trace_path in list(table_cfg.get("seed_trace_artifacts", []) or []):
        if not str(trace_path).strip():
            continue
        path = _resolve_repo_relative_path(trace_path, repo_root=repo_root)
        if not path.exists():
            continue
        payload = _load_json(path)
        trace = list(payload.get("trace", []) or [])
        if not trace:
            latest = dict(payload.get("latest_checkpoint", {}) or {})
            trace = [latest] if latest else []
        if len(trace) > seed_trace_max_points:
            selected_indices = {
                int(round(index * (len(trace) - 1) / max(seed_trace_max_points - 1, 1)))
                for index in range(seed_trace_max_points)
            }
            trace = [trace[index] for index in sorted(selected_indices)]
        for row in trace:
            if not isinstance(row, dict):
                continue
            point = dict(seed_trace_defaults)
            point["Temperature"] = float(
                row.get("mean_temperature_K", center_defaults["Temperature"])
            )
            point["Pressure"] = float(row.get("mean_pressure_Pa", center_defaults["Pressure"]))
            seeds.append(point)

    max_cells_per_dir = max(int(table_cfg.get("seed_field_max_cells_per_time_dir", 24)), 1)
    for time_dir in _seed_field_time_dirs(table_cfg, repo_root=repo_root):
        field_names = ["T", "p", *axis_order[2:]]
        field_arrays: dict[str, np.ndarray] = {}
        try:
            for field_name in field_names:
                field_arrays[field_name] = _parse_openfoam_internal_scalar_field(
                    time_dir / field_name
                )
        except (FileNotFoundError, ValueError):
            continue
        field_size = len(field_arrays["T"])
        if field_size == 0:
            continue
        if any(len(values) not in {1, field_size} for values in field_arrays.values()):
            continue
        if field_size == 1:
            sample_indices = [0]
        else:
            selected_indices: set[int] = set()
            for values in field_arrays.values():
                if len(values) <= 1:
                    continue
                selected_indices.add(int(np.argmin(values)))
                selected_indices.add(int(np.argmax(values)))
            if len(selected_indices) < max_cells_per_dir:
                evenly_spaced = np.linspace(0, field_size - 1, num=max_cells_per_dir, dtype=int)
                selected_indices.update(int(index) for index in evenly_spaced.tolist())
            sample_indices = sorted(selected_indices)[:max_cells_per_dir]
        for cell_index in sample_indices:
            point = dict(center_defaults)
            point["Temperature"] = float(
                field_arrays["T"][0 if len(field_arrays["T"]) == 1 else cell_index]
            )
            point["Pressure"] = float(
                field_arrays["p"][0 if len(field_arrays["p"]) == 1 else cell_index]
            )
            for species_name in axis_order[2:]:
                values = field_arrays[species_name]
                point[species_name] = float(values[0 if len(values) == 1 else cell_index])
            seeds.append(point)

    transformed_state_variables = _resolve_transformed_state_variables(
        table_cfg,
        axis_order=axis_order,
    )
    state_transform_floors = _state_transform_floor_map(
        table_cfg,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
    )
    species_targets = _load_species_miss_targets(
        table_cfg,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
        repo_root=repo_root,
    )
    qdot_targets = _load_current_window_qdot_targets(
        table_cfg,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
        repo_root=repo_root,
    )
    seeds.extend(dict(target["point_state"]) for target in species_targets)
    seeds.extend(dict(target["point_state"]) for target in qdot_targets)

    coverage_rows, coverage_meta = _load_coverage_corpus_rows(
        table_cfg,
        axis_order=axis_order,
        repo_root=repo_root,
    )
    field_support_rows, field_support_meta = _load_field_support_rows(
        table_cfg,
        axis_order=axis_order,
        repo_root=repo_root,
    )
    corpus_support_min_td = max(
        float(table_cfg.get("corpus_support_min_transformed_distance", 0.0) or 0.0),
        0.0,
    )
    species_support = _support_points_for_targets(
        species_targets,
        coverage_rows,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
        per_target_limit=max(int(table_cfg.get("current_window_diag_support_per_target", 12)), 0),
        min_transformed_distance=corpus_support_min_td,
    )
    qdot_support = _support_points_for_targets(
        qdot_targets,
        coverage_rows,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
        per_target_limit=max(int(table_cfg.get("current_window_qdot_support_per_target", 12)), 0),
        min_transformed_distance=corpus_support_min_td,
    )
    field_support_per_target = max(
        int(table_cfg.get("current_window_field_support_per_target", 8)),
        0,
    )
    field_species_support = _support_points_for_targets(
        species_targets,
        field_support_rows,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
        per_target_limit=field_support_per_target,
    )
    field_qdot_support = _support_points_for_targets(
        qdot_targets,
        field_support_rows,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
        per_target_limit=field_support_per_target,
    )
    seeds.extend(species_support)
    seeds.extend(qdot_support)
    seeds.extend(field_species_support)
    seeds.extend(field_qdot_support)

    unique: dict[tuple[float, ...], dict[str, float]] = {}
    for point in seeds:
        unique[_point_key(point, axis_order)] = point
    return list(unique.values()), {
        "seed_species_target_count": int(len(species_targets)),
        "seed_species_target_variables": [
            str(target.get("reject_variable", "")) for target in species_targets
        ],
        "seed_qdot_target_count": int(len(qdot_targets)),
        "seed_qdot_target_stage_names": [
            str(target.get("stage_name", "")) for target in qdot_targets
        ],
        "coverage_species_support_seed_count": int(len(species_support)),
        "coverage_qdot_support_seed_count": int(len(qdot_support)),
        "corpus_support_min_transformed_distance": float(corpus_support_min_td),
        "field_species_support_seed_count": int(len(field_species_support)),
        "field_qdot_support_seed_count": int(len(field_qdot_support)),
        **coverage_meta,
        **field_support_meta,
        "current_window_diag_target_limit": max(
            int(table_cfg.get("current_window_diag_target_limit", 4)), 1
        ),
        "current_window_qdot_target_limit": max(
            int(table_cfg.get("current_window_qdot_target_limit", 1)), 1
        ),
    }


def _normalize_samples(sample_states: np.ndarray, state_scales: np.ndarray) -> np.ndarray:
    safe_scales = np.where(state_scales > 0.0, state_scales, 1.0)
    return sample_states / safe_scales


def _local_rbf_interpolate(
    *,
    sample_states: np.ndarray,
    sample_values: np.ndarray,
    query_state: np.ndarray,
    state_scales: np.ndarray,
    neighbor_count: int,
    epsilon: float,
) -> np.ndarray:
    if sample_states.shape[0] == 0:
        raise ValueError("local RBF interpolation requires at least one sample")
    if sample_states.shape[0] == 1:
        return np.asarray(sample_values[0], dtype=float)

    safe_scales = np.where(state_scales > 0.0, state_scales, 1.0)
    normalized_samples = sample_states / safe_scales
    normalized_query = query_state / safe_scales
    distances = np.linalg.norm(normalized_samples - normalized_query[None, :], axis=1)
    if float(np.min(distances)) <= 1.0e-12:
        exact_index = int(np.argmin(distances))
        return np.asarray(sample_values[exact_index], dtype=float)

    stencil = max(2, min(int(neighbor_count), sample_states.shape[0]))
    neighbour_indices = np.argsort(distances)[:stencil]
    neighbour_states = normalized_samples[neighbour_indices]
    neighbour_values = np.asarray(sample_values[neighbour_indices], dtype=float)
    pairwise = np.linalg.norm(
        neighbour_states[:, None, :] - neighbour_states[None, :, :],
        axis=2,
    )
    eps = max(float(epsilon), 1.0e-12)
    kernel = np.exp(-((eps * pairwise) ** 2))
    kernel += np.eye(stencil) * 1.0e-10
    rhs = np.exp(
        -((eps * np.linalg.norm(neighbour_states - normalized_query[None, :], axis=1)) ** 2)
    )
    weights = np.linalg.solve(kernel, rhs)
    value = weights @ neighbour_values
    return np.asarray(value, dtype=float)


def _finite_difference_qdot(
    gas: Any,
    *,
    temperature: float,
    pressure: float,
    mass_fractions: dict[str, float],
    delta_T: float,
) -> float:
    gas.TPY = temperature + delta_T, pressure, mass_fractions
    forward_rates = np.asarray(gas.net_production_rates, dtype=float)
    forward_enthalpies = np.asarray(gas.partial_molar_enthalpies, dtype=float)
    forward_qdot = -float(np.dot(forward_rates, forward_enthalpies))
    gas.TPY = temperature, pressure, mass_fractions
    base_rates = np.asarray(gas.net_production_rates, dtype=float)
    base_enthalpies = np.asarray(gas.partial_molar_enthalpies, dtype=float)
    base_qdot = -float(np.dot(base_rates, base_enthalpies))
    return (forward_qdot - base_qdot) / delta_T


def _mass_fraction_derivative_transform(
    *,
    mole_fractions: np.ndarray,
    molecular_weights: np.ndarray,
    mass_fractions: np.ndarray,
) -> np.ndarray:
    inv_mw = 1.0 / molecular_weights
    normalization = float(np.sum(mass_fractions * inv_mw))
    if normalization <= 0.0:
        raise ValueError("Invalid mass-fraction normalization while building runtime Jacobian")
    transform = -np.outer(mole_fractions, inv_mw / normalization)
    diagonal = np.diag_indices_from(transform)
    transform[diagonal] += inv_mw / normalization
    return transform


def _sparsify_dense_matrix(
    matrix: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_cols = matrix.shape
    row_ptr = np.zeros(n_rows + 1, dtype=np.int32)
    col_idx: list[int] = []
    values: list[float] = []
    for row in range(n_rows):
        row_ptr[row] = len(col_idx)
        row_values = matrix[row]
        nz_cols = np.nonzero(np.abs(row_values) >= threshold)[0]
        if nz_cols.size == 0:
            nz_cols = np.asarray([int(np.argmax(np.abs(row_values)))], dtype=np.int32)
        for col in nz_cols.tolist():
            col_idx.append(int(col))
            values.append(float(row_values[col]))
    row_ptr[n_rows] = len(col_idx)
    return row_ptr, np.asarray(col_idx, dtype=np.int32), np.asarray(values, dtype=float)


def _evaluate_sample(
    gas: Any,
    *,
    point_state: dict[str, float],
    state_species: list[str],
    balance_species: str,
    jacobian_threshold: float,
) -> dict[str, Any]:
    temperature = float(point_state["Temperature"])
    pressure = float(point_state["Pressure"])
    mass_fractions_dict = _build_mass_fraction_state(
        state_species=state_species,
        balance_species=balance_species,
        point_values=point_state,
    )
    gas.TPY = temperature, pressure, mass_fractions_dict

    species_names = list(gas.species_names)
    molecular_weights = np.asarray(gas.molecular_weights, dtype=float)
    mass_fractions = np.asarray(
        [float(mass_fractions_dict.get(name, 0.0)) for name in species_names], dtype=float
    )
    mole_fractions = np.asarray(gas.X, dtype=float)
    net_rates = np.asarray(gas.net_production_rates, dtype=float)
    partial_molar_enthalpies = np.asarray(gas.partial_molar_enthalpies, dtype=float)
    source_terms = net_rates * molecular_weights
    qdot = -float(np.dot(net_rates, partial_molar_enthalpies))

    if hasattr(gas, "net_production_rates_ddT") and hasattr(gas, "net_production_rates_ddX"):
        dwdot_dT = np.asarray(gas.net_production_rates_ddT, dtype=float)
        dwdot_dX = np.asarray(gas.net_production_rates_ddX, dtype=float)
        source_dT = dwdot_dT * molecular_weights
        transform = _mass_fraction_derivative_transform(
            mole_fractions=mole_fractions,
            molecular_weights=molecular_weights,
            mass_fractions=mass_fractions,
        )
        dwdot_dY = dwdot_dX @ transform
        source_dY = dwdot_dY * molecular_weights[:, None]
    else:  # pragma: no cover - exercised only if Cantera derivative API is unavailable
        delta_T = max(temperature * 1.0e-4, 1.0e-3)
        gas.TPY = temperature + delta_T, pressure, mass_fractions_dict
        rates_forward = np.asarray(gas.net_production_rates, dtype=float)
        source_dT = ((rates_forward - net_rates) / delta_T) * molecular_weights
        gas.TPY = temperature, pressure, mass_fractions_dict
        source_dY = np.zeros((len(species_names), len(species_names)), dtype=float)
        for species_index, species_name in enumerate(species_names):
            perturbed = dict(mass_fractions_dict)
            delta_y = max(mass_fractions_dict.get(species_name, 0.0) * 1.0e-3, 1.0e-8)
            if species_name == balance_species:
                delta_y = min(delta_y, max(perturbed[species_name] * 0.5, 1.0e-8))
                perturbed[species_name] = max(perturbed[species_name] + delta_y, 0.0)
            else:
                transfer = min(delta_y, max(perturbed.get(balance_species, 0.0) * 0.5, 1.0e-8))
                perturbed[species_name] = max(perturbed.get(species_name, 0.0) + transfer, 0.0)
                perturbed[balance_species] = max(
                    perturbed.get(balance_species, 0.0) - transfer, 0.0
                )
                delta_y = transfer
            gas.TPY = temperature, pressure, perturbed
            rates_perturbed = np.asarray(gas.net_production_rates, dtype=float)
            source_dY[:, species_index] = (
                (rates_perturbed - net_rates) / delta_y
            ) * molecular_weights
            gas.TPY = temperature, pressure, mass_fractions_dict

    full_jacobian = np.concatenate([source_dT[:, None], source_dY], axis=1)
    qdot_dT = _finite_difference_qdot(
        gas,
        temperature=temperature,
        pressure=pressure,
        mass_fractions=mass_fractions_dict,
        delta_T=max(temperature * 1.0e-4, 1.0e-3),
    )
    row_ptr, col_idx, csr_values = _sparsify_dense_matrix(
        full_jacobian, threshold=jacobian_threshold
    )
    diag_jacobian = np.diag(source_dY).astype(float)
    temperature_sensitivity = source_dT.astype(float)

    return {
        "point_state": dict(point_state),
        "mass_fractions": mass_fractions_dict,
        "source_terms": source_terms.astype(float),
        "qdot": float(qdot),
        "qdot_temperature_sensitivity": float(qdot_dT),
        "diag_jacobian": diag_jacobian,
        "temperature_sensitivity": temperature_sensitivity,
        "csr_row_ptr": row_ptr,
        "csr_col_idx": col_idx,
        "csr_values": csr_values,
        "jacobian_shape": [int(full_jacobian.shape[0]), int(full_jacobian.shape[1])],
    }


def _tracked_output_indices(
    *,
    species_names: list[str],
    tracked_species: list[str],
) -> list[int]:
    index_by_name = {name: idx for idx, name in enumerate(species_names)}
    return [index_by_name[name] for name in tracked_species if name in index_by_name]


def _build_feature_matrix(samples: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([np.asarray(sample[key], dtype=float) for sample in samples], dtype=float)


def _candidate_error(
    *,
    query_state: np.ndarray,
    sample_states: np.ndarray,
    samples: list[dict[str, Any]],
    state_scales: np.ndarray,
    neighbor_count: int,
    epsilon: float,
    tracked_indices: list[int],
    actual_sample: dict[str, Any],
) -> tuple[float, float]:
    source_matrix = _build_feature_matrix(samples, "source_terms")
    diag_matrix = _build_feature_matrix(samples, "diag_jacobian")
    temp_matrix = _build_feature_matrix(samples, "temperature_sensitivity")
    qdot_values = np.asarray([float(sample["qdot"]) for sample in samples], dtype=float)[:, None]
    qdot_dT_values = np.asarray(
        [float(sample["qdot_temperature_sensitivity"]) for sample in samples],
        dtype=float,
    )[:, None]

    predicted_source = _local_rbf_interpolate(
        sample_states=sample_states,
        sample_values=source_matrix[:, tracked_indices],
        query_state=query_state,
        state_scales=state_scales,
        neighbor_count=neighbor_count,
        epsilon=epsilon,
    )
    predicted_diag = _local_rbf_interpolate(
        sample_states=sample_states,
        sample_values=diag_matrix[:, tracked_indices],
        query_state=query_state,
        state_scales=state_scales,
        neighbor_count=neighbor_count,
        epsilon=epsilon,
    )
    predicted_temp = _local_rbf_interpolate(
        sample_states=sample_states,
        sample_values=temp_matrix[:, tracked_indices],
        query_state=query_state,
        state_scales=state_scales,
        neighbor_count=neighbor_count,
        epsilon=epsilon,
    )
    predicted_qdot = _local_rbf_interpolate(
        sample_states=sample_states,
        sample_values=qdot_values,
        query_state=query_state,
        state_scales=state_scales,
        neighbor_count=neighbor_count,
        epsilon=epsilon,
    )[0]
    predicted_qdot_dT = _local_rbf_interpolate(
        sample_states=sample_states,
        sample_values=qdot_dT_values,
        query_state=query_state,
        state_scales=state_scales,
        neighbor_count=neighbor_count,
        epsilon=epsilon,
    )[0]

    actual_source = np.asarray(actual_sample["source_terms"], dtype=float)[tracked_indices]
    actual_diag = np.asarray(actual_sample["diag_jacobian"], dtype=float)[tracked_indices]
    actual_temp = np.asarray(actual_sample["temperature_sensitivity"], dtype=float)[tracked_indices]
    actual_qdot = float(actual_sample["qdot"])
    actual_qdot_dT = float(actual_sample["qdot_temperature_sensitivity"])

    source_scale = np.maximum(np.abs(actual_source), 1.0e-12)
    diag_scale = np.maximum(np.abs(actual_diag), 1.0e-12)
    temp_scale = np.maximum(np.abs(actual_temp), 1.0e-12)
    qdot_scale = max(abs(actual_qdot), 1.0e-12)
    qdot_dT_scale = max(abs(actual_qdot_dT), 1.0e-12)

    source_error = max(
        float(np.max(np.abs(predicted_source - actual_source) / source_scale)),
        float(abs(float(predicted_qdot) - actual_qdot) / qdot_scale),
    )
    jacobian_error = max(
        float(np.max(np.abs(predicted_diag - actual_diag) / diag_scale)),
        float(np.max(np.abs(predicted_temp - actual_temp) / temp_scale)),
        float(abs(float(predicted_qdot_dT) - actual_qdot_dT) / qdot_dT_scale),
    )
    return source_error, jacobian_error


def _foam_scalar_list(values: list[float], *, indent: str = "    ", per_line: int = 6) -> str:
    lines = [f"{indent}("]
    for offset in range(0, len(values), per_line):
        chunk = values[offset : offset + per_line]
        lines.append(f"{indent}    " + " ".join(_format_number(float(value)) for value in chunk))
    lines.append(f"{indent})")
    return "\n".join(lines)


def _foam_word_list(values: list[str], *, indent: str = "    ", per_line: int = 8) -> str:
    lines = [f"{indent}("]
    for offset in range(0, len(values), per_line):
        chunk = values[offset : offset + per_line]
        lines.append(f"{indent}    " + " ".join(_format_word(value) for value in chunk))
    lines.append(f"{indent})")
    return "\n".join(lines)


def _foam_nested_scalar_lists(values: np.ndarray, *, indent: str = "    ") -> str:
    lines = [f"{indent}("]
    for row in np.asarray(values, dtype=float):
        lines.append(
            f"{indent}    ("
            + " ".join(_format_number(float(value)) for value in row.tolist())
            + ")"
        )
    lines.append(f"{indent})")
    return "\n".join(lines)


def _runtime_table_dictionary_text(
    *,
    table_id: str,
    package_manifest: dict[str, Any],
    axis_order: list[str],
    axes: list[list[float]],
    state_species: list[str],
    balance_species: str,
    interpolation_method: str,
    fallback_policy: str,
    max_untracked_mass_fraction: float,
    species_names: list[str],
    sample_states: np.ndarray,
    state_scales: np.ndarray,
    qdot_values: np.ndarray,
    qdot_temperature_sensitivity: np.ndarray,
    source_terms: np.ndarray,
    diag_jacobian: np.ndarray,
    temperature_sensitivity: np.ndarray,
    adaptive_cfg: dict[str, Any],
    trust_region_cfg: dict[str, Any],
    transformed_state_variables: list[str],
    state_transform_floors: dict[str, float],
    skip_stencil_envelope_non_state_species: bool = False,
    rbf_diag_envelope_scale_ho2: float | None = None,
) -> str:
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      runtimeChemistryTable;",
        "}",
        "",
        f"tableId                  {_format_word(table_id)};",
        "active                   yes;",
        f"packageId                {_format_word(str(package_manifest.get('package_id', '')))};",
        f"packageHash              {_format_word(str(package_manifest.get('package_hash', '')))};",
        f"interpolation            {_format_word(interpolation_method)};",
        f"fallbackPolicy           {_format_word(fallback_policy)};",
        "jacobianMode            full_species;",
        "jacobianStorage         csr;",
        f"maxUntrackedMassFraction {_format_number(max_untracked_mass_fraction)};",
        f"balanceSpecies           {_format_word(balance_species)};",
        f"sampleCount              {int(sample_states.shape[0])};",
        f"rbfNeighborCount         {int(adaptive_cfg.get('rbf_neighbor_count', 10))};",
        f"rbfEpsilon               {_format_number(float(adaptive_cfg.get('rbf_epsilon', 1.0)))};",
        f"rbfEnvelopeScale         {_format_number(float(adaptive_cfg.get('rbf_envelope_scale', 0.1)))};",
    ]
    if rbf_diag_envelope_scale_ho2 is not None:
        lines.append(
            f"rbfDiagEnvelopeScaleHO2  {_format_number(float(rbf_diag_envelope_scale_ho2))};",
        )
    lines.extend(
        [
            f"lookupCacheQuantization  {_format_number(float(adaptive_cfg.get('lookup_cache_quantization', 0.0025)))};",
            f"coverageCorpusQuantization {_format_number(float(adaptive_cfg.get('coverage_corpus_quantization', 0.0025)))};",
            "skipStencilEnvelopeNonStateSpecies "
            f"{'yes' if skip_stencil_envelope_non_state_species else 'no'};",
            f"trustRegionMaxAbsSource  {_format_number(float(trust_region_cfg.get('max_abs_source', 1.0e12)))};",
            f"trustRegionMaxAbsJacobian {_format_number(float(trust_region_cfg.get('max_abs_jacobian', 1.0e12)))};",
            f"trustRegionMaxAbsQdot    {_format_number(float(trust_region_cfg.get('max_abs_qdot', 1.0e15)))};",
            "stateVariables",
            _foam_word_list(axis_order),
            ";",
            "stateSpecies",
            _foam_word_list(state_species),
            ";",
            "transformedStateVariables",
            _foam_word_list(transformed_state_variables),
            ";",
            "stateScales",
            _foam_scalar_list([float(value) for value in state_scales.tolist()]),
            ";",
            "stateTransformFloors",
            "{",
        ],
    )
    for axis_name in axis_order:
        lines.append(
            f"    {_format_word(axis_name)} {_format_number(float(state_transform_floors.get(axis_name, 0.0)))};"
        )
    lines.extend(
        [
            "}",
            "axes",
            "{",
        ]
    )
    for axis_name, axis_values in zip(axis_order, axes, strict=True):
        lines.extend(
            [
                f"    {_format_word(axis_name)}",
                _foam_scalar_list(axis_values, indent="    "),
                "    ;",
            ]
        )
    lines.extend(
        [
            "}",
            "sampleStates",
            _foam_nested_scalar_lists(sample_states),
            ";",
            "speciesNames",
            _foam_word_list(species_names),
            ";",
            "qdot",
            _foam_scalar_list([float(value) for value in qdot_values.tolist()]),
            ";",
            "qdotTemperatureSensitivity",
            _foam_scalar_list([float(value) for value in qdot_temperature_sensitivity.tolist()]),
            ";",
            "sourceTerms",
            "{",
        ]
    )
    for species_name, values in zip(species_names, source_terms.T, strict=True):
        lines.extend(
            [
                f"    {_format_word(species_name)}",
                _foam_scalar_list([float(value) for value in values.tolist()], indent="    "),
                "    ;",
            ]
        )
    lines.extend(["}", "diagSourceJacobian", "{"])
    for species_name, values in zip(species_names, diag_jacobian.T, strict=True):
        lines.extend(
            [
                f"    {_format_word(species_name)}",
                _foam_scalar_list([float(value) for value in values.tolist()], indent="    "),
                "    ;",
            ]
        )
    lines.extend(["}", "temperatureSourceSensitivity", "{"])
    for species_name, values in zip(species_names, temperature_sensitivity.T, strict=True):
        lines.extend(
            [
                f"    {_format_word(species_name)}",
                _foam_scalar_list([float(value) for value in values.tolist()], indent="    "),
                "    ;",
            ]
        )
    lines.extend(["}", ""])
    return "\n".join(lines)


def _default_adaptive_sampling() -> dict[str, Any]:
    return {
        "sparse_level": 2,
        "candidate_sparse_level": 3,
        "refinement_rounds": 2,
        "batch_size": 12,
        "max_samples": 256,
        "source_tolerance": 0.15,
        "jacobian_tolerance": 0.20,
        "rbf_neighbor_count": 10,
        "rbf_epsilon": 1.0,
        "rbf_envelope_scale": 0.1,
        "lookup_cache_quantization": 0.0025,
        "coverage_corpus_quantization": 0.0025,
    }


def _default_trust_region() -> dict[str, Any]:
    return {
        "max_abs_source": 1.0e12,
        "max_abs_jacobian": 1.0e12,
        "max_abs_qdot": 1.0e15,
        "derive_from_samples": True,
        "source_multiplier": 2.0,
        "jacobian_multiplier": 2.0,
        "qdot_multiplier": 2.0,
    }


def _effective_trust_region(
    *,
    trust_region_cfg: dict[str, Any],
    source_terms: np.ndarray,
    diag_jacobian: np.ndarray,
    qdot_values: np.ndarray,
    qdot_temperature_sensitivity: np.ndarray,
) -> dict[str, Any]:
    effective = dict(trust_region_cfg)
    if not bool(trust_region_cfg.get("derive_from_samples", True)):
        return effective
    effective["max_abs_source"] = max(
        float(trust_region_cfg.get("max_abs_source", 0.0)),
        float(trust_region_cfg.get("source_multiplier", 2.0)) * float(np.max(np.abs(source_terms))),
    )
    effective["max_abs_jacobian"] = max(
        float(trust_region_cfg.get("max_abs_jacobian", 0.0)),
        float(trust_region_cfg.get("jacobian_multiplier", 2.0))
        * float(np.max(np.abs(diag_jacobian))),
    )
    effective["max_abs_qdot"] = max(
        float(trust_region_cfg.get("max_abs_qdot", 0.0)),
        float(trust_region_cfg.get("qdot_multiplier", 2.0))
        * float(
            max(
                np.max(np.abs(qdot_values)),
                np.max(np.abs(qdot_temperature_sensitivity)),
            )
        ),
    )
    return effective


def build_runtime_chemistry_table_from_spec(
    table_cfg: dict[str, Any],
    *,
    refresh: bool = False,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    package_dir, package_manifest = _resolve_package_manifest(table_cfg)
    repo_root_path = Path(repo_root).resolve()
    yaml_path = _resolve_yaml_path(package_manifest, repo_root=repo_root_path)

    table_id = str(table_cfg.get("table_id", "")).strip()
    if not table_id:
        raise ValueError("Runtime chemistry table config must define table_id")

    output_dir = Path(str(table_cfg.get("output_dir", "")).strip())
    if not output_dir:
        raise ValueError("Runtime chemistry table config must define output_dir")
    if not output_dir.is_absolute():
        output_dir = repo_root_path / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dict_path = output_dir / "runtimeChemistryTable"
    manifest_path = output_dir / "runtime_chemistry_table_manifest.json"
    data_path = output_dir / "runtime_chemistry_table_data.npz"
    jacobian_path = output_dir / "runtime_chemistry_jacobian_csr.npz"
    if (
        dict_path.exists()
        and manifest_path.exists()
        and data_path.exists()
        and jacobian_path.exists()
        and not refresh
    ):
        return _load_json(manifest_path)

    state_species = [
        str(item).strip()
        for item in list(table_cfg.get("state_species", DEFAULT_STATE_SPECIES) or [])
        if str(item).strip()
    ]
    if not state_species:
        raise ValueError("Runtime chemistry table must define at least one state species")
    balance_species = (
        str(table_cfg.get("balance_species", DEFAULT_BALANCE_SPECIES)).strip()
        or DEFAULT_BALANCE_SPECIES
    )
    if balance_species in state_species:
        raise ValueError("balance_species must not also appear in state_species")

    adaptive_cfg = _default_adaptive_sampling()
    adaptive_cfg.update(dict(table_cfg.get("adaptive_sampling", {}) or {}))
    trust_region_cfg = _default_trust_region()
    trust_region_cfg.update(dict(table_cfg.get("trust_region", {}) or {}))
    interpolation_method = (
        str(table_cfg.get("interpolation_method", DEFAULT_INTERPOLATION_METHOD)).strip()
        or DEFAULT_INTERPOLATION_METHOD
    )
    fallback_policy = (
        str(table_cfg.get("fallback_policy", DEFAULT_FALLBACK_POLICY)).strip()
        or DEFAULT_FALLBACK_POLICY
    )
    jacobian_mode = (
        str(table_cfg.get("jacobian_mode", DEFAULT_JACOBIAN_MODE)).strip() or DEFAULT_JACOBIAN_MODE
    )
    jacobian_storage = (
        str(table_cfg.get("jacobian_storage", DEFAULT_JACOBIAN_STORAGE)).strip()
        or DEFAULT_JACOBIAN_STORAGE
    )
    max_untracked_mass_fraction = float(table_cfg.get("max_untracked_mass_fraction", 0.02))
    jacobian_threshold = float(table_cfg.get("jacobian_sparsity_tolerance", 1.0e-16))

    ct = _load_cantera()
    gas = ct.Solution(str(yaml_path), transport_model=None)
    species_names = list(gas.species_names)
    if balance_species not in species_names:
        raise ValueError(f"Balance species '{balance_species}' is not present in the mechanism")
    for species_name in state_species:
        if species_name not in species_names:
            raise ValueError(f"State species '{species_name}' is not present in the mechanism")

    axis_order = ["Temperature", "Pressure", *state_species]
    raw_seed_points, seed_meta = _extract_seed_points(
        table_cfg,
        axis_order=axis_order,
        gas=gas,
        repo_root=repo_root_path,
    )
    authority_points, authority_windows_meta = _collect_authority_window_points(
        table_cfg,
        axis_order=axis_order,
        repo_root=repo_root_path,
    )
    axis_seed_points: dict[tuple[float, ...], dict[str, float]] = {}
    for point in [*raw_seed_points, *authority_points]:
        axis_seed_points[_point_key(point, axis_order)] = dict(point)
    authority_enrichment_rounds = max(int(table_cfg.get("authority_enrichment_rounds", 2)), 0)
    authority_scan_max_points = max(
        int(table_cfg.get("authority_scan_max_points_per_round", 256)), 1
    )
    authority_status: dict[str, Any] = {
        "authority_pass": len(authority_points) == 0,
        "authority_miss_counts_by_variable": {},
        "authority_max_out_of_bound_by_variable": {},
        "authority_point_count": int(len(authority_points)),
        "authority_missed_point_count": 0,
        "missed_points": [],
    }
    for _round in range(authority_enrichment_rounds + 1):
        axis_order, axes = _state_axes(
            table_cfg,
            state_species=state_species,
            seed_points=list(axis_seed_points.values()),
        )
        authority_status = _authority_scan(authority_points, axis_order=axis_order, axes=axes)
        if authority_status["authority_pass"]:
            break
        for point in authority_status["missed_points"][:authority_scan_max_points]:
            axis_seed_points[_point_key(point, axis_order)] = dict(point)

    coarse_points = _sparse_points_from_axes(
        axis_order,
        axes,
        sparse_level=int(adaptive_cfg["sparse_level"]),
    )
    coarse_points += [
        _clip_point_to_axes(point, axis_order=axis_order, axes=axes)
        for point in [*raw_seed_points, *authority_points]
    ]
    initial_points: dict[tuple[float, ...], dict[str, float]] = {}
    for point in coarse_points:
        initial_points[_point_key(point, axis_order)] = _clip_point_to_axes(
            point,
            axis_order=axis_order,
            axes=axes,
        )

    refined_axis_values = _refined_axes(axes)
    candidate_points = _sparse_points_from_axes(
        axis_order,
        refined_axis_values,
        sparse_level=int(adaptive_cfg["candidate_sparse_level"]),
    )
    for point in list(initial_points.values()):
        candidate_points.append(point)
    unique_candidates: dict[tuple[float, ...], dict[str, float]] = {}
    for point in candidate_points:
        unique_candidates[_point_key(point, axis_order)] = _clip_point_to_axes(
            point,
            axis_order=axis_order,
            axes=axes,
        )
    candidate_points = list(unique_candidates.values())

    tracked_indices = _tracked_output_indices(
        species_names=species_names,
        tracked_species=[*state_species, balance_species],
    )
    transformed_state_variables = _resolve_transformed_state_variables(
        table_cfg,
        axis_order=axis_order,
    )
    state_transform_floors = _state_transform_floor_map(
        table_cfg,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
    )

    sample_records: list[dict[str, Any]] = []
    sampled_keys: set[tuple[float, ...]] = set()
    for point in initial_points.values():
        sample_records.append(
            _evaluate_sample(
                gas,
                point_state=point,
                state_species=state_species,
                balance_species=balance_species,
                jacobian_threshold=jacobian_threshold,
            )
        )
        sampled_keys.add(_point_key(point, axis_order))

    for _round in range(int(adaptive_cfg["refinement_rounds"])):
        if len(sample_records) >= int(adaptive_cfg["max_samples"]):
            break
        sample_states_raw = np.asarray(
            [
                [float(sample["point_state"][axis_name]) for axis_name in axis_order]
                for sample in sample_records
            ],
            dtype=float,
        )
        if sample_states_raw.shape[0] < 2:
            break
        sample_states = _transform_state_matrix(
            sample_states_raw,
            axis_order=axis_order,
            transformed_state_variables=transformed_state_variables,
            state_transform_floors=state_transform_floors,
        )
        state_scales = np.asarray(
            [
                max(float(np.max(sample_states[:, dimi]) - np.min(sample_states[:, dimi])), 1.0)
                for dimi in range(sample_states.shape[1])
            ],
            dtype=float,
        )
        worst_candidates: list[tuple[float, float, dict[str, Any]]] = []
        for point in candidate_points:
            key = _point_key(point, axis_order)
            if key in sampled_keys:
                continue
            actual_sample = _evaluate_sample(
                gas,
                point_state=point,
                state_species=state_species,
                balance_species=balance_species,
                jacobian_threshold=jacobian_threshold,
            )
            source_error, jacobian_error = _candidate_error(
                query_state=_transform_state_vector(
                    point,
                    axis_order=axis_order,
                    transformed_state_variables=transformed_state_variables,
                    state_transform_floors=state_transform_floors,
                ),
                sample_states=sample_states,
                samples=sample_records,
                state_scales=state_scales,
                neighbor_count=int(adaptive_cfg["rbf_neighbor_count"]),
                epsilon=float(adaptive_cfg["rbf_epsilon"]),
                tracked_indices=tracked_indices,
                actual_sample=actual_sample,
            )
            if source_error > float(adaptive_cfg["source_tolerance"]) or jacobian_error > float(
                adaptive_cfg["jacobian_tolerance"]
            ):
                worst_candidates.append((source_error, jacobian_error, actual_sample))
        if not worst_candidates:
            break
        worst_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, actual_sample in worst_candidates[: int(adaptive_cfg["batch_size"])]:
            key = _point_key(dict(actual_sample["point_state"]), axis_order)
            if key in sampled_keys:
                continue
            sample_records.append(actual_sample)
            sampled_keys.add(key)
            if len(sample_records) >= int(adaptive_cfg["max_samples"]):
                break

    sample_records.sort(
        key=lambda sample: tuple(
            float(sample["point_state"][axis_name]) for axis_name in axis_order
        )
    )

    sample_states_raw = np.asarray(
        [
            [float(sample["point_state"][axis_name]) for axis_name in axis_order]
            for sample in sample_records
        ],
        dtype=float,
    )
    sample_states = _transform_state_matrix(
        sample_states_raw,
        axis_order=axis_order,
        transformed_state_variables=transformed_state_variables,
        state_transform_floors=state_transform_floors,
    )
    state_scales = np.asarray(
        [
            max(float(np.max(sample_states[:, dimi]) - np.min(sample_states[:, dimi])), 1.0)
            for dimi in range(sample_states.shape[1])
        ],
        dtype=float,
    )
    qdot_values = np.asarray([float(sample["qdot"]) for sample in sample_records], dtype=float)
    qdot_temperature_sensitivity = np.asarray(
        [float(sample["qdot_temperature_sensitivity"]) for sample in sample_records],
        dtype=float,
    )
    source_terms = np.asarray(
        [np.asarray(sample["source_terms"], dtype=float) for sample in sample_records], dtype=float
    )
    diag_jacobian = np.asarray(
        [np.asarray(sample["diag_jacobian"], dtype=float) for sample in sample_records], dtype=float
    )
    temperature_sensitivity = np.asarray(
        [np.asarray(sample["temperature_sensitivity"], dtype=float) for sample in sample_records],
        dtype=float,
    )

    csr_row_ptr = np.asarray(
        [np.asarray(sample["csr_row_ptr"], dtype=np.int32) for sample in sample_records],
        dtype=np.int32,
    )
    csr_offsets = np.zeros(len(sample_records) + 1, dtype=np.int64)
    csr_col_idx_chunks: list[np.ndarray] = []
    csr_value_chunks: list[np.ndarray] = []
    for sample_index, sample in enumerate(sample_records):
        values = np.asarray(sample["csr_values"], dtype=float)
        cols = np.asarray(sample["csr_col_idx"], dtype=np.int32)
        csr_offsets[sample_index + 1] = csr_offsets[sample_index] + values.size
        csr_col_idx_chunks.append(cols)
        csr_value_chunks.append(values)
    csr_col_idx = (
        np.concatenate(csr_col_idx_chunks) if csr_col_idx_chunks else np.zeros(0, dtype=np.int32)
    )
    csr_values = np.concatenate(csr_value_chunks) if csr_value_chunks else np.zeros(0, dtype=float)
    effective_trust_region_cfg = _effective_trust_region(
        trust_region_cfg=trust_region_cfg,
        source_terms=source_terms,
        diag_jacobian=diag_jacobian,
        qdot_values=qdot_values,
        qdot_temperature_sensitivity=qdot_temperature_sensitivity,
    )

    rbf_diag_ho2_raw = table_cfg.get("rbf_diag_envelope_scale_ho2")
    rbf_diag_envelope_scale_ho2: float | None
    if rbf_diag_ho2_raw is None or str(rbf_diag_ho2_raw).strip() == "":
        rbf_diag_envelope_scale_ho2 = None
    else:
        rbf_diag_envelope_scale_ho2 = float(rbf_diag_ho2_raw)

    dict_path.write_text(
        _runtime_table_dictionary_text(
            table_id=table_id,
            package_manifest=package_manifest,
            axis_order=axis_order,
            axes=axes,
            state_species=state_species,
            balance_species=balance_species,
            interpolation_method=interpolation_method,
            fallback_policy=fallback_policy,
            max_untracked_mass_fraction=max_untracked_mass_fraction,
            species_names=species_names,
            sample_states=sample_states,
            state_scales=state_scales,
            qdot_values=qdot_values,
            qdot_temperature_sensitivity=qdot_temperature_sensitivity,
            source_terms=source_terms,
            diag_jacobian=diag_jacobian,
            temperature_sensitivity=temperature_sensitivity,
            adaptive_cfg=adaptive_cfg,
            trust_region_cfg=effective_trust_region_cfg,
            transformed_state_variables=transformed_state_variables,
            state_transform_floors=state_transform_floors,
            skip_stencil_envelope_non_state_species=bool(
                table_cfg.get("skip_stencil_envelope_non_state_species", False)
            ),
            rbf_diag_envelope_scale_ho2=rbf_diag_envelope_scale_ho2,
        ),
        encoding="utf-8",
    )

    np.savez_compressed(
        data_path,
        sample_states=sample_states,
        sample_states_raw=sample_states_raw,
        state_scales=state_scales,
        qdot=qdot_values,
        qdot_temperature_sensitivity=qdot_temperature_sensitivity,
        source_terms=source_terms,
        diag_source_jacobian=diag_jacobian,
        temperature_source_sensitivity=temperature_sensitivity,
    )
    np.savez_compressed(
        jacobian_path,
        sample_csr_offsets=csr_offsets,
        csr_row_ptr=csr_row_ptr,
        csr_col_idx=csr_col_idx,
        csr_values=csr_values,
        jacobian_shape=np.asarray(sample_records[0]["jacobian_shape"], dtype=np.int32),
    )

    manifest = {
        "runtime_chemistry_table_schema_version": RUNTIME_CHEMISTRY_TABLE_SCHEMA_VERSION,
        "table_id": table_id,
        "package_id": str(package_manifest.get("package_id", "")),
        "package_hash": str(package_manifest.get("package_hash", "")),
        "package_dir": str(package_dir.resolve()),
        "generated_yaml_path": str(yaml_path.resolve()),
        "table_point_count": int(sample_states.shape[0]),
        "species_count": len(species_names),
        "state_variables": axis_order,
        "state_axis_strategy": str(table_cfg.get("state_axis_strategy", "explicit")).strip()
        or "explicit",
        "state_species": state_species,
        "balance_species": balance_species,
        "state_axes": {
            axis_name: axis_values for axis_name, axis_values in zip(axis_order, axes, strict=True)
        },
        "transformed_state_variables": transformed_state_variables,
        "state_transform_floors": {
            axis_name: float(state_transform_floors.get(axis_name, 0.0)) for axis_name in axis_order
        },
        "interpolation_method": interpolation_method,
        "fallback_policy": fallback_policy,
        "max_untracked_mass_fraction": max_untracked_mass_fraction,
        "skip_stencil_envelope_non_state_species": bool(
            table_cfg.get("skip_stencil_envelope_non_state_species", False)
        ),
        **(
            {"rbf_diag_envelope_scale_ho2": float(rbf_diag_envelope_scale_ho2)}
            if rbf_diag_envelope_scale_ho2 is not None
            else {}
        ),
        "jacobian_mode": jacobian_mode,
        "jacobian_storage": jacobian_storage,
        "state_scales": [float(value) for value in state_scales.tolist()],
        "adaptive_sampling": {
            **adaptive_cfg,
            "seed_point_count": int(len(initial_points)),
            "candidate_point_count": int(len(candidate_points)),
        },
        **seed_meta,
        "authority_windows": authority_windows_meta,
        "authority_pass": bool(authority_status.get("authority_pass", False)),
        "authority_miss_counts_by_variable": dict(
            authority_status.get("authority_miss_counts_by_variable", {})
        ),
        "authority_max_out_of_bound_by_variable": dict(
            authority_status.get("authority_max_out_of_bound_by_variable", {})
        ),
        "strict_runtime_certified": bool(authority_points)
        and bool(authority_status.get("authority_pass", False)),
        "trust_region": effective_trust_region_cfg,
        "trust_region_config": trust_region_cfg,
        "files": {
            "runtimeChemistryTable": str(dict_path.resolve()),
            "runtime_chemistry_table_data": str(data_path.resolve()),
            "runtime_chemistry_jacobian_csr": str(jacobian_path.resolve()),
        },
        "generated_file_hashes": {
            "runtimeChemistryTable": _sha256_file(dict_path),
            "runtime_chemistry_table_data": _sha256_file(data_path),
            "runtime_chemistry_jacobian_csr": _sha256_file(jacobian_path),
        },
        "jacobian": {
            "mode": jacobian_mode,
            "storage": jacobian_storage,
            "row_count": len(species_names),
            "column_count": len(species_names) + 1,
            "column_variables": ["Temperature", *species_names],
            "species_basis": "mass_fraction",
            "pressure_is_interpolation_axis": True,
            "sparsity_tolerance": jacobian_threshold,
            "total_nnz": int(csr_values.size),
            "per_sample_average_nnz": float(csr_values.size / max(sample_states.shape[0], 1)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_runtime_chemistry_table(
    *,
    config_path: str | Path,
    refresh: bool = False,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    return build_runtime_chemistry_table_from_spec(
        _load_config(config_path),
        refresh=refresh,
        repo_root=repo_root,
    )


__all__ = [
    "build_runtime_chemistry_table",
    "build_runtime_chemistry_table_from_spec",
]
