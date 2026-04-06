"""Tuning-knob characterization: ingest benchmark runs, apply declarative knobs, model-based proposals.

Pairs with restart regression artifacts (extract_restart_run_artifacts) and optional BO (sklearn GP + EI).
"""

from __future__ import annotations

import copy
import glob
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .restart_regression_artifacts import extract_restart_run_artifacts
from .restart_regression_suite import analyze_restart_regression_runs

TUNING_MANIFEST_BASENAME = "tuning_experiment_manifest.json"
EXPERIMENT_STORE_SCHEMA_VERSION = 1


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_nested_value(container: dict[str, Any], dotted: str) -> Any:
    parts = dotted.split(".")
    node: Any = container
    for part in parts:
        if not isinstance(node, dict):
            raise KeyError(f"Cannot traverse '{dotted}': not a dict at '{part}'")
        if part not in node:
            raise KeyError(f"Missing path segment '{part}' in {dotted}")
        node = node[part]
    return node


def _set_nested(container: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = container
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def load_knob_schema(path: str | Path) -> dict[str, Any]:
    raw = _load_json(path)
    if "schema_id" not in raw or "knobs" not in raw:
        raise ValueError("Knob schema must define schema_id and knobs")
    return raw


def apply_knobs_to_table_config(
    *,
    table_config_path: str | Path,
    knob_schema: dict[str, Any],
    knobs: dict[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load full table JSON (with optional _comment), apply knob values under config_root_key."""
    root_key = str(knob_schema.get("config_root_key", "runtime_chemistry_table")).strip()
    payload = _load_json(table_config_path)
    if root_key not in payload:
        raise KeyError(f"Config root '{root_key}' not in {table_config_path}")
    inner = copy.deepcopy(payload[root_key])
    knob_defs = {str(k["name"]): k for k in list(knob_schema.get("knobs", []) or []) if "name" in k}
    for name, value in knobs.items():
        name = str(name).strip()
        if name not in knob_defs:
            raise KeyError(f"Unknown knob '{name}' for schema {knob_schema.get('schema_id')}")
        spec = knob_defs[name]
        path = str(spec["path"])
        v = _coerce_knob_value(value, spec)
        _set_nested(inner, path, v)
    out = copy.deepcopy(payload)
    out[root_key] = inner
    return out


def extract_knobs_from_table_config(
    *,
    table_config_path: str | Path,
    knob_schema_path: str | Path,
) -> dict[str, Any]:
    """Read knob values from a full table wrapper JSON using the same paths as apply_knobs_to_table_config."""
    knob_schema = load_knob_schema(knob_schema_path)
    root_key = str(knob_schema.get("config_root_key", "runtime_chemistry_table")).strip()
    payload = _load_json(table_config_path)
    if root_key not in payload:
        raise KeyError(f"Config root '{root_key}' not in {table_config_path}")
    inner = payload[root_key]
    if not isinstance(inner, dict):
        raise TypeError(f"Config root '{root_key}' must be an object in {table_config_path}")
    out: dict[str, Any] = {}
    for spec in list(knob_schema.get("knobs", []) or []):
        name = str(spec.get("name", "")).strip()
        if not name:
            continue
        path = str(spec["path"])
        raw = _get_nested_value(inner, path)
        out[name] = _coerce_knob_value(raw, spec)
    return out


def _coerce_knob_value(value: Any, spec: dict[str, Any]) -> Any:
    t = str(spec.get("type", "float")).lower()
    if t == "int":
        return int(round(float(value)))
    if t == "bool":
        return bool(value)
    return float(value)


def write_staged_table_config(
    *,
    payload: dict[str, Any],
    output_path: str | Path,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_input_artifact_hashes(
    *,
    table_config_path: str | Path,
    strategy_config_path: str | Path | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "table_config_path": str(Path(table_config_path).resolve()),
        "table_config_sha256": sha256_file(table_config_path),
    }
    if strategy_config_path is not None and str(strategy_config_path).strip():
        p = Path(strategy_config_path)
        if p.exists():
            out["strategy_config_path"] = str(p.resolve())
            out["strategy_config_sha256"] = sha256_file(p)
    return out


def build_tuning_manifest(
    *,
    knob_schema_id: str,
    knobs: dict[str, Any],
    input_artifact_hashes: dict[str, Any],
    experiment_id: str,
    staged_table_config_path: str = "",
    parent_experiment_id: str = "",
    git_commit: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "knob_schema_id": knob_schema_id,
        "experiment_id": experiment_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "knobs": dict(knobs),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "staged_table_config_path": staged_table_config_path,
        "parent_experiment_id": parent_experiment_id,
        "git_commit": git_commit,
    }


def write_tuning_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def load_tuning_manifest(run_dir: str | Path) -> dict[str, Any] | None:
    p = Path(run_dir) / TUNING_MANIFEST_BASENAME
    if not p.exists():
        return None
    return _load_json(p)


def compute_objective_vector(
    artifacts: dict[str, Any],
    *,
    lambda_numeric: float = 1e-4,
    mu_gate: float = 10.0,
    mu_reject_excess: float = 1.0,
    baseline_run_dir: str | Path | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Scalar and vector objectives from extract_restart_run_artifacts output."""
    profile = dict(artifacts.get("selected_profile") or {})
    trust_rejects = float(profile.get("trust_region_reject_cells", 0.0) or 0.0)
    numeric_hits = float(profile.get("total_numeric_hits", 0.0) or 0.0)
    gate = bool(profile.get("chem323_runtime_replacement_gate_passed", False))
    miss = dict(artifacts.get("authority_miss_payload") or {})
    reject_excess = float(miss.get("reject_excess", 0.0) or 0.0)
    reject_var = str(miss.get("reject_variable", "") or "")
    rv_hash = hashlib.sha256(reject_var.encode()).hexdigest()[:16]

    regression_delta = 0.0
    if baseline_run_dir is not None:
        try:
            analysis = analyze_restart_regression_runs(
                runs=[str(baseline_run_dir), str(artifacts["run_dir"])],
                profile_name=profile_name,
                history_window=2,
            )
            general = dict(analysis.get("general") or {})
            clusters = list(general.get("focus_clusters") or [])
            if clusters:
                regression_delta = float(
                    clusters[0].get("severity_score", 0.0) or clusters[0].get("score", 0.0) or 0.0
                )
        except (FileNotFoundError, ValueError, KeyError):
            regression_delta = 0.0

    gate_penalty = 0.0 if gate else 1.0
    penalized_scalar = (
        trust_rejects
        + lambda_numeric * numeric_hits
        + mu_gate * gate_penalty
        + mu_reject_excess * reject_excess
        + regression_delta
    )

    return {
        "trust_region_reject_cells": trust_rejects,
        "total_numeric_hits": numeric_hits,
        "chem323_runtime_replacement_gate_passed": 1.0 if gate else 0.0,
        "reject_excess": reject_excess,
        "reject_variable_hash": rv_hash,
        "reject_variable": reject_var,
        "regression_focus_delta": regression_delta,
        "penalized_scalar": penalized_scalar,
    }


def ingest_benchmark_run_directory(
    run_dir: str | Path,
    *,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """One experiment record: outcomes + optional trusted knobs from manifest."""
    root = Path(run_dir).resolve()
    extracted = extract_restart_run_artifacts(run_dir=root, profile_name=profile_name)
    objectives = compute_objective_vector(extracted, profile_name=profile_name)
    manifest = load_tuning_manifest(root)
    knobs_trusted = manifest is not None
    knobs = dict(manifest["knobs"]) if manifest else {}
    knob_schema_id = str(manifest.get("knob_schema_id", "")) if manifest else ""
    profile = dict(extracted.get("selected_profile") or {})
    return {
        "schema_version": EXPERIMENT_STORE_SCHEMA_VERSION,
        "run_dir": str(root),
        "profile_name": extracted.get("profile_name", ""),
        "knobs": knobs,
        "knobs_trusted": knobs_trusted,
        "knob_schema_id": knob_schema_id,
        "objectives": objectives,
        "runtime_table_hash": str(profile.get("runtime_table_hash", "") or ""),
        "ingested_utc": datetime.now(UTC).isoformat(),
    }


def maybe_log_tuning_observation(
    *,
    benchmark_outdir: str | Path,
    experiments_jsonl: str | Path,
    knob_schema_path: str | Path,
    table_config_path: str | Path,
    strategy_config_path: str | Path | None = None,
    profile_name: str | None = None,
    profile_names_in_benchmark: list[str] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write tuning_experiment_manifest.json, append one trusted row to experiments.jsonl. Never raises."""
    log = logging.getLogger(__name__)
    root = Path(repo_root).resolve() if repo_root else Path.cwd()

    def _resolve(p: str | Path) -> Path:
        pp = Path(p)
        return pp.resolve() if pp.is_absolute() else (root / pp).resolve()

    jsonl_target = str(experiments_jsonl)
    try:
        if (
            not str(experiments_jsonl).strip()
            or not str(knob_schema_path).strip()
            or not str(table_config_path).strip()
        ):
            raise ValueError(
                "experiments_jsonl, knob_schema_path, and table_config_path must be non-empty when tuning logging is enabled"
            )
        jsonl_resolved = _resolve(experiments_jsonl)
        knob_path = _resolve(knob_schema_path)
        table_path = _resolve(table_config_path)
        strat_resolved: Path | None = None
        if strategy_config_path is not None and str(strategy_config_path).strip():
            strat_resolved = _resolve(strategy_config_path)

        names = [str(n).strip() for n in (profile_names_in_benchmark or []) if str(n).strip()]
        selected = str(profile_name).strip() if profile_name else ""
        if len(names) > 1 and not selected:
            raise ValueError(
                "tuning_characterization.profile_name is required when benchmarking multiple profiles"
            )

        knobs = extract_knobs_from_table_config(
            table_config_path=table_path,
            knob_schema_path=knob_path,
        )
        schema = load_knob_schema(knob_path)
        hashes = build_input_artifact_hashes(
            table_config_path=table_path,
            strategy_config_path=strat_resolved if strat_resolved is not None else None,
        )
        experiment_id = uuid.uuid4().hex
        manifest = build_tuning_manifest(
            knob_schema_id=str(schema["schema_id"]),
            knobs=knobs,
            input_artifact_hashes=hashes,
            experiment_id=experiment_id,
        )
        out_dir = Path(benchmark_outdir).resolve()
        write_tuning_manifest(out_dir / TUNING_MANIFEST_BASENAME, manifest)
        ingest_profile = selected if selected else None
        rec = ingest_benchmark_run_directory(out_dir, profile_name=ingest_profile)
        append_experiments_jsonl(jsonl_resolved, [rec])
        return {
            "logged": True,
            "experiments_jsonl": str(jsonl_resolved),
            "error": "",
        }
    except Exception as exc:
        log.warning("Tuning observation log skipped: %s", exc)
        try:
            jsonl_display = str(_resolve(experiments_jsonl))
        except Exception:
            jsonl_display = jsonl_target
        return {
            "logged": False,
            "experiments_jsonl": jsonl_display,
            "error": str(exc),
        }


def resolve_run_directories(
    *,
    runs: list[str] | None = None,
    glob_pattern: str = "",
    latest: int | None = None,
) -> list[Path]:
    explicit = [Path(item).resolve() for item in list(runs or []) if str(item).strip()]
    if explicit and glob_pattern:
        raise ValueError("Use either runs= or glob_pattern=, not both")
    if explicit:
        return explicit
    if not glob_pattern:
        raise ValueError("Either runs= or glob_pattern= must be provided")
    matched = sorted(Path(p).resolve() for p in glob.glob(glob_pattern))
    if latest is not None:
        matched = matched[-max(int(latest), 0) :]
    if not matched:
        raise FileNotFoundError(f"No runs matched glob '{glob_pattern}'")
    return matched


def append_experiments_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")


def load_experiments_jsonl(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _knob_bounds_matrix(knob_schema: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    names: list[str] = []
    lows: list[float] = []
    highs: list[float] = []
    for spec in list(knob_schema.get("knobs", []) or []):
        n = str(spec.get("name", "")).strip()
        if not n:
            continue
        names.append(n)
        lows.append(float(spec["low"]))
        highs.append(float(spec["high"]))
    return names, np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)


class TuningSearchStrategy(Protocol):
    def propose(
        self,
        *,
        knob_schema: dict[str, Any],
        experiments: list[dict[str, Any]],
        n: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]: ...


@dataclass
class RandomSearchStrategy:
    """Uniform random in bounds; respects int knobs via schema type."""

    def propose(
        self,
        *,
        knob_schema: dict[str, Any],
        experiments: list[dict[str, Any]],
        n: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        del experiments
        specs = list(knob_schema.get("knobs", []) or [])
        out: list[dict[str, Any]] = []
        for _ in range(n):
            knobs: dict[str, Any] = {}
            for spec in specs:
                name = str(spec["name"])
                lo, hi = float(spec["low"]), float(spec["high"])
                u = rng.uniform(lo, hi)
                t = str(spec.get("type", "float")).lower()
                knobs[name] = int(round(u)) if t == "int" else float(u)
            out.append({"knobs": knobs})
        return out


@dataclass
class ExpectedImprovementGPSearchStrategy:
    """Sklearn GP regression + expected improvement (minimization on penalized_scalar)."""

    n_candidates: int = 512
    xi: float = 0.01

    def propose(
        self,
        *,
        knob_schema: dict[str, Any],
        experiments: list[dict[str, Any]],
        n: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        names, lows, highs = _knob_bounds_matrix(knob_schema)
        dim = len(names)
        if dim == 0:
            return RandomSearchStrategy().propose(
                knob_schema=knob_schema, experiments=experiments, n=n, rng=rng
            )

        trusted = [e for e in experiments if e.get("knobs_trusted") and e.get("knobs")]
        X_list: list[list[float]] = []
        y_list: list[float] = []
        for e in trusted:
            vec = [_coerce_for_vector(names, e["knobs"], knob_schema)]
            if vec[0] is None:
                continue
            X_list.append(vec[0])
            y_list.append(float(e["objectives"]["penalized_scalar"]))

        if len(X_list) < 2:
            return RandomSearchStrategy().propose(
                knob_schema=knob_schema, experiments=experiments, n=n, rng=rng
            )

        X = np.asarray(X_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        Xn = (X - lows) / (highs - lows + 1e-12)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(dim), length_scale_bounds=(1e-2, 10.0)
        )
        kernel += WhiteKernel(1e-5, (1e-8, 1e-1))
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True, random_state=int(rng.integers(0, 2**31))
        )
        gp.fit(Xn, y)

        candidates = rng.uniform(0.0, 1.0, size=(self.n_candidates, dim))
        mean, std = gp.predict(candidates, return_std=True)
        std = np.maximum(std, 1e-9)
        y_best = float(np.min(y))
        ei = _expected_improvement_min(mean, std, y_best, xi=self.xi)
        order = np.argsort(-ei)
        proposals: list[dict[str, Any]] = []
        used: set[tuple[float, ...]] = set()
        for idx in order:
            if len(proposals) >= n:
                break
            raw = candidates[idx] * (highs - lows) + lows
            knobs = _denormalize_knob_dict(names, raw.tolist(), knob_schema)
            key = tuple(float(knobs[k]) for k in names)
            if key in used:
                continue
            used.add(key)
            proposals.append({"knobs": knobs})
        while len(proposals) < n:
            extra = RandomSearchStrategy().propose(
                knob_schema=knob_schema, experiments=[], n=1, rng=rng
            )
            proposals.extend(extra)
        return proposals[:n]


def _expected_improvement_min(
    mean: np.ndarray, std: np.ndarray, y_best: float, *, xi: float
) -> np.ndarray:
    """EI for minimization (with exploration xi)."""
    from scipy.stats import norm

    imp = y_best - mean - xi
    z = imp / std
    return imp * norm.cdf(z) + std * norm.pdf(z)


def _coerce_for_vector(
    names: list[str], knobs: dict[str, Any], knob_schema: dict[str, Any]
) -> list[float] | None:
    specs = {str(s["name"]): s for s in list(knob_schema.get("knobs", []) or [])}
    row: list[float] = []
    for name in names:
        if name not in knobs:
            return None
        spec = specs[name]
        v = _coerce_knob_value(knobs[name], spec)
        row.append(float(v))
    return row


def _denormalize_knob_dict(
    names: list[str], values: list[float], knob_schema: dict[str, Any]
) -> dict[str, Any]:
    specs = {str(s["name"]): s for s in list(knob_schema.get("knobs", []) or [])}
    out: dict[str, Any] = {}
    for i, name in enumerate(names):
        spec = specs[name]
        t = str(spec.get("type", "float")).lower()
        v = values[i]
        out[name] = int(round(v)) if t == "int" else float(v)
    return out


@dataclass
class NSGAIISurrogateSearchStrategy:
    """Two-objective search on a surrogate (trust_reject_cells, numeric_hits); validates top-k with true runs externally."""

    pop_size: int = 12
    n_gen: int = 15

    def propose(
        self,
        *,
        knob_schema: dict[str, Any],
        experiments: list[dict[str, Any]],
        n: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
        from sklearn.ensemble import RandomForestRegressor

        names, lows, highs = _knob_bounds_matrix(knob_schema)
        dim = len(names)
        if dim == 0 or n <= 0:
            return RandomSearchStrategy().propose(
                knob_schema=knob_schema, experiments=experiments, n=max(1, n), rng=rng
            )

        trusted = [e for e in experiments if e.get("knobs_trusted") and e.get("knobs")]
        X_list: list[list[float]] = []
        y1: list[float] = []
        y2: list[float] = []
        for e in trusted:
            vec = _coerce_for_vector(names, e["knobs"], knob_schema)
            if vec is None:
                continue
            X_list.append(vec)
            obj = dict(e.get("objectives") or {})
            y1.append(float(obj.get("trust_region_reject_cells", 0.0)))
            y2.append(float(obj.get("total_numeric_hits", 0.0)))

        if len(X_list) < 3:
            return ExpectedImprovementGPSearchStrategy().propose(
                knob_schema=knob_schema, experiments=experiments, n=n, rng=rng
            )

        X = np.asarray(X_list, dtype=float)
        Xn = (X - lows) / (highs - lows + 1e-12)
        m1 = RandomForestRegressor(n_estimators=24, random_state=int(rng.integers(0, 2**31)))
        m2 = RandomForestRegressor(n_estimators=24, random_state=int(rng.integers(0, 2**31)))
        m1.fit(Xn, np.asarray(y1))
        m2.fit(Xn, np.asarray(y2))

        class _Surr(Problem):
            def __init__(self) -> None:
                super().__init__(n_var=dim, n_obj=2, n_constr=0, xl=0.0, xu=1.0)

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = np.column_stack([m1.predict(x), m2.predict(x)])

        problem = _Surr()
        algorithm = NSGA2(pop_size=self.pop_size)
        res = minimize(
            problem,
            algorithm,
            ("n_gen", self.n_gen),
            seed=int(rng.integers(0, 2**31)),
            verbose=False,
        )
        if res.X is None or len(res.X) == 0:
            return RandomSearchStrategy().propose(
                knob_schema=knob_schema, experiments=experiments, n=n, rng=rng
            )
        # Non-dominated sort already in res.F; take first n unique knob tuples
        proposals: list[dict[str, Any]] = []
        used: set[tuple[float, ...]] = set()
        for row in res.X:
            raw = np.asarray(row) * (highs - lows) + lows
            knobs = _denormalize_knob_dict(names, raw.tolist(), knob_schema)
            key = tuple(float(knobs[k]) for k in names)
            if key in used:
                continue
            used.add(key)
            proposals.append({"knobs": knobs})
            if len(proposals) >= n:
                break
        while len(proposals) < n:
            proposals.extend(
                RandomSearchStrategy().propose(
                    knob_schema=knob_schema, experiments=[], n=n - len(proposals), rng=rng
                )
            )
        return proposals[:n]


STRATEGY_REGISTRY: dict[str, Callable[[], TuningSearchStrategy]] = {
    "random": lambda: RandomSearchStrategy(),
    "gp_ei": lambda: ExpectedImprovementGPSearchStrategy(),
    "nsga2_surrogate": lambda: NSGAIISurrogateSearchStrategy(),
}


def run_tuning_batch(
    *,
    knob_schema_path: str | Path,
    base_table_config_path: str | Path,
    strategy_config_path: str | Path | None,
    experiments_jsonl_path: str | Path,
    study_outdir: str | Path,
    search_strategy: str,
    n_trials: int,
    refresh_table: bool,
    run_benchmark: bool,
    benchmark_run_dir: str | Path | None,
    tuned_params_path: str | Path | None,
    handoff_artifact_path: str | Path | None,
    runtime_strategy_config: str | Path | None,
    benchmark_profiles: list[str] | None,
    window_angle_deg: float,
    docker_timeout_s: int,
    refresh_custom_solver: bool,
    max_benchmarks: int,
    dry_run: bool,
    profile_name: str | None,
    rng_seed: int,
) -> dict[str, Any]:
    """Stage configs, optionally refresh table + benchmark, append JSONL."""
    knob_schema = load_knob_schema(knob_schema_path)
    schema_id = str(knob_schema["schema_id"])
    rng = np.random.default_rng(rng_seed)
    experiments_path = Path(experiments_jsonl_path)
    experiments = load_experiments_jsonl(experiments_path) if experiments_path.exists() else []

    factory = STRATEGY_REGISTRY.get(search_strategy)
    if factory is None:
        raise ValueError(f"Unknown search_strategy '{search_strategy}'")
    strategy = factory()
    proposals = strategy.propose(
        knob_schema=knob_schema, experiments=experiments, n=n_trials, rng=rng
    )

    study_root = Path(study_outdir)
    study_root.mkdir(parents=True, exist_ok=True)
    new_records: list[dict[str, Any]] = []
    benchmarks_run = 0

    for prop in proposals:
        knobs = dict(prop["knobs"])
        trial_id = f"trial_{uuid.uuid4().hex[:12]}"
        trial_dir = study_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        staged_path = trial_dir / "staged_runtime_table_config.json"
        payload = apply_knobs_to_table_config(
            table_config_path=base_table_config_path,
            knob_schema=knob_schema,
            knobs=knobs,
        )
        write_staged_table_config(payload=payload, output_path=staged_path)
        hashes = build_input_artifact_hashes(
            table_config_path=staged_path,
            strategy_config_path=strategy_config_path,
        )
        manifest = build_tuning_manifest(
            knob_schema_id=schema_id,
            knobs=knobs,
            input_artifact_hashes=hashes,
            experiment_id=trial_id,
            staged_table_config_path=str(staged_path),
        )
        write_tuning_manifest(trial_dir / TUNING_MANIFEST_BASENAME, manifest)

        if dry_run:
            new_records.append(
                {
                    "schema_version": EXPERIMENT_STORE_SCHEMA_VERSION,
                    "run_dir": str(trial_dir),
                    "profile_name": "",
                    "knobs": knobs,
                    "knobs_trusted": True,
                    "knob_schema_id": schema_id,
                    "objectives": {},
                    "runtime_table_hash": "",
                    "dry_run": True,
                    "ingested_utc": datetime.now(UTC).isoformat(),
                }
            )
            continue

        if refresh_table:
            from larrak2.cli.validate_simulation import main as _validate_simulation_main

            rc = _validate_simulation_main(
                [
                    "runtime-chemistry-table",
                    "--config",
                    str(staged_path.resolve()),
                    "--refresh",
                ]
            )
            if rc != 0:
                new_records.append(
                    {
                        "schema_version": EXPERIMENT_STORE_SCHEMA_VERSION,
                        "run_dir": str(trial_dir),
                        "knobs": knobs,
                        "knobs_trusted": True,
                        "knob_schema_id": schema_id,
                        "error": "runtime_chemistry_table_failed",
                        "returncode": rc,
                    }
                )
                continue

        bench_out = trial_dir / "benchmark_out"
        if run_benchmark and benchmark_run_dir and tuned_params_path and handoff_artifact_path:
            if benchmarks_run >= max_benchmarks:
                new_records.append(
                    {
                        "schema_version": EXPERIMENT_STORE_SCHEMA_VERSION,
                        "run_dir": str(trial_dir),
                        "knobs": knobs,
                        "knobs_trusted": True,
                        "knob_schema_id": schema_id,
                        "note": "benchmark_skipped_max_benchmarks",
                    }
                )
                continue
            bench_out.mkdir(parents=True, exist_ok=True)
            profs = benchmark_profiles or ["chem323_lookup_strict"]
            rsc = (
                runtime_strategy_config
                or "data/simulation_validation/engine_runtime_mechanism_strategy_multitable_v1.json"
            )
            argv = [
                "engine-restart-benchmark",
                "--run-dir",
                str(Path(benchmark_run_dir).resolve()),
                "--tuned-params",
                str(Path(tuned_params_path).resolve()),
                "--handoff-artifact",
                str(Path(handoff_artifact_path).resolve()),
                "--outdir",
                str(bench_out.resolve()),
                "--profiles",
                *profs,
                "--window-angle-deg",
                str(window_angle_deg),
                "--docker-timeout-s",
                str(docker_timeout_s),
                "--runtime-strategy-config",
                str(Path(rsc).resolve()),
            ]
            if refresh_custom_solver:
                argv.append("--refresh-custom-solver")
            from larrak2.cli.validate_simulation import main as _validate_simulation_main

            rc = _validate_simulation_main(argv)
            benchmarks_run += 1
            if rc != 0:
                new_records.append(
                    {
                        "schema_version": EXPERIMENT_STORE_SCHEMA_VERSION,
                        "run_dir": str(bench_out),
                        "knobs": knobs,
                        "knobs_trusted": True,
                        "knob_schema_id": schema_id,
                        "error": "benchmark_failed",
                        "returncode": rc,
                    }
                )
                continue
            # Copy manifest next to summary for ingest convention
            summary_root = bench_out
            write_tuning_manifest(summary_root / TUNING_MANIFEST_BASENAME, manifest)
            rec = ingest_benchmark_run_directory(summary_root, profile_name=profile_name)
            rec["trial_id"] = trial_id
            new_records.append(rec)
        else:
            new_records.append(
                {
                    "schema_version": EXPERIMENT_STORE_SCHEMA_VERSION,
                    "run_dir": str(trial_dir),
                    "knobs": knobs,
                    "knobs_trusted": True,
                    "knob_schema_id": schema_id,
                    "note": "staged_only_no_benchmark",
                }
            )

    if new_records and not dry_run:
        append_experiments_jsonl(experiments_path, new_records)
    return {
        "study_outdir": str(study_root),
        "proposals": len(proposals),
        "new_records": len(new_records),
        "benchmarks_run": benchmarks_run,
        "experiments_jsonl": str(experiments_path),
    }


__all__ = [
    "TUNING_MANIFEST_BASENAME",
    "EXPERIMENT_STORE_SCHEMA_VERSION",
    "apply_knobs_to_table_config",
    "append_experiments_jsonl",
    "build_input_artifact_hashes",
    "build_tuning_manifest",
    "compute_objective_vector",
    "ExpectedImprovementGPSearchStrategy",
    "extract_knobs_from_table_config",
    "ingest_benchmark_run_directory",
    "load_experiments_jsonl",
    "load_knob_schema",
    "load_tuning_manifest",
    "maybe_log_tuning_observation",
    "NSGAIISurrogateSearchStrategy",
    "RandomSearchStrategy",
    "resolve_run_directories",
    "run_tuning_batch",
    "sha256_file",
    "STRATEGY_REGISTRY",
    "write_staged_table_config",
    "write_tuning_manifest",
]
