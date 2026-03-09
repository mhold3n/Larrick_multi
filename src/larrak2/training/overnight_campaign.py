"""Overnight F2 NN campaign helpers."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.adapters.calculix import CalculiXRunner
from larrak2.architecture.contracts import CONTRACT_VERSION
from larrak2.core.artifact_paths import (
    DEFAULT_CALCULIX_NN_ARTIFACT,
    DEFAULT_OPENFOAM_NN_ARTIFACT,
    stack_artifact_path_for_fidelity,
)
from larrak2.core.constraints import get_constraint_names
from larrak2.core.encoding import N_TOTAL, bounds, mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import BreathingConfig, EvalContext
from larrak2.orchestration.adapters.simulation_adapter import (
    candidate_calculix_params,
    candidate_openfoam_geometry_args,
    candidate_openfoam_params,
)
from larrak2.pipelines.openfoam import OpenFoamPipeline
from larrak2.pipelines.principles_core import (
    expand_reduced_vector,
    reduced_bounds,
    reduced_seed_states,
)
from larrak2.surrogate.quality_contract import validate_artifact_quality
from larrak2.surrogate.stack.runtime import default_feature_names
from larrak2.training.workflows import (
    _infer_objective_names,
    train_calculix_workflow,
    train_openfoam_workflow,
    train_stack_surrogate_workflow,
)

DEFAULT_PROFILE_PATH = Path("data/training/f2_nn_overnight_core_edge_v1.json")
DEFAULT_OPENFOAM_TEMPLATE = Path("openfoam_templates/opposed_piston_rotary_valve_sliding_case")
DEFAULT_PRINCIPLES_PROFILE = Path("data/optimization/principles_frontier_profile_v2.json")
DEFAULT_ANCHOR_MANIFEST = Path("data/thermo/anchor_manifest_v1.json")


class CampaignError(RuntimeError):
    """Fatal overnight campaign error."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_overnight_campaign_profile(path: str | Path | None = None) -> dict[str, Any]:
    profile_path = (_repo_root() / Path(path or DEFAULT_PROFILE_PATH)).resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    required = ("profile_id", "fuel_name", "breathing_defaults", "openfoam", "calculix", "stack")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Overnight campaign profile missing keys: {missing}")
    payload["_path"] = str(profile_path)
    return payload


def _load_principles_profile(path: str | Path | None = None) -> dict[str, Any]:
    profile_path = (_repo_root() / Path(path or DEFAULT_PRINCIPLES_PROFILE)).resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["_path"] = str(profile_path)
    return payload


def _breathing_from_profile(profile: dict[str, Any]) -> BreathingConfig:
    raw = dict(profile.get("breathing_defaults", {}) or {})
    return BreathingConfig(
        bore_mm=float(raw.get("bore_mm", 80.0)),
        stroke_mm=float(raw.get("stroke_mm", 90.0)),
        intake_port_area_m2=float(raw.get("intake_port_area_m2", 4.0e-4)),
        exhaust_port_area_m2=float(raw.get("exhaust_port_area_m2", 4.0e-4)),
        p_manifold_Pa=float(raw.get("p_manifold_Pa", 101325.0)),
        p_back_Pa=float(raw.get("p_back_Pa", 101325.0)),
        compression_ratio=float(raw.get("compression_ratio", 10.0)),
        fuel_name=str(profile.get("fuel_name", "gasoline")),
        valve_timing_mode="candidate",
    )


def _stable_jsonl_append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _copy_with_backup(src: Path, dst: Path, *, backup_root: Path) -> dict[str, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_path = backup_root / dst.name
    if dst.exists():
        shutil.copy2(dst, backup_path)
    shutil.copy2(src, dst)
    return {"dst": str(dst), "backup": str(backup_path) if backup_path.exists() else ""}


def _latin_hypercube(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, dim), dtype=np.float64)
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.uniform(size=(n, dim))
    out = np.zeros((n, dim), dtype=np.float64)
    for j in range(dim):
        pts = cut[:n] + u[:, j] * (cut[1:] - cut[:n])
        out[:, j] = pts[rng.permutation(n)]
    return out


def _clip_candidates(X: np.ndarray) -> np.ndarray:
    xl, xu = bounds()
    return np.clip(np.asarray(X, dtype=np.float64), xl.reshape(1, -1), xu.reshape(1, -1))


def _quasirandom_candidates(n: int, *, rng: np.random.Generator) -> np.ndarray:
    xl, xu = bounds()
    lhs = _latin_hypercube(int(n), int(N_TOTAL), rng)
    return xl.reshape(1, -1) + lhs * (xu - xl).reshape(1, -1)


def _principles_candidates(
    n: int,
    *,
    rpm: float,
    rng: np.random.Generator,
    principles_profile: dict[str, Any],
) -> np.ndarray:
    seeds = list(reduced_seed_states(principles_profile).values())
    if not seeds or n <= 0:
        return np.zeros((0, N_TOTAL), dtype=np.float64)
    x_rows: list[np.ndarray] = []
    red_xl, red_xu = reduced_bounds()
    span = np.maximum(red_xu - red_xl, 1e-9)
    for i in range(int(n)):
        seed = np.asarray(seeds[i % len(seeds)], dtype=np.float64)
        noise = rng.normal(loc=0.0, scale=0.035, size=seed.shape[0]) * span
        x_full, _ = expand_reduced_vector(
            seed + noise, profile_payload=principles_profile, rpm=float(rpm)
        )
        x_rows.append(np.asarray(x_full, dtype=np.float64))
    return _clip_candidates(np.vstack(x_rows))


def _local_perturb_candidates(
    n: int,
    *,
    rpm: float,
    rng: np.random.Generator,
    principles_profile: dict[str, Any],
) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, N_TOTAL), dtype=np.float64)
    xl, xu = bounds()
    span = np.maximum(xu - xl, 1e-9)
    base_candidates = [np.asarray(mid_bounds_candidate(), dtype=np.float64)]
    for seed in reduced_seed_states(principles_profile).values():
        x_full, _ = expand_reduced_vector(seed, profile_payload=principles_profile, rpm=float(rpm))
        base_candidates.append(np.asarray(x_full, dtype=np.float64))
    x_rows: list[np.ndarray] = []
    for i in range(int(n)):
        base = np.asarray(base_candidates[i % len(base_candidates)], dtype=np.float64)
        noise = rng.normal(loc=0.0, scale=0.02, size=base.shape[0]) * span
        x_rows.append(base + noise)
    return _clip_candidates(np.vstack(x_rows))


def _build_candidate_pool(
    *,
    rpm: float,
    rng: np.random.Generator,
    principles_profile: dict[str, Any],
    n_quasi: int,
    n_principles: int,
    n_local: int,
) -> np.ndarray:
    rows = [
        _quasirandom_candidates(n_quasi, rng=rng),
        _principles_candidates(
            n_principles, rpm=rpm, rng=rng, principles_profile=principles_profile
        ),
        _local_perturb_candidates(n_local, rpm=rpm, rng=rng, principles_profile=principles_profile),
    ]
    rows = [row for row in rows if row.size > 0]
    if not rows:
        return np.zeros((0, N_TOTAL), dtype=np.float64)
    return _clip_candidates(np.vstack(rows))


def _f0_context(profile: dict[str, Any], *, rpm: float, torque: float, seed: int) -> EvalContext:
    return EvalContext(
        rpm=float(rpm),
        torque=float(torque),
        fidelity=0,
        seed=int(seed),
        breathing=_breathing_from_profile(profile),
        surrogate_validation_mode="off",
        thermo_model="two_zone_eq_v1",
        thermo_anchor_manifest_path=str(DEFAULT_ANCHOR_MANIFEST),
        thermo_timing_profile_path="data/thermo/valve_timing_profile_v1.json",
        thermo_chemistry_profile_path="data/thermo/hybrid_chemistry_profile_v1.json",
    )


def _f2_context(
    profile: dict[str, Any],
    *,
    rpm: float,
    torque: float,
    seed: int,
    openfoam_model_path: str,
    calculix_model_path: str,
    anchor_manifest_path: str,
) -> EvalContext:
    return EvalContext(
        rpm=float(rpm),
        torque=float(torque),
        fidelity=2,
        seed=int(seed),
        breathing=_breathing_from_profile(profile),
        surrogate_validation_mode="strict",
        openfoam_model_path=str(openfoam_model_path),
        calculix_stress_mode="nn",
        calculix_model_path=str(calculix_model_path),
        thermo_model="two_zone_eq_v1",
        thermo_anchor_manifest_path=str(anchor_manifest_path),
        thermo_timing_profile_path="data/thermo/valve_timing_profile_v1.json",
        thermo_chemistry_profile_path="data/thermo/hybrid_chemistry_profile_v1.json",
    )


def _collect_stack_targets(result: Any, *, constraint_names: tuple[str, ...]) -> np.ndarray:
    diag = dict(result.diag or {})
    obj_names = [str(v) for v in ((diag.get("objectives", {}) or {}).get("names", []) or [])]
    if len(obj_names) != len(result.F):
        raise ValueError("Objective names/values mismatch in evaluator diagnostics")
    obj_map = {name: float(result.F[i]) for i, name in enumerate(obj_names)}
    con_map: dict[str, float] = {}
    for rec in list(diag.get("constraints", []) or []):
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name", "")).strip()
        if not name:
            continue
        con_map[name] = float(rec.get("scaled_raw", rec.get("scaled", 0.0)))
    missing = [name for name in constraint_names if name not in con_map]
    if missing:
        raise ValueError(f"Constraint mapping missing names: {missing}")
    ordered = [obj_map[name] for name in obj_names] + [con_map[name] for name in constraint_names]
    return np.asarray(ordered, dtype=np.float64)


def _build_truth_cases(
    profile: dict[str, Any], *, principles_profile: dict[str, Any]
) -> list[dict[str, Any]]:
    points = list((profile.get("openfoam", {}) or {}).get("truth_operating_points", []) or [])
    rng = np.random.default_rng(int((profile.get("openfoam", {}) or {}).get("truth_seed", 0)))
    seeds = list(reduced_seed_states(principles_profile).values())
    if not seeds:
        raise ValueError("Principles seed states are required for truth-case construction")
    cases: list[dict[str, Any]] = []
    for i, rec in enumerate(points):
        rpm = float(rec["rpm"])
        torque = float(rec["torque"])
        reduced = np.asarray(seeds[i % len(seeds)], dtype=np.float64)
        if i >= len(seeds):
            reduced = reduced + rng.normal(loc=0.0, scale=0.02, size=reduced.shape[0])
        x_full, _ = expand_reduced_vector(reduced, profile_payload=principles_profile, rpm=rpm)
        cases.append(
            {
                "id": f"truth_{i:02d}",
                "rpm": rpm,
                "torque": torque,
                "x": np.asarray(x_full, dtype=np.float64),
                "category": "truth_anchor",
            }
        )
    return cases


def _run_openfoam_case(
    *,
    case: dict[str, Any],
    profile: dict[str, Any],
    pipeline: OpenFoamPipeline,
    run_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = _f0_context(
        profile, rpm=float(case["rpm"]), torque=float(case["torque"]), seed=int(seed)
    )
    candidate = {"id": str(case["id"]), "x": np.asarray(case["x"], dtype=np.float64)}
    eval_result = evaluate_candidate(np.asarray(case["x"], dtype=np.float64), context)
    params = candidate_openfoam_params(candidate, context, eval_diag=eval_result.diag)
    geometry_args = candidate_openfoam_geometry_args(candidate, context)
    result = pipeline.execute(run_dir=run_dir, params=params, geometry_args=geometry_args)
    openfoam_payload = {
        "scavenging_efficiency": float(result.get("scavenging_efficiency", float("nan"))),
        "trapped_mass": float(result.get("trapped_mass", float("nan"))),
        "residual_fraction": float(result.get("residual_fraction", float("nan"))),
        "trapped_o2_mass": float(result.get("trapped_o2_mass", float("nan"))),
        "stage": str(result.get("stage", "complete")),
    }
    record = {
        **params,
        "ok": bool(result.get("ok", False)),
        "m_air_trapped": float(result.get("trapped_mass", 0.0)),
        "scavenging_efficiency": float(result.get("scavenging_efficiency", 0.0)),
        "residual_fraction": float(result.get("residual_fraction", 0.0)),
        "trapped_o2_mass": float(result.get("trapped_o2_mass", 0.0)),
        "candidate_id": str(case["id"]),
        "category": str(case.get("category", "core")),
        "run_dir": str(run_dir),
    }
    truth_record = {
        "run_id": str(case.get("run_id", "overnight_f2")),
        "candidate_id": str(case["id"]),
        "truth_backend": "openfoam_dispatch",
        "truth_ok": bool(result.get("ok", False)),
        "rpm": float(case["rpm"]),
        "torque": float(case["torque"]),
        "operating_point": {
            "rpm": float(case["rpm"]),
            "torque": float(case["torque"]),
            "fidelity": 2,
        },
        "candidate": {"id": str(case["id"]), "x": np.asarray(case["x"], dtype=np.float64).tolist()},
        "openfoam": openfoam_payload,
        "diag": {"thermo": dict((eval_result.diag or {}).get("thermo", {}))},
    }
    return record, truth_record


def _filter_openfoam_success(raw_jsonl: Path, filtered_jsonl: Path) -> dict[str, int]:
    from larrak2.surrogate.openfoam_nn import DEFAULT_TARGET_KEYS

    n_total = 0
    n_kept = 0
    filtered_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with filtered_jsonl.open("w", encoding="utf-8") as handle:
        for line in raw_jsonl.read_text(encoding="utf-8").splitlines():
            row = line.strip()
            if not row:
                continue
            n_total += 1
            rec = json.loads(row)
            if not bool(rec.get("ok", False)):
                continue
            try:
                values = [float(rec.get(name)) for name in DEFAULT_TARGET_KEYS]
            except Exception:
                continue
            if any(not np.isfinite(v) for v in values):
                continue
            handle.write(json.dumps(rec, ensure_ascii=True) + "\n")
            n_kept += 1
    return {"n_total": int(n_total), "n_kept": int(n_kept)}


def _build_anchor_manifest(
    *,
    input_paths: list[Path],
    output_path: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    from tools.build_thermo_anchor_manifest import build_manifest

    openfoam_cfg = dict(profile.get("openfoam", {}) or {})
    args = argparse.Namespace(
        input=[str(p) for p in input_paths],
        output=str(output_path),
        version="thermo_anchor_v1",
        source="truth_runs",
        max_anchors=0,
        rpm_min=float((openfoam_cfg.get("validated_envelope", {}) or {}).get("rpm_min", 1000.0)),
        rpm_max=float((openfoam_cfg.get("validated_envelope", {}) or {}).get("rpm_max", 7000.0)),
        torque_min=float(
            (openfoam_cfg.get("validated_envelope", {}) or {}).get("torque_min", 40.0)
        ),
        torque_max=float(
            (openfoam_cfg.get("validated_envelope", {}) or {}).get("torque_max", 400.0)
        ),
        delta_m_air_rel_max=float(
            (openfoam_cfg.get("thresholds", {}) or {}).get("delta_m_air_rel_max", 0.10)
        ),
        delta_residual_abs_max=float(
            (openfoam_cfg.get("thresholds", {}) or {}).get("delta_residual_abs_max", 0.05)
        ),
        delta_scavenging_abs_max=float(
            (openfoam_cfg.get("thresholds", {}) or {}).get("delta_scavenging_abs_max", 0.08)
        ),
    )
    manifest = build_manifest(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_truth_anchor_bundle(
    *,
    profile: dict[str, Any],
    outdir: Path,
    template_dir: Path,
    solver_cmd: str,
    docker_timeout_s: int,
    docker_image: str | None,
    log_fn,
) -> dict[str, Any]:
    principles_profile = _load_principles_profile(profile.get("principles_profile_path"))
    cfg = dict(profile.get("openfoam", {}) or {})
    cases = _build_truth_cases(profile, principles_profile=principles_profile)
    truth_jsonl = outdir / "truth_records_generated.jsonl"
    raw_jsonl = outdir / "truth_case_records.jsonl"
    pipeline = OpenFoamPipeline(
        template_dir=template_dir,
        solver_cmd=str(solver_cmd),
        docker_timeout_s=int(docker_timeout_s),
        docker_image=docker_image,
    )
    successes = 0
    for i, case in enumerate(cases):
        case = dict(case)
        case["run_id"] = outdir.name
        run_dir = outdir / "truth_runs" / str(case["id"])
        record, truth_record = _run_openfoam_case(
            case=case,
            profile=profile,
            pipeline=pipeline,
            run_dir=run_dir,
            seed=int(cfg.get("truth_seed", 0)) + i,
        )
        _stable_jsonl_append(raw_jsonl, record)
        if bool(truth_record.get("truth_ok", False)):
            successes += 1
            _stable_jsonl_append(truth_jsonl, truth_record)
    existing_inputs = [
        (_repo_root() / Path(path)).resolve()
        for path in list(cfg.get("existing_truth_inputs", []) or [])
        if (_repo_root() / Path(path)).resolve().exists()
    ]
    anchor_manifest_path = outdir / "anchor_manifest_truth.json"
    manifest = _build_anchor_manifest(
        input_paths=[*existing_inputs, truth_jsonl],
        output_path=anchor_manifest_path,
        profile=profile,
    )
    target_anchor_count = int(cfg.get("truth_anchor_target", 9))
    if int(len(manifest.get("anchors", []))) < target_anchor_count:
        raise CampaignError(
            "Truth anchor bundle is insufficient for overnight promotion: "
            f"anchors={len(manifest.get('anchors', []))}, required>={target_anchor_count}"
        )
    log_fn(
        "Built truth anchor bundle",
        truth_cases=len(cases),
        truth_successes=successes,
        anchor_count=len(manifest.get("anchors", [])),
        anchor_manifest=anchor_manifest_path,
    )
    return {
        "truth_records_path": str(truth_jsonl),
        "truth_case_records_path": str(raw_jsonl),
        "anchor_manifest_path": str(anchor_manifest_path),
        "anchor_count": int(len(manifest.get("anchors", []))),
        "existing_truth_inputs": [str(p) for p in existing_inputs],
        "truth_successes": int(successes),
        "truth_cases": int(len(cases)),
    }


def build_openfoam_training_dataset(
    *,
    profile: dict[str, Any],
    outdir: Path,
    template_dir: Path,
    solver_cmd: str,
    docker_timeout_s: int,
    docker_image: str | None,
    log_fn,
) -> tuple[Path, dict[str, Any]]:

    principles_profile = _load_principles_profile(profile.get("principles_profile_path"))
    cfg = dict(profile.get("openfoam", {}) or {})
    runtime = dict(cfg.get("runtime", {}) or {})
    rng = np.random.default_rng(int(cfg.get("dataset_seed", 42)))
    pipeline = OpenFoamPipeline(
        template_dir=template_dir,
        solver_cmd=str(solver_cmd),
        docker_timeout_s=int(docker_timeout_s),
        docker_image=docker_image,
    )
    raw_jsonl = outdir / "results_raw.jsonl"
    filtered_jsonl = outdir / "results_train.jsonl"
    runs_root = outdir / "runs"

    core = dict(cfg.get("core_corridor", {}) or {})
    edge_points = list(cfg.get("edge_points", []) or [])
    n_core = int(cfg.get("n_core_samples", 0))
    n_edge = int(cfg.get("n_edge_samples", 0))
    n_local = int(cfg.get("n_anchor_perturbations", 0))
    core_points = [
        (
            float(
                rng.uniform(float(core.get("rpm_min", 1800.0)), float(core.get("rpm_max", 2800.0)))
            ),
            float(
                rng.uniform(
                    float(core.get("torque_min", 80.0)), float(core.get("torque_max", 160.0))
                )
            ),
            "core",
        )
        for _ in range(n_core)
    ]
    sampled_edges = [
        (
            float(edge_points[i % len(edge_points)]["rpm"]),
            float(edge_points[i % len(edge_points)]["torque"]),
            "edge",
        )
        for i in range(n_edge)
    ]
    perturb_points = [
        (
            float(edge_points[i % len(edge_points)]["rpm"]),
            float(edge_points[i % len(edge_points)]["torque"]),
            "anchor_perturb",
        )
        for i in range(n_local)
    ]
    plan = core_points + sampled_edges + perturb_points
    success_count = 0
    fail_fast_after = int(cfg.get("fail_fast_after", 12))
    fail_fast_min_success = int(cfg.get("fail_fast_min_success", 8))
    for i, (rpm, torque, category) in enumerate(plan):
        if category == "core":
            x = _build_candidate_pool(
                rpm=rpm,
                rng=rng,
                principles_profile=principles_profile,
                n_quasi=1,
                n_principles=0,
                n_local=0,
            )[0]
        elif category == "edge":
            x = _build_candidate_pool(
                rpm=rpm,
                rng=rng,
                principles_profile=principles_profile,
                n_quasi=0,
                n_principles=1,
                n_local=0,
            )[0]
        else:
            x = _build_candidate_pool(
                rpm=rpm,
                rng=rng,
                principles_profile=principles_profile,
                n_quasi=0,
                n_principles=0,
                n_local=1,
            )[0]
        candidate = {"id": f"of_{i:03d}", "x": np.asarray(x, dtype=np.float64)}
        context = _f0_context(
            profile, rpm=rpm, torque=torque, seed=int(cfg.get("dataset_seed", 42)) + i
        )
        eval_result = evaluate_candidate(np.asarray(x, dtype=np.float64), context)
        params = candidate_openfoam_params(candidate, context, eval_diag=eval_result.diag)
        params["endTime"] = float(runtime.get("endTime", params.get("endTime", 3.0e-4)))
        params["deltaT"] = float(runtime.get("deltaT", params.get("deltaT", 1.0e-4)))
        params["writeInterval"] = int(runtime.get("writeInterval", params.get("writeInterval", 1)))
        params["metricWriteInterval"] = int(
            runtime.get("metricWriteInterval", params.get("metricWriteInterval", 1))
        )
        result = pipeline.execute(
            run_dir=runs_root / str(candidate["id"]),
            params=params,
            geometry_args=candidate_openfoam_geometry_args(candidate, context),
        )
        record = {
            **params,
            "ok": bool(result.get("ok", False)),
            "m_air_trapped": float(result.get("trapped_mass", 0.0)),
            "scavenging_efficiency": float(result.get("scavenging_efficiency", 0.0)),
            "residual_fraction": float(result.get("residual_fraction", 0.0)),
            "trapped_o2_mass": float(result.get("trapped_o2_mass", 0.0)),
            "candidate_id": str(candidate["id"]),
            "category": str(category),
            "run_dir": str(runs_root / str(candidate["id"])),
        }
        _stable_jsonl_append(raw_jsonl, record)
        if bool(record["ok"]):
            success_count += 1
        if i + 1 == fail_fast_after and success_count < fail_fast_min_success:
            raise CampaignError(
                "OpenFOAM overnight fail-fast tripped: "
                f"successes={success_count}/{fail_fast_after}, required>={fail_fast_min_success}"
            )
    counts = _filter_openfoam_success(raw_jsonl, filtered_jsonl)
    min_success_rows = int(cfg.get("min_success_rows", 36))
    if counts["n_kept"] < min_success_rows:
        raise CampaignError(
            "OpenFOAM overnight dataset insufficient: "
            f"successes={counts['n_kept']} required>={min_success_rows}"
        )
    meta = {
        "source": "doe_generated",
        "raw_jsonl": str(raw_jsonl),
        "filtered_jsonl": str(filtered_jsonl),
        "n_total_cases": int(counts["n_total"]),
        "n_success_cases": int(counts["n_kept"]),
        "runtime": runtime,
        "profile_id": str(profile.get("profile_id", "")),
    }
    log_fn(
        "Built OpenFOAM DOE dataset",
        n_total=counts["n_total"],
        n_success=counts["n_kept"],
        filtered_jsonl=filtered_jsonl,
    )
    return filtered_jsonl, meta


def build_calculix_training_dataset(
    *,
    profile: dict[str, Any],
    outdir: Path,
    template_path: Path,
    solver_cmd: str,
    log_fn,
) -> tuple[Path, dict[str, Any]]:
    principles_profile = _load_principles_profile(profile.get("principles_profile_path"))
    cfg = dict(profile.get("calculix", {}) or {})
    rng = np.random.default_rng(int(cfg.get("dataset_seed", 43)))
    runner = CalculiXRunner(template_path=template_path, solver_cmd=str(solver_cmd))
    runs_root = outdir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "results.jsonl"
    core = dict(cfg.get("core_corridor", {}) or {})
    edge_points = list(cfg.get("edge_points", []) or [])
    plan = []
    for _ in range(int(cfg.get("n_core_samples", 0))):
        plan.append(
            (
                float(
                    rng.uniform(
                        float(core.get("rpm_min", 1800.0)), float(core.get("rpm_max", 2800.0))
                    )
                ),
                float(
                    rng.uniform(
                        float(core.get("torque_min", 80.0)), float(core.get("torque_max", 160.0))
                    )
                ),
                "core",
            )
        )
    for i in range(int(cfg.get("n_edge_samples", 0))):
        point = edge_points[i % len(edge_points)]
        plan.append((float(point["rpm"]), float(point["torque"]), "edge"))
    for i in range(int(cfg.get("n_local_samples", 0))):
        point = edge_points[i % len(edge_points)]
        plan.append((float(point["rpm"]), float(point["torque"]), "local"))

    X_rows: list[list[float]] = []
    Y_rows: list[list[float]] = []
    success_count = 0
    fail_fast_after = int(cfg.get("fail_fast_after", 32))
    fail_fast_min_success = int(cfg.get("fail_fast_min_success", 24))
    for i, (rpm, torque, category) in enumerate(plan):
        if category == "core":
            x = _build_candidate_pool(
                rpm=rpm,
                rng=rng,
                principles_profile=principles_profile,
                n_quasi=1,
                n_principles=0,
                n_local=0,
            )[0]
        elif category == "edge":
            x = _build_candidate_pool(
                rpm=rpm,
                rng=rng,
                principles_profile=principles_profile,
                n_quasi=0,
                n_principles=1,
                n_local=0,
            )[0]
        else:
            x = _build_candidate_pool(
                rpm=rpm,
                rng=rng,
                principles_profile=principles_profile,
                n_quasi=0,
                n_principles=0,
                n_local=1,
            )[0]
        context = _f0_context(
            profile, rpm=rpm, torque=torque, seed=int(cfg.get("dataset_seed", 43)) + i
        )
        params = candidate_calculix_params(
            {"id": f"ccx_{i:03d}", "x": np.asarray(x, dtype=np.float64)}, context
        )
        result = runner.execute(
            run_dir=runs_root / f"case_{i:06d}",
            job_name=f"job_{i:06d}",
            params={
                **params,
                "base_radius": float(params["base_radius_mm"]),
                "face_width": float(params["face_width_mm"]),
                "module": float(params["module_mm"]),
            },
        )
        max_stress = result.get("max_stress")
        ok = bool(np.isfinite(float(max_stress))) if max_stress is not None else False
        record = {
            **params,
            "ok": ok,
            "max_stress": float(max_stress) if ok else None,
            "category": str(category),
            "run_dir": str(runs_root / f"case_{i:06d}"),
        }
        _stable_jsonl_append(jsonl_path, record)
        if ok:
            success_count += 1
            X_rows.append([float(params[k]) for k in CCX_KEYS])
            Y_rows.append([float(max_stress)])
        if i + 1 == fail_fast_after and success_count < fail_fast_min_success:
            raise CampaignError(
                "CalculiX overnight fail-fast tripped: "
                f"successes={success_count}/{fail_fast_after}, required>={fail_fast_min_success}"
            )
    if len(X_rows) < int(cfg.get("min_success_rows", 140)):
        raise CampaignError(
            f"CalculiX overnight dataset insufficient: successes={len(X_rows)} required>={int(cfg.get('min_success_rows', 140))}"
        )
    train_path = outdir / "train.npz"
    np.savez(
        train_path,
        X=np.asarray(X_rows, dtype=np.float64),
        Y=np.asarray(Y_rows, dtype=np.float64),
        feature_names=np.asarray(CCX_KEYS, dtype=object),
        target_names=np.asarray(["max_stress"], dtype=object),
    )
    meta = {
        "source": "doe_generated",
        "jsonl": str(jsonl_path),
        "n_total_cases": int(len(plan)),
        "n_success_cases": int(len(X_rows)),
        "profile_id": str(profile.get("profile_id", "")),
    }
    log_fn(
        "Built CalculiX DOE dataset",
        n_total=len(plan),
        n_success=len(X_rows),
        train_npz=train_path,
    )
    return train_path, meta


def build_stack_dataset(
    *,
    profile: dict[str, Any],
    outdir: Path,
    openfoam_model_path: str,
    calculix_model_path: str,
    anchor_manifest_path: str,
    log_fn,
) -> tuple[Path, dict[str, Any]]:
    principles_profile = _load_principles_profile(profile.get("principles_profile_path"))
    cfg = dict(profile.get("stack", {}) or {})
    rng = np.random.default_rng(int(cfg.get("dataset_seed", 44)))
    op_points = [
        (float(rec["rpm"]), float(rec["torque"]))
        for rec in list(cfg.get("operating_points", []) or [])
    ]
    if not op_points:
        raise ValueError("Stack campaign requires non-empty operating_points")
    per_point = dict(cfg.get("per_point", {}) or {})
    n_quasi = int(per_point.get("quasi_random", 96))
    n_principles = int(per_point.get("principles", 48))
    n_local = int(per_point.get("local_perturb", 48))
    feature_names = default_feature_names(N_TOTAL)
    X_rows: list[np.ndarray] = []
    Y_rows: list[np.ndarray] = []
    objective_names: tuple[str, ...] | None = None
    constraint_names = tuple(get_constraint_names(2))
    attempted = 0
    failures = 0
    records_jsonl = outdir / "candidate_records.jsonl"
    failure_limit = float(cfg.get("fail_fast_max_failure_fraction", 0.20))
    for op_idx, (rpm, torque) in enumerate(op_points):
        pool = _build_candidate_pool(
            rpm=rpm,
            rng=rng,
            principles_profile=principles_profile,
            n_quasi=n_quasi,
            n_principles=n_principles,
            n_local=n_local,
        )
        for row_idx, x in enumerate(pool):
            attempted += 1
            context = _f2_context(
                profile,
                rpm=rpm,
                torque=torque,
                seed=int(cfg.get("dataset_seed", 44)) + attempted,
                openfoam_model_path=str(openfoam_model_path),
                calculix_model_path=str(calculix_model_path),
                anchor_manifest_path=str(anchor_manifest_path),
            )
            record = {
                "operating_point": {"rpm": rpm, "torque": torque},
                "candidate_id": f"stack_{op_idx:02d}_{row_idx:03d}",
            }
            try:
                result = evaluate_candidate(np.asarray(x, dtype=np.float64), context)
                diag = dict(result.diag or {})
                names = tuple(
                    str(v) for v in ((diag.get("objectives", {}) or {}).get("names", []) or [])
                )
                if not names:
                    names = _infer_objective_names(
                        fidelity=2, rpm=rpm, torque=torque, n_obj=len(result.F)
                    )
                if objective_names is None:
                    objective_names = names
                feats = np.asarray(
                    list(np.asarray(x, dtype=np.float64)) + [float(rpm), float(torque)],
                    dtype=np.float64,
                )
                targets = _collect_stack_targets(result, constraint_names=constraint_names)
                X_rows.append(feats)
                Y_rows.append(targets)
                record["ok"] = True
            except Exception as exc:
                failures += 1
                record["ok"] = False
                record["error"] = str(exc)
            _stable_jsonl_append(records_jsonl, record)
            if attempted >= 100 and failures / max(attempted, 1) > failure_limit:
                raise CampaignError(
                    "Stack overnight fail-fast tripped: "
                    f"failures={failures}, attempted={attempted}, limit={failure_limit:.3f}"
                )
    if len(X_rows) < int(cfg.get("min_success_rows", 800)):
        raise CampaignError(
            f"Stack overnight dataset insufficient: successes={len(X_rows)} required>={int(cfg.get('min_success_rows', 800))}"
        )
    if objective_names is None:
        raise CampaignError("Stack overnight dataset produced no successful objective rows")
    dataset_path = outdir / "stack_f2_dataset.npz"
    np.savez_compressed(
        dataset_path,
        X=np.asarray(X_rows, dtype=np.float64),
        Y=np.asarray(Y_rows, dtype=np.float64),
        feature_names=np.asarray(feature_names, dtype=object),
        objective_names=np.asarray(objective_names, dtype=object),
        constraint_names=np.asarray(constraint_names, dtype=object),
    )
    meta = {
        "dataset_path": str(dataset_path),
        "records_jsonl": str(records_jsonl),
        "n_attempted": int(attempted),
        "n_success": int(len(X_rows)),
        "n_failures": int(failures),
        "operating_points": [{"rpm": rpm, "torque": torque} for rpm, torque in op_points],
    }
    log_fn(
        "Built stack F2 dataset",
        attempted=attempted,
        success=len(X_rows),
        failures=failures,
        dataset=dataset_path,
    )
    return dataset_path, meta


def run_overnight_f2_nn_campaign(
    args: argparse.Namespace,
    *,
    log_fn,
) -> dict[str, Any]:
    profile = load_overnight_campaign_profile(getattr(args, "profile", ""))
    run_id = str(getattr(args, "run_id", "")).strip() or time.strftime("%Y%m%d_%H%M%S")
    outdir_root = Path(str(getattr(args, "outdir_root", "outputs/overnight_f2")))
    outdir = outdir_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    install_canonical = bool(getattr(args, "install_canonical", True))
    summary: dict[str, Any] = {
        "profile_id": str(profile.get("profile_id", "")),
        "profile_path": str(profile.get("_path", "")),
        "run_id": run_id,
        "outdir": str(outdir),
        "contract_version": str(CONTRACT_VERSION),
        "steps": {},
        "promotion": {"install_canonical": install_canonical},
    }

    openfoam_template = Path(
        str(getattr(args, "openfoam_template", "")).strip() or str(DEFAULT_OPENFOAM_TEMPLATE)
    )
    if not openfoam_template.exists():
        raise FileNotFoundError(f"OpenFOAM template directory not found: {openfoam_template}")
    calculix_template_raw = str(getattr(args, "calculix_template", "")).strip()
    if not calculix_template_raw:
        raise FileNotFoundError(
            "CalculiX template path is required for overnight F2 campaign. "
            "Provide --calculix-template to run the real DOE leg."
        )
    calculix_template = Path(calculix_template_raw)
    if not calculix_template.exists():
        raise FileNotFoundError(f"CalculiX template file not found: {calculix_template}")

    truth_bundle = build_truth_anchor_bundle(
        profile=profile,
        outdir=outdir / "truth_anchors",
        template_dir=openfoam_template,
        solver_cmd=str(getattr(args, "openfoam_solver", "rhoPimpleFoam")),
        docker_timeout_s=int(getattr(args, "openfoam_docker_timeout_s", 1800)),
        docker_image=str(getattr(args, "openfoam_docker_image", "")).strip() or None,
        log_fn=log_fn,
    )
    summary["steps"]["truth_anchors"] = truth_bundle

    openfoam_data, openfoam_data_meta = build_openfoam_training_dataset(
        profile=profile,
        outdir=outdir / "openfoam_doe",
        template_dir=openfoam_template,
        solver_cmd=str(getattr(args, "openfoam_solver", "rhoPimpleFoam")),
        docker_timeout_s=int(getattr(args, "openfoam_docker_timeout_s", 1800)),
        docker_image=str(getattr(args, "openfoam_docker_image", "")).strip() or None,
        log_fn=log_fn,
    )
    summary["steps"]["openfoam_dataset"] = openfoam_data_meta

    openfoam_stage_dir = outdir / "openfoam_artifact"
    openfoam_summary = train_openfoam_workflow(
        argparse.Namespace(
            data=str(openfoam_data),
            outdir=str(openfoam_stage_dir),
            seed=int(getattr(args, "seed", 42)),
            epochs=int(getattr(args, "openfoam_epochs", 120)),
            lr=float(getattr(args, "openfoam_lr", 1e-3)),
            hidden=str(getattr(args, "openfoam_hidden", "64,64")),
            weight_decay=float(getattr(args, "openfoam_weight_decay", 0.0)),
            name=str(getattr(args, "openfoam_name", "openfoam_breathing.pt")),
            data_provenance_kind="doe_generated",
            authoritative_for_strict_f2=True,
            anchor_manifest=str(truth_bundle["anchor_manifest_path"]),
            truth_source_summary=json.dumps(truth_bundle, sort_keys=True),
            authority_bundle_root=str(
                getattr(args, "openfoam_authority_bundle_root", outdir / "openfoam_authority")
            ),
            source_metadata_json=json.dumps(openfoam_data_meta, sort_keys=True),
            doe_template_path=str(openfoam_template),
            authority_run_id=run_id,
        )
    )
    summary["steps"]["openfoam_train"] = openfoam_summary
    if not bool((openfoam_summary.get("authority_bundle", {}) or {}).get("promotable", False)):
        raise CampaignError("OpenFOAM authority bundle is not promotable after overnight training")

    openfoam_artifact_path = Path(str(openfoam_summary.get("artifact_path", "")))
    validate_artifact_quality(
        openfoam_artifact_path, surrogate_kind="openfoam", validation_mode="strict"
    )
    if install_canonical:
        from larrak2.surrogate.openfoam_authority import promote_openfoam_artifact

        promote_result = promote_openfoam_artifact(
            staged_dir=Path(
                str((openfoam_summary.get("authority_bundle", {}) or {}).get("staged_dir", ""))
            ),
            canonical_dir=str(DEFAULT_OPENFOAM_NN_ARTIFACT.parent),
            backup_root=str(_repo_root() / "outputs/artifacts/surrogates/openfoam_nn/archive"),
        )
        anchor_install = _copy_with_backup(
            Path(str(truth_bundle["anchor_manifest_path"])),
            _repo_root() / DEFAULT_ANCHOR_MANIFEST,
            backup_root=outdir / "backups" / "anchors",
        )
        summary["promotion"]["openfoam"] = promote_result
        summary["promotion"]["anchor_manifest"] = anchor_install
    openfoam_runtime_path = str(
        DEFAULT_OPENFOAM_NN_ARTIFACT if install_canonical else openfoam_artifact_path
    )

    calculix_data, calculix_data_meta = build_calculix_training_dataset(
        profile=profile,
        outdir=outdir / "calculix_doe",
        template_path=calculix_template,
        solver_cmd=str(getattr(args, "calculix_solver", "ccx")),
        log_fn=log_fn,
    )
    summary["steps"]["calculix_dataset"] = calculix_data_meta

    calculix_stage_dir = outdir / "calculix_artifact"
    train_calculix_workflow(
        argparse.Namespace(
            data=str(calculix_data),
            outdir=str(calculix_stage_dir),
            seed=int(getattr(args, "seed", 42)),
            epochs=int(getattr(args, "calculix_epochs", 120)),
            lr=float(getattr(args, "calculix_lr", 1e-3)),
            hidden=str(getattr(args, "calculix_hidden", "64,64")),
            weight_decay=float(getattr(args, "calculix_weight_decay", 0.0)),
            name=str(getattr(args, "calculix_name", "calculix_stress.pt")),
        )
    )
    calculix_artifact_path = calculix_stage_dir / str(
        getattr(args, "calculix_name", "calculix_stress.pt")
    )
    validate_artifact_quality(
        calculix_artifact_path, surrogate_kind="calculix", validation_mode="strict"
    )
    summary["steps"]["calculix_train"] = {
        "artifact_path": str(calculix_artifact_path),
        "quality_report": str(calculix_stage_dir / "quality_report.json"),
    }
    if install_canonical:
        calc_backup = _copy_with_backup(
            calculix_artifact_path,
            _repo_root() / DEFAULT_CALCULIX_NN_ARTIFACT,
            backup_root=outdir / "backups" / "calculix",
        )
        calc_report_backup = _copy_with_backup(
            calculix_stage_dir / "quality_report.json",
            _repo_root() / DEFAULT_CALCULIX_NN_ARTIFACT.parent / "quality_report.json",
            backup_root=outdir / "backups" / "calculix",
        )
        summary["promotion"]["calculix"] = {
            "artifact": calc_backup,
            "quality_report": calc_report_backup,
        }
    calculix_runtime_path = str(
        DEFAULT_CALCULIX_NN_ARTIFACT if install_canonical else calculix_artifact_path
    )

    stack_dataset_path, stack_dataset_meta = build_stack_dataset(
        profile=profile,
        outdir=outdir / "stack_dataset",
        openfoam_model_path=openfoam_runtime_path,
        calculix_model_path=calculix_runtime_path,
        anchor_manifest_path=(
            str(DEFAULT_ANCHOR_MANIFEST)
            if install_canonical
            else str(truth_bundle["anchor_manifest_path"])
        ),
        log_fn=log_fn,
    )
    summary["steps"]["stack_dataset"] = stack_dataset_meta

    stack_stage_dir = outdir / "stack_artifact"
    stack_summary = train_stack_surrogate_workflow(
        argparse.Namespace(
            outdir=str(stack_stage_dir),
            name=str(getattr(args, "stack_name", "stack_f2_surrogate.npz")),
            dataset=str(stack_dataset_path),
            pareto_dir="",
            fidelity=2,
            rpm=2300.0,
            torque=120.0,
            hidden=str(getattr(args, "stack_hidden", "128,128")),
            activation=str(getattr(args, "stack_activation", "relu")),
            leaky_relu_slope=float(getattr(args, "stack_leaky_relu_slope", 0.01)),
            epochs=int(getattr(args, "stack_epochs", 200)),
            lr=float(getattr(args, "stack_lr", 1e-3)),
            weight_decay=float(getattr(args, "stack_weight_decay", 1e-6)),
            val_frac=float((profile.get("stack", {}).get("split", {}) or {}).get("val_frac", 0.15)),
            test_frac=float(
                (profile.get("stack", {}).get("split", {}) or {}).get("test_frac", 0.15)
            ),
            seed=int(getattr(args, "seed", 42)),
        )
    )
    stack_artifact_path = Path(str(stack_summary.get("artifact_path", "")))
    validate_artifact_quality(stack_artifact_path, surrogate_kind="stack", validation_mode="strict")
    summary["steps"]["stack_train"] = stack_summary
    if install_canonical:
        stack_target = _repo_root() / stack_artifact_path_for_fidelity(2)
        stack_backup = _copy_with_backup(
            stack_artifact_path,
            stack_target,
            backup_root=outdir / "backups" / "stack",
        )
        stack_report_backup = _copy_with_backup(
            stack_artifact_path.parent / "quality_report.json",
            stack_target.parent / "quality_report.json",
            backup_root=outdir / "backups" / "stack",
        )
        summary["promotion"]["stack"] = {
            "artifact": stack_backup,
            "quality_report": stack_report_backup,
        }

    summary["status"] = "ok"
    return summary


__all__ = [
    "CampaignError",
    "DEFAULT_PROFILE_PATH",
    "build_calculix_training_dataset",
    "build_openfoam_training_dataset",
    "build_stack_dataset",
    "build_truth_anchor_bundle",
    "load_overnight_campaign_profile",
    "run_overnight_f2_nn_campaign",
]
