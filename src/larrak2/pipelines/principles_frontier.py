"""Reduced-order principles region synthesis for explore-exploit workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.core.archive_io import save_archive
from larrak2.core.constraints import get_constraint_names
from larrak2.core.encoding import N_TOTAL
from larrak2.core.types import EvalContext
from larrak2.optimization.candidate_store import CandidateStore
from larrak2.pipelines.principles_core import (
    PRINCIPLES_OBJECTIVE_NAMES,
    objective_scales_from_profile,
    reduced_release_stages,
    reduced_seed_states,
    reduced_variable_names,
    weight_vectors_from_profile,
)
from larrak2.pipelines.principles_search import PrinciplesDiagnosis, search_principles_region


@dataclass(frozen=True)
class OperatingPoint:
    rpm: float
    torque: float
    source: str


@dataclass(frozen=True)
class PrinciplesFrontierResult:
    store: CandidateStore
    pareto_source: Path
    profile_name: str
    profile_path: Path
    profile_payload: dict[str, Any]
    gate: dict[str, Any]
    artifacts: dict[str, Any]
    region_summary: dict[str, Any]
    proxy_vs_canonical: dict[str, Any]
    diagnosis: dict[str, Any]


_CANONICAL_V2_FIELDS = {
    "blend_weights",
    "canonical_alignment",
    "reduced_core",
    "expansion_policy",
    "normalization_scales",
    "weight_vectors",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_profile_path(profile_name: str, repo_root: Path) -> Path:
    norm = str(profile_name).strip().lower()
    if norm == "iso_litvin_v1":
        return repo_root / "data" / "optimization" / "principles_frontier_profile_v1.json"
    if norm == "iso_litvin_v2":
        return repo_root / "data" / "optimization" / "principles_frontier_profile_v2.json"
    raw = Path(str(profile_name))
    if raw.is_absolute():
        return raw
    return (repo_root / raw).resolve()


def _validate_base_profile(payload: dict[str, Any]) -> None:
    required = ("profile_id", "anchor_manifest", "envelope", "gate_thresholds", "source_references")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Principles profile missing required keys: {missing}")
    envelope = payload.get("envelope", {})
    for key in ("rpm_min", "rpm_max", "torque_min", "torque_max"):
        if key not in envelope:
            raise ValueError(f"Principles profile envelope missing '{key}'")
        _ = float(envelope[key])
    refs = payload.get("source_references", [])
    if not isinstance(refs, list) or not refs:
        raise ValueError("Principles profile requires non-empty 'source_references'")


def _upgrade_legacy_profile_if_needed(
    path: Path,
    payload: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if _CANONICAL_V2_FIELDS.issubset(payload.keys()):
        return payload
    canonical_v2 = _read_json(repo_root / "data" / "optimization" / "principles_frontier_profile_v2.json")
    upgraded = dict(canonical_v2)
    upgraded.update({
        "profile_id": str(payload.get("profile_id", upgraded.get("profile_id", "iso_litvin_v2"))),
        "profile_version": str(payload.get("profile_version", payload.get("profile_id", "1.0"))),
        "description": str(payload.get("description", upgraded.get("description", ""))),
        "anchor_manifest": str(payload.get("anchor_manifest", upgraded.get("anchor_manifest", ""))),
        "envelope": dict(payload.get("envelope", upgraded.get("envelope", {}))),
        "gate_thresholds": {
            **dict(upgraded.get("gate_thresholds", {})),
            **dict(payload.get("gate_thresholds", {})),
        },
        "source_references": list(payload.get("source_references", upgraded.get("source_references", []))),
        "compatibility_profile_source": str(path),
    })
    return upgraded


def load_principles_profile(
    profile_name: str,
    *,
    repo_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = _repo_root() if repo_root is None else Path(repo_root)
    path = _resolve_profile_path(profile_name, root)
    if not path.exists():
        raise FileNotFoundError(f"Principles profile not found: {path}")
    payload = _read_json(path)
    _validate_base_profile(payload)
    upgraded = _upgrade_legacy_profile_if_needed(path, payload, repo_root=root)
    reduced_variable_names(upgraded)
    reduced_release_stages(upgraded)
    reduced_seed_states(upgraded)
    objective_scales_from_profile(upgraded)
    weight_vectors_from_profile(upgraded)
    return path, upgraded


def _load_anchor_manifest(
    profile_payload: dict[str, Any], repo_root: Path
) -> tuple[Path, dict[str, Any]]:
    raw = Path(str(profile_payload.get("anchor_manifest", "")))
    path = raw if raw.is_absolute() else (repo_root / raw)
    if not path.exists():
        raise FileNotFoundError(f"Anchor manifest not found: {path}")
    payload = _read_json(path)
    anchors = payload.get("anchors", [])
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("Anchor manifest must contain non-empty 'anchors'")
    for i, rec in enumerate(anchors):
        rpm = float(rec.get("rpm", 0.0))
        torque = float(rec.get("torque", -1.0))
        if rpm <= 0.0 or torque < 0.0:
            raise ValueError(f"Invalid anchor[{i}] operating point: rpm={rpm}, torque={torque}")
    return path, payload


def _build_operating_points(
    *,
    profile_payload: dict[str, Any],
    anchor_manifest: dict[str, Any],
) -> list[OperatingPoint]:
    env = dict(profile_payload.get("envelope", {}) or {})
    rpm_min = float(env.get("rpm_min", 1000.0))
    rpm_max = float(env.get("rpm_max", rpm_min))
    torque_min = float(env.get("torque_min", 40.0))
    torque_max = float(env.get("torque_max", torque_min))
    rpm_mid = 0.5 * (rpm_min + rpm_max)
    torque_mid = 0.5 * (torque_min + torque_max)

    points: list[OperatingPoint] = []
    for rec in anchor_manifest.get("anchors", []):
        points.append(
            OperatingPoint(
                rpm=float(rec.get("rpm", rpm_mid)),
                torque=float(rec.get("torque", torque_mid)),
                source=str(rec.get("label", "anchor")),
            )
        )
    points.extend(
        [
            OperatingPoint(rpm=rpm_min, torque=torque_min, source="envelope_corner"),
            OperatingPoint(rpm=rpm_min, torque=torque_max, source="envelope_corner"),
            OperatingPoint(rpm=rpm_max, torque=torque_min, source="envelope_corner"),
            OperatingPoint(rpm=rpm_max, torque=torque_max, source="envelope_corner"),
            OperatingPoint(rpm=rpm_mid, torque=torque_mid, source="envelope_center"),
        ]
    )

    dedup: dict[tuple[float, float], OperatingPoint] = {}
    for point in points:
        key = (round(float(point.rpm), 9), round(float(point.torque), 9))
        dedup.setdefault(key, point)
    return list(dedup.values())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonify(row), sort_keys=True) + "\n")


def _legacy_gate_payload(
    *,
    ctx: EvalContext,
    diagnosis: PrinciplesDiagnosis,
    region_summary: dict[str, Any],
    allow_nonproduction_paths: bool,
    source_region_pass: bool,
) -> dict[str, Any]:
    gate_basis = "region_ready"
    if int(ctx.fidelity) == 0:
        gate_basis = "f0_diagnostic_non_blocking"
    return {
        "profile_name": str(diagnosis.metrics.get("profile_name", "")),
        "gate_basis": gate_basis,
        "frontier_gate_pass": bool(source_region_pass),
        "diagnosis_classification": str(diagnosis.classification),
        "n_hard_feasible_explore": int(region_summary.get("n_proxy_feasible", 0)),
        "n_nondominated": int(region_summary.get("n_region_candidates", 0)),
        "n_gate_feasible_explore": int(region_summary.get("n_region_candidates", 0)),
        "n_gate_nondominated": int(region_summary.get("n_region_candidates", 0)),
        "coverage_rpm_fraction": float(region_summary.get("envelope_coverage_rpm_fraction", 0.0)),
        "coverage_torque_fraction": float(region_summary.get("envelope_coverage_torque_fraction", 0.0)),
        "min_frontier_size_required": int(region_summary.get("region_min_size_required", 0)),
        "frontier_gate_required": bool(int(ctx.fidelity) >= 1),
        "allow_nonproduction_paths": bool(allow_nonproduction_paths),
    }


def synthesize_principles_frontier(
    *,
    outdir: str | Path,
    ctx: EvalContext,
    profile_name: str = "iso_litvin_v2",
    seed: int = 42,
    seed_count: int = 64,
    min_frontier_size: int = 12,
    root_max_iter: int = 80,
    export_archive_dir: str | Path | None = None,
    contract_version: str = "",
    allow_nonproduction_paths: bool = False,
    alignment_mode: str = "blend",
    alignment_fidelity: int = 1,
) -> PrinciplesFrontierResult:
    out_root = Path(outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    repo_root = _repo_root()

    profile_path, profile_payload = load_principles_profile(profile_name, repo_root=repo_root)
    anchor_path, anchor_manifest = _load_anchor_manifest(profile_payload, repo_root)
    operating_points = _build_operating_points(
        profile_payload=profile_payload,
        anchor_manifest=anchor_manifest,
    )
    if not operating_points:
        raise RuntimeError("Principles region synthesis requires at least one operating point.")

    search_result = search_principles_region(
        profile_payload=profile_payload,
        operating_points=[{"rpm": op.rpm, "torque": op.torque, "source": op.source} for op in operating_points],
        base_ctx=ctx,
        alignment_mode=str(alignment_mode),
        alignment_fidelity=int(alignment_fidelity),
        alignment_phase="explore",
        stage_max_iter=int(root_max_iter),
        region_min_size=int(max(1, min_frontier_size)),
    )
    records = list(search_result["records"])
    region_indices = [int(v) for v in search_result["region_indices"]]
    region_summary = dict(search_result["region_summary"])
    proxy_vs_canonical = dict(search_result["proxy_vs_canonical"])
    diagnosis = search_result["diagnosis"]

    thresholds = dict(profile_payload.get("gate_thresholds", {}) or {})
    coverage_ok = (
        float(region_summary.get("envelope_coverage_rpm_fraction", 0.0))
        >= float(thresholds.get("coverage_rpm_fraction_min", 0.6))
        and float(region_summary.get("envelope_coverage_torque_fraction", 0.0))
        >= float(thresholds.get("coverage_torque_fraction_min", 0.6))
    )
    source_region_pass = bool(diagnosis.classification == "region_ready" and coverage_ok)
    source_region_policy = "strict_region_ready"
    if int(ctx.fidelity) == 0 and diagnosis.classification != "misconfiguration_or_data_gap":
        source_region_pass = True
        source_region_policy = "diagnostic_non_blocking_f0"

    n_constraints = len(get_constraint_names(max(int(ctx.fidelity), int(alignment_fidelity), 1)))
    X_region = np.asarray([records[i]["x_full"] for i in region_indices], dtype=np.float64) if region_indices else np.zeros((0, N_TOTAL), dtype=np.float64)
    F_region = np.asarray([records[i]["F_blend"] for i in region_indices], dtype=np.float64) if region_indices else np.zeros((0, len(PRINCIPLES_OBJECTIVE_NAMES)), dtype=np.float64)
    G_region = np.asarray([records[i]["G_combined"] for i in region_indices], dtype=np.float64) if region_indices else np.zeros((0, n_constraints), dtype=np.float64)
    Z_region = np.asarray([records[i]["z_reduced"] for i in region_indices], dtype=np.float64) if region_indices else np.zeros((0, len(reduced_variable_names(profile_payload))), dtype=np.float64)
    proxy_F = np.asarray([records[i]["proxy"].F for i in region_indices], dtype=np.float64) if region_indices else np.zeros((0, len(PRINCIPLES_OBJECTIVE_NAMES)), dtype=np.float64)
    proxy_G = np.asarray([records[i]["proxy"].G for i in region_indices], dtype=np.float64) if region_indices else np.zeros((0, n_constraints), dtype=np.float64)
    align_F = np.asarray([records[i]["alignment"].F for i in region_indices], dtype=np.float64) if region_indices else np.zeros((0, len(PRINCIPLES_OBJECTIVE_NAMES)), dtype=np.float64)
    align_G = np.asarray([records[i]["alignment"].G for i in region_indices], dtype=np.float64) if region_indices else np.zeros((0, n_constraints), dtype=np.float64)

    diagnosis_payload = {
        "classification": str(diagnosis.classification),
        "metrics": _jsonify(diagnosis.metrics),
        "source_region_pass": bool(source_region_pass),
        "source_region_policy": str(source_region_policy),
    }
    region_summary_payload = {
        **region_summary,
        "profile_name": str(profile_name),
        "profile_path": str(profile_path),
        "anchor_manifest_path": str(anchor_path),
        "contract_version": str(contract_version or ""),
        "seed_count_input": int(seed_count),
        "stage_max_iter_input": int(root_max_iter),
        "source_region_pass": bool(source_region_pass),
        "source_region_policy": str(source_region_policy),
        "diagnosis_classification": str(diagnosis.classification),
        "source_references": list(profile_payload.get("source_references", [])),
        "reduced_core": {
            "variable_names": list(reduced_variable_names(profile_payload)),
            "release_stages": list(reduced_release_stages(profile_payload)),
            "seed_states": {k: v.tolist() for k, v in reduced_seed_states(profile_payload).items()},
        },
        "expansion_policy": dict(profile_payload.get("expansion_policy", {})),
        "blend_weights": dict(profile_payload.get("blend_weights", {})),
        "canonical_alignment": dict(profile_payload.get("canonical_alignment", {})),
        "normalization_scales": dict(profile_payload.get("normalization_scales", {})),
    }
    proxy_vs_canonical_payload = {
        **proxy_vs_canonical,
        "source_region_pass": bool(source_region_pass),
        "diagnosis_classification": str(diagnosis.classification),
    }
    legacy_gate_payload = _legacy_gate_payload(
        ctx=ctx,
        diagnosis=diagnosis,
        region_summary=region_summary_payload,
        allow_nonproduction_paths=allow_nonproduction_paths,
        source_region_pass=source_region_pass,
    )

    for idx, rec in enumerate(records):
        rec["region_candidate"] = bool(idx in set(region_indices))
        rec["proxy"] = {
            "F": np.asarray(rec["proxy"].F, dtype=np.float64).tolist(),
            "G": np.asarray(rec["proxy"].G, dtype=np.float64).tolist(),
            "diag": _jsonify(rec["proxy"].diag),
            "objective_names": list(rec["proxy"].objective_names),
            "constraint_names": list(rec["proxy"].constraint_names),
            "expansion_policy": _jsonify(rec["proxy"].expansion_policy),
            "error_signature": str(rec["proxy"].error_signature),
        }
        rec["alignment"] = {
            "F": np.asarray(rec["alignment"].F, dtype=np.float64).tolist(),
            "G": np.asarray(rec["alignment"].G, dtype=np.float64).tolist(),
            "diag": _jsonify(rec["alignment"].diag),
            "objective_names": list(rec["alignment"].objective_names),
            "constraint_names": list(rec["alignment"].constraint_names),
            "error_signature": str(rec["alignment"].error_signature),
        }

    np.save(out_root / "principles_reduced_Z.npy", Z_region)
    np.save(out_root / "principles_expanded_X.npy", X_region)
    np.save(out_root / "principles_proxy_F.npy", proxy_F)
    np.save(out_root / "principles_proxy_G.npy", proxy_G)
    np.save(out_root / "principles_alignment_F.npy", align_F)
    np.save(out_root / "principles_alignment_G.npy", align_G)
    np.save(out_root / "principles_frontier_X.npy", X_region)
    np.save(out_root / "principles_frontier_F.npy", F_region)
    np.save(out_root / "principles_frontier_G.npy", G_region)

    region_summary_path = out_root / "principles_region_summary.json"
    proxy_vs_canonical_path = out_root / "principles_proxy_vs_canonical.json"
    diagnosis_path = out_root / "principles_diagnosis.json"
    candidate_records_path = out_root / "principles_candidate_records.jsonl"
    legacy_gate_path = out_root / "principles_frontier_gate.json"
    legacy_summary_path = out_root / "principles_frontier_summary.json"
    _write_json(region_summary_path, region_summary_payload)
    _write_json(proxy_vs_canonical_path, proxy_vs_canonical_payload)
    _write_json(diagnosis_path, diagnosis_payload)
    _write_json(legacy_gate_path, legacy_gate_payload)
    _write_json(legacy_summary_path, {**region_summary_payload, "legacy_gate": legacy_gate_payload})
    _write_jsonl(candidate_records_path, records)

    archive_dir = Path(export_archive_dir) if export_archive_dir is not None else out_root / "principles_pareto"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_summary = {
        "n_pareto": int(X_region.shape[0]),
        "n_evals": int(len(records)),
        "rpm": float(ctx.rpm),
        "torque": float(ctx.torque),
        "fidelity": int(ctx.fidelity),
        "seed": int(seed),
        "objective_names": list(PRINCIPLES_OBJECTIVE_NAMES),
        "profile_name": str(profile_name),
        "profile_path": str(profile_path),
        "source_region_pass": bool(source_region_pass),
        "diagnosis_classification": str(diagnosis.classification),
    }
    save_archive(archive_dir, X_region, F_region, G_region, archive_summary)
    store = CandidateStore.from_arrays(
        X=X_region,
        F=F_region,
        G=G_region,
        summary=archive_summary,
        source_dir=archive_dir,
    )

    artifacts = {
        "principles_reduced_Z": str(out_root / "principles_reduced_Z.npy"),
        "principles_expanded_X": str(out_root / "principles_expanded_X.npy"),
        "principles_proxy_F": str(out_root / "principles_proxy_F.npy"),
        "principles_proxy_G": str(out_root / "principles_proxy_G.npy"),
        "principles_alignment_F": str(out_root / "principles_alignment_F.npy"),
        "principles_alignment_G": str(out_root / "principles_alignment_G.npy"),
        "principles_region_summary": str(region_summary_path),
        "principles_proxy_vs_canonical": str(proxy_vs_canonical_path),
        "principles_diagnosis": str(diagnosis_path),
        "principles_candidate_records": str(candidate_records_path),
        "principles_frontier_X": str(out_root / "principles_frontier_X.npy"),
        "principles_frontier_F": str(out_root / "principles_frontier_F.npy"),
        "principles_frontier_G": str(out_root / "principles_frontier_G.npy"),
        "principles_frontier_summary": str(legacy_summary_path),
        "principles_frontier_gate": str(legacy_gate_path),
        "principles_export_archive_dir": str(archive_dir),
    }

    return PrinciplesFrontierResult(
        store=store,
        pareto_source=archive_dir,
        profile_name=str(profile_name),
        profile_path=profile_path,
        profile_payload=profile_payload,
        gate=legacy_gate_payload,
        artifacts=artifacts,
        region_summary=region_summary_payload,
        proxy_vs_canonical=proxy_vs_canonical_payload,
        diagnosis=diagnosis_payload,
    )


__all__ = [
    "OperatingPoint",
    "PrinciplesFrontierResult",
    "load_principles_profile",
    "synthesize_principles_frontier",
]
