"""Authority staging and promotion helpers for OpenFOAM surrogate artifacts."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from larrak2.core.artifact_paths import (
    DEFAULT_OPENFOAM_NN_ARTIFACT,
    DEFAULT_OPENFOAM_NN_DIR,
    OUTPUTS_ROOT,
)
from larrak2.surrogate.quality_contract import (
    load_quality_report,
    openfoam_quality_fail_reasons,
    openfoam_strict_f2_provenance_status,
    sha256_file,
)

DEFAULT_OPENFOAM_AUTHORITY_ROOT = OUTPUTS_ROOT / "openfoam_authority"
DEFAULT_OPENFOAM_PROMOTION_ARCHIVE_DIR = DEFAULT_OPENFOAM_NN_DIR / "archive"


def _utc_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _sha_dir(dir_path: Path) -> str:
    payload: list[str] = []
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            payload.append(f"{p.relative_to(dir_path)}:{p.stat().st_size}")
    import hashlib

    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def inspect_truth_anchor_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    out: dict[str, Any] = {
        "path": str(manifest_path),
        "exists": manifest_path.exists(),
        "truth_backed": False,
        "version": "",
        "anchor_count": 0,
        "source_type": "",
        "input_files": [],
        "reasons": [],
    }
    if not manifest_path.exists():
        out["reasons"] = ["anchor_manifest_missing"]
        return out

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["reasons"] = [f"anchor_manifest_invalid_json:{exc}"]
        return out
    if not isinstance(payload, dict):
        out["reasons"] = ["anchor_manifest_not_object"]
        return out

    anchors = payload.get("anchors", [])
    provenance = payload.get("provenance", {})
    input_files = provenance.get("input_files", []) if isinstance(provenance, dict) else []
    source_type = (
        str(provenance.get("source_type", "")).strip() if isinstance(provenance, dict) else ""
    )

    out["version"] = str(payload.get("version", "")).strip()
    out["anchor_count"] = int(len(anchors) if isinstance(anchors, list) else 0)
    out["source_type"] = source_type
    out["input_files"] = [str(x) for x in input_files] if isinstance(input_files, list) else []

    reasons: list[str] = []
    if not isinstance(anchors, list) or not anchors:
        reasons.append("anchor_manifest_empty")
    if source_type != "truth_runs":
        reasons.append("anchor_manifest_not_truth_backed")
    if not isinstance(input_files, list) or not input_files:
        reasons.append("anchor_manifest_missing_truth_inputs")

    out["truth_backed"] = not reasons
    out["reasons"] = reasons
    return out


def build_openfoam_dataset_manifest(
    *,
    data_path: str | Path,
    source_meta: dict[str, Any] | None = None,
    template_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = Path(data_path)
    manifest: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.exists(),
        "dataset_sha256": sha256_file(dataset_path)
        if dataset_path.exists() and dataset_path.is_file()
        else "",
        "source_meta": dict(source_meta or {}),
        "template_path": "",
        "template_exists": False,
        "template_hash": "",
        "doe_success_counts": {
            "n_total_cases": int((source_meta or {}).get("n_total_cases", 0) or 0),
            "n_success_cases": int((source_meta or {}).get("n_success_cases", 0) or 0),
        },
    }
    if template_path is not None and str(template_path).strip():
        template_dir = Path(template_path)
        manifest["template_path"] = str(template_dir)
        manifest["template_exists"] = template_dir.exists()
        manifest["template_hash"] = (
            _sha_dir(template_dir) if template_dir.exists() and template_dir.is_dir() else ""
        )
    return manifest


def build_openfoam_split_manifest(
    *,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    seed: int,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "train_indices": [int(i) for i in train_idx],
        "val_indices": [int(i) for i in val_idx],
        "test_indices": [int(i) for i in test_idx],
    }


def _required_bundle_files(staged_dir: Path, artifact_file: str) -> dict[str, str]:
    return {
        "artifact": str(staged_dir / artifact_file),
        "quality_report": str(staged_dir / "quality_report.json"),
        "dataset_manifest": str(staged_dir / "dataset_manifest.json"),
        "split_manifest": str(staged_dir / "split_manifest.json"),
    }


def build_openfoam_authority_validation_report(
    *,
    staged_dir: str | Path,
    quality_report: dict[str, Any],
    dataset_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    staged_path = Path(staged_dir)
    artifact_file = str(quality_report.get("artifact_file", "")).strip()
    required_files = _required_bundle_files(staged_path, artifact_file)
    missing_files = [path for path in required_files.values() if not Path(path).exists()]

    provenance_status = openfoam_strict_f2_provenance_status(quality_report)
    quality_reasons = openfoam_quality_fail_reasons(quality_report)
    anchor_manifest_path = str(
        ((provenance_status.get("data_provenance") or {}).get("anchor_manifest_path", "")).strip()
    )
    anchor_status = (
        inspect_truth_anchor_manifest(anchor_manifest_path)
        if anchor_manifest_path
        else {
            "path": "",
            "exists": False,
            "truth_backed": False,
            "version": "",
            "anchor_count": 0,
            "source_type": "",
            "input_files": [],
            "reasons": ["anchor_manifest_missing"],
        }
    )

    reasons: list[str] = []
    reasons.extend([f"missing_bundle_file:{Path(p).name}" for p in missing_files])
    if not bool(dataset_manifest.get("dataset_exists", False)):
        reasons.append("dataset_missing")
    if str(dataset_manifest.get("source_meta", {}).get("source", "")).strip() != "doe_generated":
        reasons.append("dataset_not_doe_generated")
    if not bool(dataset_manifest.get("template_exists", False)):
        reasons.append("template_missing")
    if int(dataset_manifest.get("doe_success_counts", {}).get("n_success_cases", 0)) <= 0:
        reasons.append("doe_success_cases_missing")
    reasons.extend([f"quality:{r}" for r in quality_reasons])
    if not bool(provenance_status.get("strict_f2_eligible", False)):
        reasons.append(f"provenance:{provenance_status.get('gate_failure_reason', 'invalid')}")
    reasons.extend([f"anchor:{r}" for r in anchor_status.get("reasons", [])])

    report = {
        "staged_dir": str(staged_path),
        "artifact_file": artifact_file,
        "required_files": required_files,
        "missing_files": missing_files,
        "dataset_manifest": dataset_manifest,
        "split_manifest_summary": {
            "train_count": int(split_manifest.get("train_count", 0)),
            "val_count": int(split_manifest.get("val_count", 0)),
            "test_count": int(split_manifest.get("test_count", 0)),
            "seed": int(split_manifest.get("seed", 0)),
        },
        "quality_profile_pass": len(quality_reasons) == 0,
        "quality_fail_reasons": quality_reasons,
        "provenance_status": provenance_status,
        "truth_anchor_status": anchor_status,
        "promotable": len(reasons) == 0,
        "failure_reasons": reasons,
        "artifact_sha256": str(quality_report.get("artifact_sha256", "")).strip(),
    }
    return report


def write_staged_openfoam_authority_bundle(
    *,
    artifact_path: str | Path,
    quality_report_path: str | Path,
    bundle_root: str | Path = DEFAULT_OPENFOAM_AUTHORITY_ROOT,
    data_path: str | Path,
    source_meta: dict[str, Any] | None,
    template_path: str | Path | None,
    split_manifest: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    artifact = Path(artifact_path)
    quality_report = load_quality_report(quality_report_path)
    bundle_root_path = Path(bundle_root)
    run_token = str(run_id).strip() if run_id else f"{_utc_stamp()}_{artifact.stem}"
    staged_dir = bundle_root_path / run_token
    staged_dir.mkdir(parents=True, exist_ok=True)

    staged_artifact = staged_dir / artifact.name
    shutil.copy2(artifact, staged_artifact)
    shutil.copy2(quality_report_path, staged_dir / "quality_report.json")

    dataset_manifest = build_openfoam_dataset_manifest(
        data_path=data_path,
        source_meta=source_meta,
        template_path=template_path,
    )
    (staged_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (staged_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    authority_validation = build_openfoam_authority_validation_report(
        staged_dir=staged_dir,
        quality_report=quality_report,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
    )
    (staged_dir / "authority_validation_report.json").write_text(
        json.dumps(authority_validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    promotion_manifest = {
        "status": "staged",
        "created_at_utc": _utc_stamp(),
        "staged_dir": str(staged_dir),
        "canonical_target": str(DEFAULT_OPENFOAM_NN_ARTIFACT),
        "artifact_file": artifact.name,
        "promotable": bool(authority_validation["promotable"]),
    }
    (staged_dir / "promotion_manifest.json").write_text(
        json.dumps(promotion_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "staged_dir": str(staged_dir),
        "artifact_path": str(staged_artifact),
        "quality_report_path": str(staged_dir / "quality_report.json"),
        "dataset_manifest_path": str(staged_dir / "dataset_manifest.json"),
        "split_manifest_path": str(staged_dir / "split_manifest.json"),
        "authority_validation_report_path": str(staged_dir / "authority_validation_report.json"),
        "promotion_manifest_path": str(staged_dir / "promotion_manifest.json"),
        "promotable": bool(authority_validation["promotable"]),
        "failure_reasons": list(authority_validation["failure_reasons"]),
    }


def resolve_latest_openfoam_authority_dir(
    bundle_root: str | Path = DEFAULT_OPENFOAM_AUTHORITY_ROOT,
) -> Path:
    root = Path(bundle_root)
    candidates = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No staged OpenFOAM authority bundles found under '{root}'")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def validate_staged_openfoam_authority_bundle(staged_dir: str | Path) -> dict[str, Any]:
    staged_path = Path(staged_dir)
    quality_report = load_quality_report(staged_path / "quality_report.json")
    dataset_manifest = json.loads(
        (staged_path / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads((staged_path / "split_manifest.json").read_text(encoding="utf-8"))
    report = build_openfoam_authority_validation_report(
        staged_dir=staged_path,
        quality_report=quality_report,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
    )
    (staged_path / "authority_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def promote_openfoam_artifact(
    *,
    staged_dir: str | Path,
    canonical_dir: str | Path = DEFAULT_OPENFOAM_NN_DIR,
    backup_root: str | Path = DEFAULT_OPENFOAM_PROMOTION_ARCHIVE_DIR,
) -> dict[str, Any]:
    staged_path = Path(staged_dir)
    validation = validate_staged_openfoam_authority_bundle(staged_path)
    if not bool(validation.get("promotable", False)):
        raise ValueError(
            "OpenFOAM authority bundle is not promotable: "
            + "; ".join(str(x) for x in validation.get("failure_reasons", []))
        )

    report = load_quality_report(staged_path / "quality_report.json")
    artifact_file = str(report.get("artifact_file", "")).strip()
    if not artifact_file:
        raise ValueError("Staged OpenFOAM quality report is missing artifact_file")
    staged_artifact = staged_path / artifact_file
    if not staged_artifact.exists():
        raise FileNotFoundError(f"Staged OpenFOAM artifact missing: '{staged_artifact}'")
    expected_hash = str(report.get("artifact_sha256", "")).strip()
    if expected_hash and sha256_file(staged_artifact) != expected_hash:
        raise ValueError("Staged OpenFOAM artifact hash does not match quality report")

    canonical_root = Path(canonical_dir)
    canonical_root.mkdir(parents=True, exist_ok=True)
    canonical_artifact = canonical_root / artifact_file
    canonical_report = canonical_root / "quality_report.json"

    backup_path = ""
    if canonical_artifact.exists() or canonical_report.exists():
        archive_root = Path(backup_root)
        archive_root.mkdir(parents=True, exist_ok=True)
        backup_dir = archive_root / f"{_utc_stamp()}_{staged_path.name}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        if canonical_artifact.exists():
            shutil.copy2(canonical_artifact, backup_dir / canonical_artifact.name)
        if canonical_report.exists():
            shutil.copy2(canonical_report, backup_dir / canonical_report.name)
        backup_path = str(backup_dir)

    shutil.copy2(staged_artifact, canonical_artifact)
    shutil.copy2(staged_path / "quality_report.json", canonical_report)

    promotion_manifest = {
        "status": "promoted",
        "promoted_at_utc": _utc_stamp(),
        "staged_dir": str(staged_path),
        "canonical_artifact": str(canonical_artifact),
        "canonical_quality_report": str(canonical_report),
        "backup_path": backup_path,
        "authority_validation_report": str(staged_path / "authority_validation_report.json"),
    }
    (staged_path / "promotion_manifest.json").write_text(
        json.dumps(promotion_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (canonical_root / "promotion_manifest.json").write_text(
        json.dumps(promotion_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return promotion_manifest


__all__ = [
    "DEFAULT_OPENFOAM_AUTHORITY_ROOT",
    "build_openfoam_authority_validation_report",
    "build_openfoam_dataset_manifest",
    "build_openfoam_split_manifest",
    "inspect_truth_anchor_manifest",
    "promote_openfoam_artifact",
    "resolve_latest_openfoam_authority_dir",
    "validate_staged_openfoam_authority_bundle",
    "write_staged_openfoam_authority_bundle",
]
