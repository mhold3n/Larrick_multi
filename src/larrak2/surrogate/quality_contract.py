"""Surrogate quality-report contract schema and runtime validators."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

_ALLOWED_MODES = {"strict", "warn", "off"}
_ALLOWED_KINDS = {"stack", "openfoam", "calculix", "hifi", "thermo_symbolic"}
_OPENFOAM_PROVENANCE_KINDS = {"synthetic_rehearsal", "doe_generated", "truth_records"}
_THERMO_SYMBOLIC_BALANCED_PROFILE = {
    "profile": "balanced_v1",
    "normalization_method": "p95_p05_range",
    "val_nrmse_max": 0.20,
    "test_nrmse_max": 0.25,
    "min_r2": 0.40,
}
_OPENFOAM_QUALITY_PROFILE = {
    "profile": "openfoam_authority_v1",
    "normalization_method": "p95_p05_range",
    "val_nrmse_max": 0.30,
    "test_nrmse_max": 0.35,
    "min_r2": 0.25,
    "require_finite_slice_metrics": True,
}


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest_for_file(
    dataset_path: str | Path,
    *,
    n_samples: int,
    n_features: int,
    n_targets: int,
) -> dict[str, Any]:
    p = Path(dataset_path)
    payload: dict[str, Any] = {
        "source_path": str(p),
        "num_samples": int(n_samples),
        "num_features": int(n_features),
        "num_targets": int(n_targets),
    }
    if p.exists() and p.is_file():
        payload["dataset_sha256"] = sha256_file(p)
    else:
        payload["dataset_sha256"] = ""
    return payload


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    err = yp - yt
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err * err))
    yt_mean = float(np.mean(yt))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _target_scale_p95_p05(y_true: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if yt.size == 0:
        return float("nan")
    q95 = float(np.percentile(yt, 95))
    q05 = float(np.percentile(yt, 5))
    return float(max(q95 - q05, 1e-12))


def per_target_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: list[str] | tuple[str, ...],
) -> list[dict[str, float | str]]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim != 2 or yp.ndim != 2 or yt.shape != yp.shape:
        raise ValueError("per_target_regression_metrics expects matching 2D arrays")
    if yt.shape[1] != len(target_names):
        raise ValueError(
            f"target_names length mismatch: {len(target_names)} vs n_targets={yt.shape[1]}"
        )

    out: list[dict[str, float | str]] = []
    for i, name in enumerate(target_names):
        base = regression_metrics(yt[:, i], yp[:, i])
        scale = _target_scale_p95_p05(yt[:, i])
        rmse = float(base["rmse"])
        nrmse = float(rmse / scale) if np.isfinite(rmse) and np.isfinite(scale) else float("nan")
        out.append(
            {
                "name": str(name),
                "rmse": rmse,
                "mae": float(base["mae"]),
                "r2": float(base["r2"]),
                "nrmse": nrmse,
                "target_scale": float(scale),
            }
        )
    return out


def _raise_or_warn(mode: str, message: str, *, exc_type: type[Exception]) -> None:
    if mode == "strict":
        raise exc_type(message)
    if mode == "warn":
        LOGGER.warning(message)


def _report_path_for_artifact(artifact_path: Path) -> Path:
    if artifact_path.is_dir() or (not artifact_path.exists() and artifact_path.suffix == ""):
        return artifact_path / "quality_report.json"
    return artifact_path.parent / "quality_report.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def thermo_symbolic_balanced_profile() -> dict[str, Any]:
    """Return the canonical balanced quality profile for thermo symbolic artifacts."""
    return dict(_THERMO_SYMBOLIC_BALANCED_PROFILE)


def openfoam_quality_profile() -> dict[str, Any]:
    """Return the canonical promotion profile for OpenFOAM authority bundles."""
    return dict(_OPENFOAM_QUALITY_PROFILE)


def openfoam_default_data_provenance(
    *,
    source_path: str = "",
    kind: str = "synthetic_rehearsal",
    authoritative_for_strict_f2: bool = False,
    anchor_manifest_path: str = "",
    anchor_manifest_version: str = "",
    anchor_count: int = 0,
    truth_source_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": str(kind),
        "authoritative_for_strict_f2": bool(authoritative_for_strict_f2),
        "anchor_manifest_path": str(anchor_manifest_path),
        "anchor_manifest_version": str(anchor_manifest_version),
        "anchor_count": int(anchor_count),
        "truth_source_summary": dict(
            truth_source_summary
            or {
                "source_path": str(source_path),
            }
        ),
    }


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def openfoam_data_provenance(report: dict[str, Any]) -> dict[str, Any]:
    provenance = report.get("data_provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    dataset_manifest = report.get("dataset_manifest", {})
    source_path = ""
    if isinstance(dataset_manifest, dict):
        source_path = str(dataset_manifest.get("source_path", "")).strip()
    base = openfoam_default_data_provenance(source_path=source_path)
    base.update(
        {
            "kind": str(provenance.get("kind", base["kind"])).strip() or base["kind"],
            "authoritative_for_strict_f2": bool(
                provenance.get(
                    "authoritative_for_strict_f2",
                    base["authoritative_for_strict_f2"],
                )
            ),
            "anchor_manifest_path": str(
                provenance.get("anchor_manifest_path", base["anchor_manifest_path"])
            ).strip(),
            "anchor_manifest_version": str(
                provenance.get("anchor_manifest_version", base["anchor_manifest_version"])
            ).strip(),
            "anchor_count": int(provenance.get("anchor_count", base["anchor_count"]) or 0),
            "truth_source_summary": dict(
                provenance.get("truth_source_summary", base["truth_source_summary"]) or {}
            ),
        }
    )
    return base


def openfoam_strict_f2_provenance_status(report: dict[str, Any]) -> dict[str, Any]:
    provenance = openfoam_data_provenance(report)
    kind = str(provenance.get("kind", "")).strip()
    authoritative = bool(provenance.get("authoritative_for_strict_f2", False))
    anchor_count = int(provenance.get("anchor_count", 0))

    if kind not in _OPENFOAM_PROVENANCE_KINDS:
        gate_failure_reason = "invalid_openfoam_data_provenance_kind"
    elif kind == "synthetic_rehearsal":
        gate_failure_reason = "synthetic_artifact_not_allowed_in_strict_f2"
    elif not authoritative:
        gate_failure_reason = "openfoam_artifact_not_marked_authoritative_for_strict_f2"
    elif anchor_count <= 0:
        gate_failure_reason = "openfoam_artifact_missing_anchor_linkage_for_strict_f2"
    else:
        gate_failure_reason = ""

    strict_f2_eligible = gate_failure_reason == ""
    if strict_f2_eligible:
        benchmark_authority = "truth_like_authoritative"
    elif kind == "synthetic_rehearsal":
        benchmark_authority = "synthetic_non_authoritative"
    else:
        benchmark_authority = "non_authoritative"

    return {
        "data_provenance": provenance,
        "strict_f2_eligible": bool(strict_f2_eligible),
        "benchmark_authority": benchmark_authority,
        "gate_failure_reason": gate_failure_reason,
    }


def validate_openfoam_strict_f2_provenance(
    report: dict[str, Any],
    *,
    validation_mode: str = "strict",
) -> dict[str, Any]:
    mode = str(validation_mode).strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            f"validation_mode must be one of {sorted(_ALLOWED_MODES)}, got {validation_mode!r}"
        )
    status = openfoam_strict_f2_provenance_status(report)
    if mode == "off":
        return status
    if not bool(status["strict_f2_eligible"]):
        _raise_or_warn(
            mode,
            "openfoam strict F2 provenance gate failed: " + str(status["gate_failure_reason"]),
            exc_type=ValueError,
        )
    return status


def openfoam_quality_fail_reasons(report: dict[str, Any]) -> list[str]:
    """Return OpenFOAM promotion-quality failures from a quality report."""
    if str(report.get("surrogate_kind", "")).strip() != "openfoam":
        return []

    profile = dict(openfoam_quality_profile())
    profile.update(report.get("quality_profile", {}) or {})
    val_nrmse_max = _as_float(profile.get("val_nrmse_max"))
    test_nrmse_max = _as_float(profile.get("test_nrmse_max"))
    min_r2 = _as_float(profile.get("min_r2"))
    require_finite_slice_metrics = bool(profile.get("require_finite_slice_metrics", True))

    metrics = report.get("metrics", {}) if isinstance(report.get("metrics", {}), dict) else {}
    out: list[str] = []
    for split, nrmse_limit in (("val", val_nrmse_max), ("test", test_nrmse_max)):
        split_metrics = metrics.get(split, {}) if isinstance(metrics.get(split, {}), dict) else {}
        per_target = split_metrics.get("per_target", [])
        if not isinstance(per_target, list) or not per_target:
            out.append(f"{split}.per_target missing or empty")
            continue
        for rec in per_target:
            if not isinstance(rec, dict):
                out.append(f"{split}.per_target has non-object entry")
                continue
            name = str(rec.get("name", "")).strip() or "<unnamed>"
            nrmse = _as_float(rec.get("nrmse"))
            r2 = _as_float(rec.get("r2"))
            if not np.isfinite(nrmse):
                out.append(f"{split}:{name} nrmse is non-finite")
            elif np.isfinite(nrmse_limit) and nrmse > nrmse_limit:
                out.append(f"{split}:{name} nrmse={nrmse:.6f} exceeds {nrmse_limit:.6f}")
            if not np.isfinite(r2):
                out.append(f"{split}:{name} r2 is non-finite")
            elif np.isfinite(min_r2) and r2 < min_r2:
                out.append(f"{split}:{name} r2={r2:.6f} below {min_r2:.6f}")

    slice_metrics = metrics.get("slice_metrics", [])
    if require_finite_slice_metrics:
        if not isinstance(slice_metrics, list) or not slice_metrics:
            out.append("metrics.slice_metrics missing or empty")
        else:
            for i, rec in enumerate(slice_metrics):
                if not isinstance(rec, dict):
                    out.append(f"slice_metrics[{i}] is not an object")
                    continue
                name = str(rec.get("name", "")).strip() or f"slice_{i}"
                for field in ("rmse", "mae", "r2"):
                    value = _as_float(rec.get(field))
                    if not np.isfinite(value):
                        out.append(f"slice_metrics[{name}].{field} is non-finite")
    return out


def validate_openfoam_quality(
    report: dict[str, Any],
    *,
    validation_mode: str = "strict",
) -> list[str]:
    """Enforce OpenFOAM promotion-quality thresholds by mode."""
    mode = str(validation_mode).strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            f"validation_mode must be one of {sorted(_ALLOWED_MODES)}, got {validation_mode!r}"
        )
    if mode == "off":
        return []
    if str(report.get("surrogate_kind", "")).strip() != "openfoam":
        return []

    reasons = openfoam_quality_fail_reasons(report)
    if reasons:
        _raise_or_warn(
            mode,
            "openfoam quality gates failed: " + "; ".join(reasons),
            exc_type=ValueError,
        )
    return reasons


def thermo_symbolic_quality_fail_reasons(report: dict[str, Any]) -> list[str]:
    """Return target-specific quality gate failures for thermo symbolic reports."""
    if str(report.get("surrogate_kind", "")).strip() != "thermo_symbolic":
        return []

    profile = dict(thermo_symbolic_balanced_profile())
    profile.update(report.get("quality_profile", {}) or {})
    val_nrmse_max = _as_float(profile.get("val_nrmse_max"))
    test_nrmse_max = _as_float(profile.get("test_nrmse_max"))
    min_r2 = _as_float(profile.get("min_r2"))

    metrics = report.get("metrics", {}) if isinstance(report.get("metrics", {}), dict) else {}
    out: list[str] = []
    for split, nrmse_limit in (("val", val_nrmse_max), ("test", test_nrmse_max)):
        split_metrics = metrics.get(split, {}) if isinstance(metrics.get(split, {}), dict) else {}
        per_target = split_metrics.get("per_target", [])
        if not isinstance(per_target, list) or not per_target:
            out.append(f"{split}.per_target missing or empty")
            continue
        for rec in per_target:
            if not isinstance(rec, dict):
                out.append(f"{split}.per_target has non-object entry")
                continue
            name = str(rec.get("name", "")).strip() or "<unnamed>"
            nrmse = _as_float(rec.get("nrmse"))
            r2 = _as_float(rec.get("r2"))
            if not np.isfinite(nrmse):
                out.append(f"{split}:{name} nrmse is non-finite")
            elif np.isfinite(nrmse_limit) and nrmse > nrmse_limit:
                out.append(f"{split}:{name} nrmse={nrmse:.6f} exceeds {nrmse_limit:.6f}")
            if not np.isfinite(r2):
                out.append(f"{split}:{name} r2 is non-finite")
            elif np.isfinite(min_r2) and r2 < min_r2:
                out.append(f"{split}:{name} r2={r2:.6f} below {min_r2:.6f}")
    return out


def validate_thermo_symbolic_quality(
    report: dict[str, Any],
    *,
    validation_mode: str = "strict",
) -> list[str]:
    """Enforce thermo symbolic per-target quality thresholds by mode."""
    mode = str(validation_mode).strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            f"validation_mode must be one of {sorted(_ALLOWED_MODES)}, got {validation_mode!r}"
        )
    if mode == "off":
        return []
    if str(report.get("surrogate_kind", "")).strip() != "thermo_symbolic":
        return []

    reasons = thermo_symbolic_quality_fail_reasons(report)
    if reasons:
        _raise_or_warn(
            mode,
            "thermo symbolic quality gates failed: " + "; ".join(reasons),
            exc_type=ValueError,
        )
    return reasons


def validate_quality_report_schema(report: dict[str, Any]) -> None:
    """Validate quality report contract schema."""
    _require(isinstance(report, dict), "quality_report must be a JSON object")
    _require(str(report.get("schema_version", "")).strip(), "schema_version is required")
    kind = str(report.get("surrogate_kind", "")).strip()
    _require(kind in _ALLOWED_KINDS, f"surrogate_kind must be one of {sorted(_ALLOWED_KINDS)}")
    _require(
        "dataset_manifest" in report and isinstance(report["dataset_manifest"], dict),
        "dataset_manifest must be an object",
    )
    _require(
        "metrics" in report and isinstance(report["metrics"], dict), "metrics must be an object"
    )
    metrics = report["metrics"]
    for split in ("train", "val", "test"):
        _require(
            split in metrics and isinstance(metrics[split], dict),
            f"metrics.{split} must be an object",
        )
    _require(
        "slice_metrics" in metrics and isinstance(metrics["slice_metrics"], list),
        "metrics.slice_metrics must be a list",
    )
    _require(
        "ood_thresholds" in report and isinstance(report["ood_thresholds"], dict),
        "ood_thresholds must be an object",
    )
    _require(
        "uncertainty_calibration" in report and isinstance(report["uncertainty_calibration"], dict),
        "uncertainty_calibration must be an object",
    )
    _require("pass" in report and isinstance(report["pass"], bool), "pass must be a bool")
    _require(
        "fail_reasons" in report and isinstance(report["fail_reasons"], list),
        "fail_reasons must be a list",
    )
    if "required_artifacts" in report:
        _require(
            isinstance(report["required_artifacts"], list), "required_artifacts must be a list"
        )
    if kind == "thermo_symbolic":
        _require(
            "quality_profile" in report and isinstance(report["quality_profile"], dict),
            "quality_profile must be an object for thermo_symbolic reports",
        )
        profile = report["quality_profile"]
        _require(
            str(profile.get("normalization_method", "")).strip() == "p95_p05_range",
            "quality_profile.normalization_method must be 'p95_p05_range' for thermo_symbolic",
        )
        for key in ("val_nrmse_max", "test_nrmse_max", "min_r2"):
            _require(key in profile, f"quality_profile.{key} is required for thermo_symbolic")
        for split in ("train", "val", "test"):
            sm = metrics[split]
            _require(
                "per_target" in sm and isinstance(sm["per_target"], list),
                f"metrics.{split}.per_target must be a list for thermo_symbolic",
            )
            for i, rec in enumerate(sm["per_target"]):
                _require(
                    isinstance(rec, dict),
                    f"metrics.{split}.per_target[{i}] must be an object",
                )
                for key in ("name", "rmse", "mae", "r2", "nrmse", "target_scale"):
                    _require(
                        key in rec,
                        f"metrics.{split}.per_target[{i}].{key} is required for thermo_symbolic",
                    )
    if kind == "openfoam":
        _require(
            "quality_profile" in report and isinstance(report["quality_profile"], dict),
            "quality_profile must be an object for openfoam reports",
        )
        profile = report["quality_profile"]
        _require(
            str(profile.get("normalization_method", "")).strip() == "p95_p05_range",
            "quality_profile.normalization_method must be 'p95_p05_range' for openfoam",
        )
        for key in ("val_nrmse_max", "test_nrmse_max", "min_r2", "require_finite_slice_metrics"):
            _require(key in profile, f"quality_profile.{key} is required for openfoam")
        for split in ("train", "val", "test"):
            sm = metrics[split]
            _require(
                "per_target" in sm and isinstance(sm["per_target"], list),
                f"metrics.{split}.per_target must be a list for openfoam",
            )
            for i, rec in enumerate(sm["per_target"]):
                _require(
                    isinstance(rec, dict),
                    f"metrics.{split}.per_target[{i}] must be an object",
                )
                for key in ("name", "rmse", "mae", "r2", "nrmse", "target_scale"):
                    _require(
                        key in rec,
                        f"metrics.{split}.per_target[{i}].{key} is required for openfoam",
                    )
        _require(
            "data_provenance" in report and isinstance(report["data_provenance"], dict),
            "data_provenance must be an object for openfoam reports",
        )
        provenance = report["data_provenance"]
        _require(
            str(provenance.get("kind", "")).strip() in _OPENFOAM_PROVENANCE_KINDS,
            "data_provenance.kind must be one of "
            f"{sorted(_OPENFOAM_PROVENANCE_KINDS)} for openfoam reports",
        )
        _require(
            isinstance(provenance.get("authoritative_for_strict_f2"), bool),
            "data_provenance.authoritative_for_strict_f2 must be a bool for openfoam reports",
        )
        _require(
            isinstance(provenance.get("anchor_manifest_path", ""), str),
            "data_provenance.anchor_manifest_path must be a string for openfoam reports",
        )
        _require(
            isinstance(provenance.get("anchor_manifest_version", ""), str),
            "data_provenance.anchor_manifest_version must be a string for openfoam reports",
        )
        anchor_count = provenance.get("anchor_count")
        _require(
            isinstance(anchor_count, int) and int(anchor_count) >= 0,
            "data_provenance.anchor_count must be a non-negative int for openfoam reports",
        )
        _require(
            isinstance(provenance.get("truth_source_summary", {}), dict),
            "data_provenance.truth_source_summary must be an object for openfoam reports",
        )


def load_quality_report(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"quality report at '{p}' is not a JSON object")
    validate_quality_report_schema(payload)
    return payload


def write_quality_report(path: str | Path, report: dict[str, Any]) -> Path:
    validate_quality_report_schema(report)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return p


def validate_artifact_quality(
    artifact_path: str | Path,
    *,
    surrogate_kind: str,
    validation_mode: str = "strict",
    required_artifacts: list[str] | None = None,
) -> dict[str, Any] | None:
    """Validate a surrogate artifact against quality_report.json contract."""
    mode = str(validation_mode).strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            f"validation_mode must be one of {sorted(_ALLOWED_MODES)}, got {validation_mode!r}"
        )
    if mode == "off":
        return None

    target = Path(artifact_path)
    report_path = _report_path_for_artifact(target)
    if not report_path.exists():
        _raise_or_warn(
            mode,
            f"Missing required quality report for {surrogate_kind} artifact: '{report_path}'",
            exc_type=FileNotFoundError,
        )
        return None

    try:
        report = load_quality_report(report_path)
    except Exception as exc:
        _raise_or_warn(
            mode,
            f"Invalid quality report schema at '{report_path}': {exc}",
            exc_type=ValueError,
        )
        return None
    if str(report.get("surrogate_kind", "")).strip() != str(surrogate_kind).strip():
        _raise_or_warn(
            mode,
            f"quality_report surrogate_kind mismatch: expected '{surrogate_kind}', got '{report.get('surrogate_kind')}'",
            exc_type=ValueError,
        )
        return None
    if not bool(report.get("pass", False)):
        _raise_or_warn(
            mode,
            f"quality_report marks artifact as failed: reasons={report.get('fail_reasons', [])}",
            exc_type=ValueError,
        )
        return None

    required: list[str] = []
    if isinstance(report.get("required_artifacts"), list):
        required.extend([str(x) for x in report["required_artifacts"]])
    required.extend([str(x) for x in (required_artifacts or [])])

    # Stable de-dup preserving first-seen order.
    deduped: list[str] = []
    seen: set[str] = set()
    for item in required:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)

    root = (
        target
        if target.is_dir() or (not target.exists() and target.suffix == "")
        else target.parent
    )
    for rel in deduped:
        rel_path = Path(rel)
        rp = rel_path if rel_path.is_absolute() else root / rel_path
        if not rp.exists():
            _raise_or_warn(
                mode,
                f"quality_report required artifact missing: '{rp}'",
                exc_type=FileNotFoundError,
            )
            return None

    if target.is_file():
        expected_hash = str(report.get("artifact_sha256", "")).strip()
        if expected_hash:
            actual_hash = sha256_file(target)
            if actual_hash != expected_hash:
                _raise_or_warn(
                    mode,
                    f"artifact hash mismatch for '{target}': expected {expected_hash}, got {actual_hash}",
                    exc_type=ValueError,
                )
                return None

    return report


__all__ = [
    "dataset_manifest_for_file",
    "load_quality_report",
    "openfoam_data_provenance",
    "openfoam_default_data_provenance",
    "openfoam_quality_fail_reasons",
    "openfoam_quality_profile",
    "openfoam_strict_f2_provenance_status",
    "per_target_regression_metrics",
    "regression_metrics",
    "sha256_file",
    "thermo_symbolic_balanced_profile",
    "thermo_symbolic_quality_fail_reasons",
    "validate_openfoam_quality",
    "validate_thermo_symbolic_quality",
    "validate_artifact_quality",
    "validate_openfoam_strict_f2_provenance",
    "validate_quality_report_schema",
    "write_quality_report",
]
