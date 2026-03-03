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
_ALLOWED_KINDS = {"stack", "openfoam", "calculix", "hifi"}


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


def _raise_or_warn(mode: str, message: str, *, exc_type: type[Exception]) -> None:
    if mode == "strict":
        raise exc_type(message)
    if mode == "warn":
        LOGGER.warning(message)


def _report_path_for_artifact(artifact_path: Path) -> Path:
    if artifact_path.is_dir():
        return artifact_path / "quality_report.json"
    return artifact_path.parent / "quality_report.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_quality_report_schema(report: dict[str, Any]) -> None:
    """Validate quality report contract schema."""
    _require(isinstance(report, dict), "quality_report must be a JSON object")
    _require(str(report.get("schema_version", "")).strip(), "schema_version is required")
    kind = str(report.get("surrogate_kind", "")).strip()
    _require(kind in _ALLOWED_KINDS, f"surrogate_kind must be one of {sorted(_ALLOWED_KINDS)}")
    _require("dataset_manifest" in report and isinstance(report["dataset_manifest"], dict), "dataset_manifest must be an object")
    _require("metrics" in report and isinstance(report["metrics"], dict), "metrics must be an object")
    metrics = report["metrics"]
    for split in ("train", "val", "test"):
        _require(split in metrics and isinstance(metrics[split], dict), f"metrics.{split} must be an object")
    _require("slice_metrics" in metrics and isinstance(metrics["slice_metrics"], list), "metrics.slice_metrics must be a list")
    _require("ood_thresholds" in report and isinstance(report["ood_thresholds"], dict), "ood_thresholds must be an object")
    _require(
        "uncertainty_calibration" in report and isinstance(report["uncertainty_calibration"], dict),
        "uncertainty_calibration must be an object",
    )
    _require("pass" in report and isinstance(report["pass"], bool), "pass must be a bool")
    _require("fail_reasons" in report and isinstance(report["fail_reasons"], list), "fail_reasons must be a list")
    if "required_artifacts" in report:
        _require(isinstance(report["required_artifacts"], list), "required_artifacts must be a list")


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
        raise ValueError(f"validation_mode must be one of {sorted(_ALLOWED_MODES)}, got {validation_mode!r}")
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

    report = load_quality_report(report_path)
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

    required = list(required_artifacts or [])
    if not required and isinstance(report.get("required_artifacts"), list):
        required = [str(x) for x in report["required_artifacts"]]

    root = target if target.is_dir() else target.parent
    for rel in required:
        rp = root / str(rel)
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
    "regression_metrics",
    "sha256_file",
    "validate_artifact_quality",
    "validate_quality_report_schema",
    "write_quality_report",
]
