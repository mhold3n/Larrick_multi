#!/usr/bin/env python3
"""Deterministic readiness audit for artifact/data contracts.

Outputs a machine-readable report with explicit pass/fail checks,
severity, A->C blocking flags, and remediation commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from larrak_runtime.core.artifact_paths import (  # noqa: E402
    DEFAULT_CALCULIX_NN_ARTIFACT,
    DEFAULT_HIFI_SURROGATE_DIR,
    DEFAULT_OPENFOAM_NN_ARTIFACT,
    DEFAULT_STACK_SURROGATE_ARTIFACT,
    DEFAULT_THERMO_SYMBOLIC_ARTIFACT,
)
from larrak_runtime.surrogate.quality_contract import validate_artifact_quality  # noqa: E402


@dataclass
class CheckResult:
    check_id: str
    category: str
    description: str
    severity: str
    blocking_for_a_to_c: bool
    passed: bool
    details: str
    evidence_path: str = ""
    remediation_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "blocking_for_A_to_C": self.blocking_for_a_to_c,
            "pass": self.passed,
            "details": self.details,
            "evidence_path": self.evidence_path,
            "remediation_command": self.remediation_command,
        }


def _norm(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _append(results: list[CheckResult], result: CheckResult) -> None:
    results.append(result)


def _artifact_quality_check(
    *,
    results: list[CheckResult],
    check_id: str,
    description: str,
    target: Path,
    surrogate_kind: str,
    required_artifacts: list[str],
    severity: str,
    blocking_for_a_to_c: bool,
    remediation: str,
) -> None:
    if not target.exists():
        _append(
            results,
            CheckResult(
                check_id=check_id,
                category="artifact",
                description=description,
                severity=severity,
                blocking_for_a_to_c=blocking_for_a_to_c,
                passed=False,
                details=f"Missing artifact target: {target}",
                evidence_path=_norm(target),
                remediation_command=remediation,
            ),
        )
        return
    try:
        validate_artifact_quality(
            target,
            surrogate_kind=surrogate_kind,
            validation_mode="strict",
            required_artifacts=required_artifacts,
        )
        _append(
            results,
            CheckResult(
                check_id=check_id,
                category="artifact",
                description=description,
                severity=severity,
                blocking_for_a_to_c=blocking_for_a_to_c,
                passed=True,
                details="quality_report contract and required artifacts validated",
                evidence_path=_norm(target),
                remediation_command=remediation,
            ),
        )
    except Exception as exc:
        _append(
            results,
            CheckResult(
                check_id=check_id,
                category="artifact",
                description=description,
                severity=severity,
                blocking_for_a_to_c=blocking_for_a_to_c,
                passed=False,
                details=f"{type(exc).__name__}: {exc}",
                evidence_path=_norm(target),
                remediation_command=remediation,
            ),
        )


def _anchor_manifest_check(results: list[CheckResult]) -> None:
    manifest_path = REPO_ROOT / "data" / "thermo" / "anchor_manifest_v1.json"
    remediation = (
        "PYTHONPATH=src python tools/build_thermo_anchor_manifest.py "
        "--input outputs/orchestration/truth_records.jsonl "
        "--output data/thermo/anchor_manifest_v1.json --source truth_runs"
    )
    if not manifest_path.exists():
        _append(
            results,
            CheckResult(
                check_id="thermo.anchor_manifest.exists",
                category="thermo_anchor",
                description="Default thermo anchor manifest exists",
                severity="hard",
                blocking_for_a_to_c=False,
                passed=False,
                details="anchor_manifest_v1.json not found",
                evidence_path=_norm(manifest_path),
                remediation_command=remediation,
            ),
        )
        return

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        anchors = payload.get("anchors", [])
        if not isinstance(anchors, list):
            raise ValueError("anchors must be a list")
        if not anchors:
            raise ValueError("anchors list is empty")
        for i, anchor in enumerate(anchors):
            if not isinstance(anchor, dict):
                raise ValueError(f"anchor[{i}] is not an object")
            rpm = float(anchor.get("rpm"))
            torque = float(anchor.get("torque"))
            if not math.isfinite(rpm) or rpm <= 0:
                raise ValueError(f"anchor[{i}].rpm must be finite and > 0")
            if not math.isfinite(torque) or torque < 0:
                raise ValueError(f"anchor[{i}].torque must be finite and >= 0")
        _append(
            results,
            CheckResult(
                check_id="thermo.anchor_manifest.valid",
                category="thermo_anchor",
                description="Default thermo anchor manifest is schema-valid and non-empty",
                severity="hard",
                blocking_for_a_to_c=False,
                passed=True,
                details=f"validated {len(anchors)} anchors",
                evidence_path=_norm(manifest_path),
                remediation_command=remediation,
            ),
        )
    except Exception as exc:
        _append(
            results,
            CheckResult(
                check_id="thermo.anchor_manifest.valid",
                category="thermo_anchor",
                description="Default thermo anchor manifest is schema-valid and non-empty",
                severity="hard",
                blocking_for_a_to_c=False,
                passed=False,
                details=f"{type(exc).__name__}: {exc}",
                evidence_path=_norm(manifest_path),
                remediation_command=remediation,
            ),
        )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _cem_table_checks(results: list[CheckResult]) -> None:
    cem_dir = REPO_ROOT / "data" / "cem"
    required_tables = {
        "tribology_ehl_coefficients.csv": {
            "required_columns": [
                "oil_type",
                "finish_tier",
                "temp_C_min",
                "temp_C_max",
                "unit_system",
                "provenance",
                "version",
            ],
        },
        "scuffing_critical_temperatures.csv": {
            "required_columns": [
                "oil_type",
                "additive_package",
                "method",
                "load_stage",
                "unit_temp",
                "provenance",
                "version",
            ],
        },
        "micropitting_lambda_perm.csv": {
            "required_columns": [
                "oil_type",
                "finish_tier",
                "unit_lambda",
                "provenance",
                "version",
            ],
        },
        "fzg_step_load_map.csv": {
            "required_columns": [
                "test_standard",
                "test_method",
                "load_stage",
                "oil_type",
                "additive_package",
                "unit_temp",
                "provenance",
                "version",
            ],
        },
        "limit_stress_numbers.csv": {
            "required_columns": [
                "route_id",
                "sigma_Hlim_MPa",
                "provenance",
                "version",
            ],
        },
        "route_metadata.csv": {
            "required_columns": [
                "route_id",
                "cleanliness_grade_proxy",
                "max_service_temp_C",
                "provenance",
                "version",
            ],
        },
    }

    for fname, cfg in required_tables.items():
        table_path = cem_dir / fname
        remediation = f"Populate/repair data table: {table_path}"
        if not table_path.exists():
            _append(
                results,
                CheckResult(
                    check_id=f"data.cem.{fname}.exists",
                    category="data_contract",
                    description=f"Required CEM table exists: {fname}",
                    severity="hard",
                    blocking_for_a_to_c=True,
                    passed=False,
                    details="file missing",
                    evidence_path=_norm(table_path),
                    remediation_command=remediation,
                ),
            )
            continue
        try:
            cols, rows = _read_csv(table_path)
            missing_cols = [c for c in cfg["required_columns"] if c not in cols]
            if missing_cols:
                raise ValueError(f"missing required columns: {missing_cols}")
            if len(rows) == 0:
                raise ValueError("table has zero data rows")
            _append(
                results,
                CheckResult(
                    check_id=f"data.cem.{fname}.schema",
                    category="data_contract",
                    description=f"Required CEM table schema/data valid: {fname}",
                    severity="hard",
                    blocking_for_a_to_c=True,
                    passed=True,
                    details=f"{len(rows)} data rows; required columns present",
                    evidence_path=_norm(table_path),
                    remediation_command=remediation,
                ),
            )
        except Exception as exc:
            _append(
                results,
                CheckResult(
                    check_id=f"data.cem.{fname}.schema",
                    category="data_contract",
                    description=f"Required CEM table schema/data valid: {fname}",
                    severity="hard",
                    blocking_for_a_to_c=True,
                    passed=False,
                    details=f"{type(exc).__name__}: {exc}",
                    evidence_path=_norm(table_path),
                    remediation_command=remediation,
                ),
            )


def run_audit(*, check_hifi: bool, check_thermo_symbolic: bool) -> dict[str, Any]:
    results: list[CheckResult] = []

    _artifact_quality_check(
        results=results,
        check_id="artifact.openfoam.default_contract",
        description="Default OpenFOAM artifact passes strict quality contract",
        target=REPO_ROOT / DEFAULT_OPENFOAM_NN_ARTIFACT,
        surrogate_kind="openfoam",
        required_artifacts=[Path(DEFAULT_OPENFOAM_NN_ARTIFACT).name],
        severity="hard",
        blocking_for_a_to_c=True,
        remediation=(
            "PYTHONPATH=src python -m larrak2.cli.run train-surrogates "
            "--single-condition --openfoam-epochs 5 --calculix-epochs 5"
        ),
    )
    _artifact_quality_check(
        results=results,
        check_id="artifact.calculix.default_contract",
        description="Default CalculiX artifact passes strict quality contract",
        target=REPO_ROOT / DEFAULT_CALCULIX_NN_ARTIFACT,
        surrogate_kind="calculix",
        required_artifacts=[Path(DEFAULT_CALCULIX_NN_ARTIFACT).name],
        severity="hard",
        blocking_for_a_to_c=True,
        remediation=(
            "PYTHONPATH=src python -m larrak2.cli.run train-surrogates "
            "--single-condition --openfoam-epochs 5 --calculix-epochs 5"
        ),
    )
    _artifact_quality_check(
        results=results,
        check_id="artifact.stack.default_contract",
        description="Default stack artifact passes strict quality contract",
        target=REPO_ROOT / DEFAULT_STACK_SURROGATE_ARTIFACT,
        surrogate_kind="stack",
        required_artifacts=[Path(DEFAULT_STACK_SURROGATE_ARTIFACT).name],
        severity="hard",
        blocking_for_a_to_c=True,
        remediation=(
            "PYTHONPATH=src python -m larrak2.cli.run train-stack-surrogate "
            "--pareto-dir outputs/dress_rehearsal/optimization --fidelity 1 --epochs 8"
        ),
    )

    if check_thermo_symbolic:
        _artifact_quality_check(
            results=results,
            check_id="artifact.thermo_symbolic.default_contract",
            description="Default thermo symbolic artifact passes strict quality contract",
            target=REPO_ROOT / DEFAULT_THERMO_SYMBOLIC_ARTIFACT,
            surrogate_kind="thermo_symbolic",
            required_artifacts=[Path(DEFAULT_THERMO_SYMBOLIC_ARTIFACT).name],
            severity="hard",
            blocking_for_a_to_c=False,
            remediation=(
                "PYTHONPATH=src python -m larrak2.cli.run train-thermo-symbolic "
                "--fidelity 1 --n-samples 256"
            ),
        )
    else:
        _append(
            results,
            CheckResult(
                check_id="artifact.thermo_symbolic.default_contract",
                category="artifact",
                description="Default thermo symbolic artifact check skipped (feature optional/off)",
                severity="info",
                blocking_for_a_to_c=False,
                passed=True,
                details="thermo symbolic check disabled",
                evidence_path=_norm(REPO_ROOT / DEFAULT_THERMO_SYMBOLIC_ARTIFACT),
                remediation_command="",
            ),
        )

    if check_hifi:
        _artifact_quality_check(
            results=results,
            check_id="artifact.hifi.default_contract",
            description="Default HiFi model directory passes strict quality contract",
            target=REPO_ROOT / DEFAULT_HIFI_SURROGATE_DIR,
            surrogate_kind="hifi",
            required_artifacts=[
                "thermal_surrogate.pt",
                "structural_surrogate.pt",
                "flow_surrogate.pt",
                "normalization.json",
            ],
            severity="hard",
            blocking_for_a_to_c=False,
            remediation=(
                "Provide/restore validated HiFi artifacts and quality_report.json "
                "under outputs/artifacts/surrogates/hifi"
            ),
        )
    else:
        _append(
            results,
            CheckResult(
                check_id="artifact.hifi.default_contract",
                category="artifact",
                description="Default HiFi artifact check skipped",
                severity="info",
                blocking_for_a_to_c=False,
                passed=True,
                details="hifi check disabled",
                evidence_path=_norm(REPO_ROOT / DEFAULT_HIFI_SURROGATE_DIR),
                remediation_command="",
            ),
        )

    _anchor_manifest_check(results)
    _cem_table_checks(results)

    total = len(results)
    failed = sum(1 for r in results if not r.passed)
    hard_failed = sum(1 for r in results if (not r.passed and r.severity == "hard"))
    a_to_c_hard_failed = sum(
        1 for r in results if (not r.passed and r.severity == "hard" and r.blocking_for_a_to_c)
    )
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": _norm(REPO_ROOT),
        "summary": {
            "total_checks": total,
            "passed": total - failed,
            "failed": failed,
            "hard_failed": hard_failed,
            "a_to_c_hard_failed": a_to_c_hard_failed,
        },
        "checks": [r.to_dict() for r in results],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit pipeline artifact/data contracts.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/readiness/artifact_contract_audit.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--check-hifi",
        action="store_true",
        help="Enable strict quality-contract check for default HiFi model directory",
    )
    parser.add_argument(
        "--check-thermo-symbolic",
        action="store_true",
        help="Enable strict quality-contract check for thermo symbolic artifact",
    )
    parser.add_argument(
        "--fail-on-a-to-c-hard-fail",
        action="store_true",
        help="Return non-zero when A->C blocking hard failures are present",
    )
    args = parser.parse_args()

    payload = run_audit(
        check_hifi=bool(args.check_hifi),
        check_thermo_symbolic=bool(args.check_thermo_symbolic),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote audit report: {out_path}")
    print(json.dumps(payload["summary"], indent=2))

    if bool(args.fail_on_a_to_c_hard_fail) and int(payload["summary"]["a_to_c_hard_failed"]) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
