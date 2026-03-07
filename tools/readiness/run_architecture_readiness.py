#!/usr/bin/env python3
"""Architecture-first readiness probes for Explore->Lifetime orchestration."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from larrak2.architecture.contracts import (  # noqa: E402
    CONNECTION_CONTRACT_V1,
    CONTRACT_VERSION,
    CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C,
    EDGE_IDS,
)

THERMO_CONSTANTS_REL_PATH = Path("data/thermo/literature_constants_v1.json")
THERMO_ANCHOR_REL_PATH = Path("data/thermo/anchor_manifest_v1.json")
KNOWN_BLOCKER_TYPES = {
    "orchestration_wiring_gap",
    "contract_shape_gap",
    "fidelity_routing_gap",
    "runtime_dependency_gap",
}


@dataclass(frozen=True)
class ProbeSpec:
    workflow: str
    fidelity: int
    outdir: Path
    manifest_path: Path
    command: list[str]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _classify_failure(message: str) -> str:
    text = message.lower()
    if "contract routing violation" in text or "routing violation" in text:
        return "fidelity_routing_gap"
    if (
        "no module named" in text
        or "filenotfounderror" in text
        or "quality_report" in text
        or "missing artifact" in text
        or "no hard-feasible high-fidelity candidates qualified for downselect" in text
        or "no_hard_feasible_high_fidelity_candidates" in text
        or "thermo validation failed" in text
        or "anchor manifest" in text
        or "not found" in text
        or "source_region_not_ready" in text
        or "source_region_classification:" in text
    ):
        return "runtime_dependency_gap"
    if (
        "missing required" in text
        or "objective dimensionality mismatch" in text
        or "objective name mismatch" in text
    ):
        return "contract_shape_gap"
    return "orchestration_wiring_gap"


def _extract_values(payload: Any, key_path: str) -> list[Any]:
    parts = [p for p in str(key_path).split(".") if p]
    if not parts:
        return []

    def _walk(obj: Any, i: int) -> list[Any]:
        if i >= len(parts):
            return [obj]
        part = parts[i]
        wants_array = part.endswith("[]")
        key = part[:-2] if wants_array else part
        if not isinstance(obj, dict) or key not in obj:
            return []
        nxt = obj[key]
        if wants_array:
            if not isinstance(nxt, list):
                return []
            out: list[Any] = []
            for item in nxt:
                out.extend(_walk(item, i + 1))
            return out
        return _walk(nxt, i + 1)

    return _walk(payload, 0)


def _is_real_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, list):
        if not value:
            return False
        numeric = [v for v in value if isinstance(v, (int, float))]
        if len(numeric) == len(value):
            return all(math.isfinite(float(v)) for v in numeric)
        return True
    if isinstance(value, dict):
        return len(value) > 0
    return value is not None


def _probe_operating_point(fidelity: int) -> tuple[float, float]:
    if int(fidelity) == 2:
        # Intentionally outside validated envelope to preserve strict thresholds
        # while keeping architecture contract observability active.
        return 2000.0, 450.0
    return 2000.0, 80.0


def _build_probe_specs(outdir: Path) -> list[ProbeSpec]:
    specs: list[ProbeSpec] = []
    for fidelity in (0, 1, 2):
        rpm, torque = _probe_operating_point(fidelity)
        pareto_out = outdir / f"run_pareto_f{fidelity}"
        specs.append(
            ProbeSpec(
                workflow="run_pareto",
                fidelity=fidelity,
                outdir=pareto_out,
                manifest_path=pareto_out / "summary.json",
                command=[
                    sys.executable,
                    "-m",
                    "larrak2.cli.run_pareto",
                    "--pop",
                    "8",
                    "--gen",
                    "2",
                    "--rpm",
                    str(rpm),
                    "--torque",
                    str(torque),
                    "--fidelity",
                    str(fidelity),
                    "--seed",
                    str(100 + fidelity),
                    "--constraint-phase",
                    "explore",
                    "--surrogate-validation-mode",
                    "off",
                    "--thermo-model",
                    "two_zone_eq_v1",
                    "--thermo-constants-path",
                    str(THERMO_CONSTANTS_REL_PATH),
                    "--thermo-anchor-manifest",
                    str(THERMO_ANCHOR_REL_PATH),
                    "--no-strict-data",
                    "--outdir",
                    str(pareto_out),
                    "--enforce-contract-routing",
                ],
            )
        )

        ee_out = outdir / f"explore_exploit_f{fidelity}"
        specs.append(
            ProbeSpec(
                workflow="explore_exploit",
                fidelity=fidelity,
                outdir=ee_out,
                manifest_path=ee_out / "explore_exploit_manifest.json",
                command=[
                    sys.executable,
                    "-m",
                    "larrak2.cli.run",
                    "explore-exploit",
                    "--outdir",
                    str(ee_out),
                    "--rpm",
                    str(rpm),
                    "--torque",
                    str(torque),
                    "--seed",
                    str(200 + fidelity),
                    "--explore-source",
                    "principles",
                    "--principles-profile",
                    "iso_litvin_v2",
                    "--principles-region-min-size",
                    "2",
                    "--principles-alignment-mode",
                    "blend",
                    "--principles-canonical-alignment-fidelity",
                    "1",
                    "--principles-root-max-iter",
                    "12",
                    "--top-k",
                    "1",
                    "--mode",
                    "weighted_sum",
                    "--backend",
                    "scipy",
                    "--skip-tribology",
                    "--refine-indices",
                    "0,1",
                    "--max-iter",
                    "4",
                    "--tol",
                    "1e-6",
                    "--explore-fidelity",
                    str(fidelity),
                    "--hifi-fidelity",
                    str(fidelity),
                    "--hifi-constraint-phase",
                    "explore",
                    "--thermo-constants-path",
                    str(THERMO_CONSTANTS_REL_PATH),
                    "--thermo-anchor-manifest",
                    str(THERMO_ANCHOR_REL_PATH),
                    "--thermo-symbolic-mode",
                    "off",
                    "--architecture-probe-mode",
                    "--enforce-contract-routing",
                ],
            )
        )

        orch_out = outdir / f"orchestrate_f{fidelity}"
        specs.append(
            ProbeSpec(
                workflow="orchestrate",
                fidelity=fidelity,
                outdir=orch_out,
                manifest_path=orch_out / "orchestrate_manifest.json",
                command=[
                    sys.executable,
                    "-m",
                    "larrak2.cli.run",
                    "orchestrate",
                    "--outdir",
                    str(orch_out),
                    "--rpm",
                    str(rpm),
                    "--torque",
                    str(torque),
                    "--fidelity",
                    str(fidelity),
                    "--seed",
                    str(300 + fidelity),
                    "--sim-budget",
                    "4",
                    "--batch-size",
                    "4",
                    "--max-iterations",
                    "2",
                    "--truth-dispatch-mode",
                    "auto",
                    "--truth-auto-top-k",
                    "1",
                    "--allow-heuristic-surrogate-fallback",
                    "--surrogate-validation-mode",
                    "off",
                    "--thermo-constants-path",
                    str(THERMO_CONSTANTS_REL_PATH),
                    "--thermo-anchor-manifest",
                    str(THERMO_ANCHOR_REL_PATH),
                    "--thermo-symbolic-mode",
                    "off",
                    "--enforce-contract-routing",
                ],
            )
        )
    return specs


def run_probes(outdir: Path) -> dict[str, Any]:
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    probe_specs = _build_probe_specs(outdir)
    probe_results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    traces_by_fidelity: dict[int, list[dict[str, Any]]] = defaultdict(list)
    manifests_by_probe: dict[str, dict[str, Any]] = {}

    for spec in probe_specs:
        if spec.outdir.exists():
            shutil.rmtree(spec.outdir)
        spec.outdir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{spec.workflow}_f{spec.fidelity}.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_ROOT)
        proc = subprocess.run(
            spec.command,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        full_output = (
            (proc.stdout or "")
            + ("\n" if proc.stdout and proc.stderr else "")
            + (proc.stderr or "")
        )
        log_path.write_text(full_output, encoding="utf-8")
        trace_path = spec.outdir / "contract_trace.jsonl"
        summary_path = spec.outdir / "contract_summary.json"

        manifest_payload: dict[str, Any] = {}
        if spec.manifest_path.exists():
            try:
                manifest_payload = _read_json(spec.manifest_path)
            except Exception:
                manifest_payload = {}
        manifests_by_probe[f"{spec.workflow}:f{spec.fidelity}"] = manifest_payload

        if summary_path.exists():
            try:
                summaries.append(_read_json(summary_path))
            except Exception:
                pass
        traces_by_fidelity[int(spec.fidelity)].extend(_read_jsonl(trace_path))

        blocker_type = ""
        blocker_detail = ""
        manifest_exists = bool(spec.manifest_path.exists())
        contract_trace_exists = bool(trace_path.exists())
        contract_summary_exists = bool(summary_path.exists())
        contract_artifacts_emitted = bool(
            manifest_exists and contract_trace_exists and contract_summary_exists
        )
        source_region_fields_present = bool(
            isinstance(manifest_payload, dict)
            and "principles_problem" in manifest_payload
            and "reduced_core" in manifest_payload
            and "expansion_policy" in manifest_payload
            and "region_summary" in manifest_payload
            and "proxy_vs_canonical" in manifest_payload
            and "diagnosis_classification" in manifest_payload
            and "source_region_pass" in manifest_payload
            and "optimization_pass" in manifest_payload
        )
        source_region_pass = bool(
            manifest_payload.get("source_region_pass", False)
            if isinstance(manifest_payload, dict)
            else False
        )
        diagnosis_classification = str(
            manifest_payload.get("diagnosis_classification", "")
            if isinstance(manifest_payload, dict)
            else ""
        ).strip()

        if proc.returncode != 0:
            blocker_type = _classify_failure(full_output)
            blocker_detail = (full_output.strip().splitlines() or [""])[-1][:400]
        elif spec.workflow == "explore_exploit" and not source_region_fields_present:
            blocker_type = "contract_shape_gap"
            blocker_detail = (
                "explore-exploit manifest missing required source-region contract fields"
            )

        probe_results.append(
            {
                "workflow": spec.workflow,
                "fidelity": int(spec.fidelity),
                "command": spec.command,
                "outdir": str(spec.outdir),
                "log_file": str(log_path),
                "exit_code": int(proc.returncode),
                "process_success": bool(proc.returncode == 0),
                "success": bool(proc.returncode == 0),
                "manifest_file": str(spec.manifest_path),
                "manifest_exists": manifest_exists,
                "contract_trace_file": str(trace_path),
                "contract_summary_file": str(summary_path),
                "contract_trace_exists": contract_trace_exists,
                "contract_summary_exists": contract_summary_exists,
                "contract_artifacts_emitted": contract_artifacts_emitted,
                "source_region_fields_present": bool(source_region_fields_present),
                "source_region_pass": bool(source_region_pass),
                "diagnosis_classification": diagnosis_classification,
                "blocker_type": blocker_type,
                "blocker_detail": blocker_detail,
            }
        )

    blocker_counts: dict[str, int] = {}
    unclassified_blockers: list[dict[str, Any]] = []
    for item in probe_results:
        if bool(item.get("success", False)):
            continue
        bt = str(item.get("blocker_type", "")).strip()
        if bt:
            blocker_counts[bt] = blocker_counts.get(bt, 0) + 1
        if bt not in KNOWN_BLOCKER_TYPES:
            unclassified_blockers.append(
                {
                    "workflow": str(item.get("workflow", "")),
                    "fidelity": int(item.get("fidelity", -1)),
                    "blocker_type": bt,
                    "blocker_detail": str(item.get("blocker_detail", "")),
                    "log_file": str(item.get("log_file", "")),
                }
            )

    return {
        "generated_at": _now(),
        "contract_version": CONTRACT_VERSION,
        "probes": probe_results,
        "blocker_counts": blocker_counts,
        "known_blocker_types": sorted(KNOWN_BLOCKER_TYPES),
        "unclassified_blockers": unclassified_blockers,
        "summaries": summaries,
        "traces_by_fidelity": traces_by_fidelity,
        "manifests_by_probe": manifests_by_probe,
    }


def build_edge_coverage(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    per_fidelity: dict[int, dict[str, Any]] = defaultdict(lambda: defaultdict(dict))
    coverage: dict[str, Any] = {
        "generated_at": _now(),
        "contract_version": CONTRACT_VERSION,
        "required_edges": list(EDGE_IDS),
        "by_fidelity": {},
    }

    for fidelity in (0, 1, 2):
        for edge in EDGE_IDS:
            per_fidelity[fidelity][edge] = {
                "seen": 0,
                "ok": 0,
                "error": 0,
                "modes_seen": set(),
                "routing_violations": 0,
                "missing_required_request_keys": set(),
                "missing_required_response_keys": set(),
            }

    for summary in summaries:
        fidelity = int(summary.get("fidelity", -1))
        edges = summary.get("edges", {})
        for edge in EDGE_IDS:
            stats = per_fidelity.get(fidelity, {}).get(edge)
            if stats is None:
                continue
            edge_payload = edges.get(edge, {})
            stats["seen"] += int(edge_payload.get("seen", 0))
            stats["ok"] += int(edge_payload.get("ok", 0))
            stats["error"] += int(edge_payload.get("error", 0))
            stats["routing_violations"] += int(edge_payload.get("routing_violations", 0))
            stats["modes_seen"].update(edge_payload.get("modes_seen", []))
            stats["missing_required_request_keys"].update(
                edge_payload.get("missing_required_request_keys", [])
            )
            stats["missing_required_response_keys"].update(
                edge_payload.get("missing_required_response_keys", [])
            )

    f0_all_seen = True
    f0_all_placeholder = True
    for edge in EDGE_IDS:
        stats = per_fidelity[0][edge]
        if int(stats["seen"]) <= 0:
            f0_all_seen = False
        modes = set(stats["modes_seen"])
        if not modes or not modes.issubset({"placeholder"}):
            f0_all_placeholder = False

    by_fidelity_payload: dict[str, Any] = {}
    for fidelity in (0, 1, 2):
        edge_payload: dict[str, Any] = {}
        for edge in EDGE_IDS:
            stats = per_fidelity[fidelity][edge]
            edge_payload[edge] = {
                "seen": int(stats["seen"]),
                "ok": int(stats["ok"]),
                "error": int(stats["error"]),
                "modes_seen": sorted(stats["modes_seen"]),
                "routing_violations": int(stats["routing_violations"]),
                "missing_required_request_keys": sorted(stats["missing_required_request_keys"]),
                "missing_required_response_keys": sorted(stats["missing_required_response_keys"]),
            }
        by_fidelity_payload[str(fidelity)] = edge_payload

    coverage["by_fidelity"] = by_fidelity_payload
    coverage["f0_all_required_edges_seen"] = bool(f0_all_seen)
    coverage["f0_all_required_edges_placeholder"] = bool(f0_all_placeholder)
    coverage["f0_contract_policy_pass"] = bool(f0_all_seen and f0_all_placeholder)
    return coverage


def build_key_parity(traces_by_fidelity: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    required_edges = CONNECTION_CONTRACT_V1.get("required_edges", [])
    edge_spec = {str(item.get("edge_id", "")): item for item in required_edges}
    traces_f0 = traces_by_fidelity.get(0, [])
    traces_f2 = traces_by_fidelity.get(2, [])

    observed: dict[int, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"request": set(), "response": set()})
    )
    for fidelity, rows in ((0, traces_f0), (2, traces_f2)):
        for row in rows:
            edge = str(row.get("edge_id", ""))
            observed[fidelity][edge]["request"].update(row.get("request_keys", []))
            observed[fidelity][edge]["response"].update(row.get("response_keys", []))

    edges_payload: dict[str, Any] = {}
    parity_pass = True
    for edge in EDGE_IDS:
        spec = edge_spec.get(edge, {})
        req_required = set(spec.get("required_request_keys", []))
        resp_required = set(spec.get("required_response_keys", []))
        req_f0 = observed[0][edge]["request"]
        req_f2 = observed[2][edge]["request"]
        resp_f0 = observed[0][edge]["response"]
        resp_f2 = observed[2][edge]["response"]

        missing_f0_req = sorted(req_required - req_f0)
        missing_f2_req = sorted(req_required - req_f2)
        missing_f0_resp = sorted(resp_required - resp_f0)
        missing_f2_resp = sorted(resp_required - resp_f2)

        req_presence_mismatch = sorted((req_required & req_f0) ^ (req_required & req_f2))
        resp_presence_mismatch = sorted((resp_required & resp_f0) ^ (resp_required & resp_f2))

        edge_pass = (
            len(missing_f0_req) == 0
            and len(missing_f2_req) == 0
            and len(missing_f0_resp) == 0
            and len(missing_f2_resp) == 0
            and len(req_presence_mismatch) == 0
            and len(resp_presence_mismatch) == 0
        )
        if not edge_pass:
            parity_pass = False

        edges_payload[edge] = {
            "required_request_keys": sorted(req_required),
            "required_response_keys": sorted(resp_required),
            "observed_request_keys_f0": sorted(req_f0),
            "observed_request_keys_f2": sorted(req_f2),
            "observed_response_keys_f0": sorted(resp_f0),
            "observed_response_keys_f2": sorted(resp_f2),
            "missing_required_request_keys_f0": missing_f0_req,
            "missing_required_request_keys_f2": missing_f2_req,
            "missing_required_response_keys_f0": missing_f0_resp,
            "missing_required_response_keys_f2": missing_f2_resp,
            "request_presence_mismatch": req_presence_mismatch,
            "response_presence_mismatch": resp_presence_mismatch,
            "pass": bool(edge_pass),
        }

    return {
        "generated_at": _now(),
        "contract_version": CONTRACT_VERSION,
        "required_key_parity_pass": bool(parity_pass),
        "edges": edges_payload,
    }


def build_critical_real_key_report(
    traces_by_fidelity: dict[int, list[dict[str, Any]]],
    manifests_by_probe: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    traces_f1 = traces_by_fidelity.get(1, [])
    payloads: list[dict[str, Any]] = []
    for row in traces_f1:
        if isinstance(row.get("response_payload"), dict):
            payloads.append(dict(row["response_payload"]))
        if isinstance(row.get("request_payload"), dict):
            payloads.append(dict(row["request_payload"]))
    for key, manifest in manifests_by_probe.items():
        if ":f1" in key and isinstance(manifest, dict):
            payloads.append(manifest)

    report: dict[str, Any] = {}
    all_pass = True
    for key_path in CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C:
        observed_values: list[Any] = []
        for payload in payloads:
            observed_values.extend(_extract_values(payload, key_path))
        observed_values = [v for v in observed_values if v is not None]
        pass_flag = any(_is_real_value(v) for v in observed_values)
        if not pass_flag:
            all_pass = False
        report[key_path] = {
            "present": bool(len(observed_values) > 0),
            "real_valued": bool(pass_flag),
            "sample": observed_values[0] if observed_values else None,
        }

    return {
        "generated_at": _now(),
        "contract_version": CONTRACT_VERSION,
        "critical_key_set": list(CRITICAL_REAL_KEY_PATHS_STAGE_A_TO_C),
        "all_critical_keys_real_valued": bool(all_pass),
        "keys": report,
    }


def build_gap_ledger(
    *,
    workflow_probe_results: dict[str, Any],
    edge_coverage: dict[str, Any],
    key_parity: dict[str, Any],
    critical_real_keys: dict[str, Any],
) -> dict[str, Any]:
    probes = workflow_probe_results.get("probes", [])
    blocker_counts = dict(workflow_probe_results.get("blocker_counts", {}))
    failed = [p for p in probes if not bool(p.get("success", False))]

    manifest_fields_ok = True
    wiring_ok = True
    for probe in probes:
        if not bool(probe.get("manifest_exists", False)):
            manifest_fields_ok = False
            wiring_ok = False
        if str(probe.get("workflow", "")) == "explore_exploit" and not bool(
            probe.get("source_region_fields_present", False)
        ):
            manifest_fields_ok = False

    routing_violations = 0
    for fidelity_payload in edge_coverage.get("by_fidelity", {}).values():
        for edge_payload in fidelity_payload.values():
            routing_violations += int(edge_payload.get("routing_violations", 0))

    def _status(ok: bool, partial_on_failures: bool = True) -> str:
        if ok:
            return "closed"
        if partial_on_failures and failed:
            return "partial"
        return "open"

    gaps = [
        {
            "gap_id": "arch.wiring.workflow_execution",
            "area": "orchestration_wiring",
            "planned_gap_text": "Workflow probes execute end-to-end with manifests/traces emitted.",
            "status": _status(wiring_ok),
            "evidence_files": [str(p.get("log_file", "")) for p in probes],
            "blocking_for_A_to_C": not bool(wiring_ok),
            "blocker_type": "orchestration_wiring_gap" if not wiring_ok else "",
        },
        {
            "gap_id": "arch.contract.manifest_metadata",
            "area": "contract_shape",
            "planned_gap_text": "Workflow manifests expose additive contract metadata fields.",
            "status": _status(manifest_fields_ok),
            "evidence_files": [str(p.get("manifest_file", "")) for p in probes],
            "blocking_for_A_to_C": not bool(manifest_fields_ok),
            "blocker_type": "contract_shape_gap" if not manifest_fields_ok else "",
        },
        {
            "gap_id": "arch.coverage.f0_placeholder",
            "area": "fidelity_routing",
            "planned_gap_text": "Fidelity 0 triggers all required edges in placeholder mode.",
            "status": "closed"
            if bool(edge_coverage.get("f0_contract_policy_pass", False))
            else "open",
            "evidence_files": [],
            "blocking_for_A_to_C": not bool(edge_coverage.get("f0_contract_policy_pass", False)),
            "blocker_type": "fidelity_routing_gap"
            if not bool(edge_coverage.get("f0_contract_policy_pass", False))
            else "",
        },
        {
            "gap_id": "arch.parity.f0_vs_f2_required_keys",
            "area": "contract_shape",
            "planned_gap_text": "Exact required key-path parity is preserved between fidelity 0 and 2.",
            "status": "closed"
            if bool(key_parity.get("required_key_parity_pass", False))
            else "open",
            "evidence_files": [],
            "blocking_for_A_to_C": not bool(key_parity.get("required_key_parity_pass", False)),
            "blocker_type": "contract_shape_gap"
            if not bool(key_parity.get("required_key_parity_pass", False))
            else "",
        },
        {
            "gap_id": "arch.f1.critical_real_keys",
            "area": "contract_shape",
            "planned_gap_text": "Fidelity 1 Stage A->C critical keys are present and real-valued.",
            "status": "closed"
            if bool(critical_real_keys.get("all_critical_keys_real_valued", False))
            else "open",
            "evidence_files": [],
            "blocking_for_A_to_C": not bool(
                critical_real_keys.get("all_critical_keys_real_valued", False)
            ),
            "blocker_type": "contract_shape_gap"
            if not bool(critical_real_keys.get("all_critical_keys_real_valued", False))
            else "",
        },
        {
            "gap_id": "arch.routing.violations",
            "area": "fidelity_routing",
            "planned_gap_text": "No observed edge routing violations against fidelity policy.",
            "status": "closed" if int(routing_violations) == 0 else "open",
            "evidence_files": [],
            "blocking_for_A_to_C": int(routing_violations) > 0,
            "blocker_type": "fidelity_routing_gap" if int(routing_violations) > 0 else "",
        },
    ]

    counts = {"closed": 0, "partial": 0, "open": 0}
    for gap in gaps:
        counts[str(gap["status"])] += 1

    unresolved_wiring_contract_count = sum(
        1
        for gap in gaps
        if str(gap.get("area", "")) in {"orchestration_wiring", "contract_shape"}
        and str(gap.get("status", "")) != "closed"
    )
    unclassified_blocker_count = len(workflow_probe_results.get("unclassified_blockers", []))
    requirements = {
        "required_key_parity_pass": bool(key_parity.get("required_key_parity_pass", False)),
        "unresolved_wiring_contract_count": int(unresolved_wiring_contract_count),
        "unclassified_blocker_count": int(unclassified_blocker_count),
    }

    return {
        "generated_at": _now(),
        "contract_version": CONTRACT_VERSION,
        "counts": counts,
        "blocker_counts_from_probes": blocker_counts,
        "requirements": requirements,
        "gaps": gaps,
    }


def write_summary(
    *,
    out_path: Path,
    ledger: dict[str, Any],
    workflow_probe_results: dict[str, Any],
    edge_coverage: dict[str, Any],
    key_parity: dict[str, Any],
    critical_real_keys: dict[str, Any],
) -> None:
    blocker_counts = dict(workflow_probe_results.get("blocker_counts", {}))
    gaps = list(ledger.get("gaps", []))
    unresolved_wiring = sum(
        1
        for gap in gaps
        if str(gap.get("area", "")) == "orchestration_wiring"
        and str(gap.get("status", "")) != "closed"
    )
    unresolved_shape = sum(
        1
        for gap in gaps
        if str(gap.get("area", "")) == "contract_shape" and str(gap.get("status", "")) != "closed"
    )
    unresolved_routing = sum(
        1
        for gap in gaps
        if str(gap.get("area", "")) == "fidelity_routing" and str(gap.get("status", "")) != "closed"
    )
    unresolved_runtime = int(blocker_counts.get("runtime_dependency_gap", 0))
    unclassified = len(workflow_probe_results.get("unclassified_blockers", []))

    architecture_ready = bool(
        unresolved_wiring == 0
        and unresolved_shape == 0
        and unclassified == 0
        and bool(edge_coverage.get("f0_contract_policy_pass", False))
        and bool(key_parity.get("required_key_parity_pass", False))
        and bool(critical_real_keys.get("all_critical_keys_real_valued", False))
    )

    lines = [
        "# Pipeline Architecture Readiness Summary",
        "",
        f"- Generated: {_now()}",
        f"- Contract version: `{CONTRACT_VERSION}`",
        f"- Closed/partial/open: `{ledger.get('counts', {}).get('closed', 0)}`/"
        f"`{ledger.get('counts', {}).get('partial', 0)}`/"
        f"`{ledger.get('counts', {}).get('open', 0)}`",
        "",
        "## A->C Architecture Go/No-Go",
        f"- Decision: **{'GO' if architecture_ready else 'NO-GO'}**",
        f"- F0 edge placeholder policy pass: `{bool(edge_coverage.get('f0_contract_policy_pass', False))}`",
        f"- F0 vs F2 required key parity pass: `{bool(key_parity.get('required_key_parity_pass', False))}`",
        "- F1 critical key set real-valued: "
        f"`{bool(critical_real_keys.get('all_critical_keys_real_valued', False))}`",
        "",
        "## Blockers (Ordered)",
        f"1. orchestration_wiring_gap: `{unresolved_wiring}`",
        f"2. contract_shape_gap: `{unresolved_shape}`",
        f"3. fidelity_routing_gap: `{unresolved_routing}`",
        f"4. runtime_dependency_gap: `{unresolved_runtime}`",
        f"5. unclassified: `{unclassified}`",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "outputs" / "readiness" / "architecture",
        help="Output directory for architecture readiness artifacts",
    )
    parser.add_argument(
        "--require-architecture-pass",
        action="store_true",
        help="Return non-zero unless required parity/wiring/contract/unclassified architecture gates pass",
    )
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    probe_data = run_probes(outdir)
    workflow_probe_results = {
        "generated_at": probe_data["generated_at"],
        "contract_version": CONTRACT_VERSION,
        "probes": probe_data["probes"],
        "blocker_counts": probe_data["blocker_counts"],
        "known_blocker_types": probe_data["known_blocker_types"],
        "unclassified_blockers": probe_data["unclassified_blockers"],
    }
    _json_dump(outdir / "workflow_probe_results.json", workflow_probe_results)

    edge_coverage = build_edge_coverage(probe_data["summaries"])
    _json_dump(outdir / "edge_coverage_report.json", edge_coverage)

    key_parity = build_key_parity(probe_data["traces_by_fidelity"])
    _json_dump(outdir / "key_parity_report_f0_vs_f2.json", key_parity)

    critical_real_keys = build_critical_real_key_report(
        probe_data["traces_by_fidelity"], probe_data["manifests_by_probe"]
    )
    _json_dump(outdir / "critical_real_key_report_f1.json", critical_real_keys)

    ledger = build_gap_ledger(
        workflow_probe_results=workflow_probe_results,
        edge_coverage=edge_coverage,
        key_parity=key_parity,
        critical_real_keys=critical_real_keys,
    )
    _json_dump(outdir / "architecture_gap_ledger.json", ledger)

    write_summary(
        out_path=outdir / "pipeline_arch_readiness_summary.md",
        ledger=ledger,
        workflow_probe_results=workflow_probe_results,
        edge_coverage=edge_coverage,
        key_parity=key_parity,
        critical_real_keys=critical_real_keys,
    )

    if bool(args.require_architecture_pass):
        req = dict(ledger.get("requirements", {}))
        hard_fail = (
            not bool(req.get("required_key_parity_pass", False))
            or int(req.get("unresolved_wiring_contract_count", 0)) > 0
            or int(req.get("unclassified_blocker_count", 0)) > 0
        )
        if hard_fail:
            print(
                "Architecture readiness gate failed: "
                f"required_key_parity_pass={bool(req.get('required_key_parity_pass', False))}, "
                f"unresolved_wiring_contract_count={int(req.get('unresolved_wiring_contract_count', 0))}, "
                f"unclassified_blocker_count={int(req.get('unclassified_blocker_count', 0))}"
            )
            return 2

    print(f"Architecture readiness artifacts written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
