#!/usr/bin/env python3
"""
Larrick GUI bridge CLI.

This module is the first-party process boundary between the Kotlin desktop UI in
`GUI/desktop` and the Python stack in `larrick_multi`. The bridge keeps a
single JSON contract so UI adapters can run in deterministic stub mode by
default and optionally route to real orchestration behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    """Read input JSON payload from bridge input file."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write output JSON payload for Kotlin adapters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _sanitize_json(obj: Any) -> Any:
    """Convert complex values to JSON-safe representations."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    return obj


def _build_optimize_stub(payload: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic optimization output compatible with existing UI parser."""
    point_count = 73
    theta = [idx * (360.0 / (point_count - 1)) for idx in range(point_count)]
    stroke = float(payload.get("strokeLengthMm", 100.0))
    rpm = float(payload.get("rpm", 3000.0))
    gear_ratio = float(payload.get("gearRatio", 2.0))
    journal_radius = float(payload.get("journalRadius", 5.0))
    ring_thickness = float(payload.get("ringThickness", 3.0))
    rpm_scale = min(3.0, max(0.5, rpm / 3000.0))

    displacement = []
    velocity = []
    acceleration = []
    ratio_wave = []
    journal_offset = []
    r_sun = []
    r_planet = []
    r_ring = []
    for idx, angle in enumerate(theta):
        radians = math.radians(angle)
        displacement.append((stroke / 2.0) * (1.0 - math.sin(radians)))
        velocity.append((stroke * rpm_scale) * 0.35 * math.cos(radians))
        acceleration.append(-(stroke * rpm_scale * rpm_scale) * 0.8 * math.sin(radians))
        ratio_wave.append(gear_ratio + 0.08 * math.sin(radians * 2.0))
        journal_offset.append(0.4 * journal_radius * math.sin(radians))
        r_sun.append(95.0 + gear_ratio * 5.0 + idx * 0.2)
        r_planet.append(140.0 + journal_radius * 1.5 + idx * 0.15)
        r_ring.append(360.0 + ring_thickness * 2.0 + idx * 0.35)

    # Keep legacy keys (`motion_law`, `optimal_profiles`, etc.) for parser
    # compatibility while also exposing a uniform `payload` object.
    return {
        "status": "success",
        "mode": "optimize",
        "backend": "larrick-stub",
        "execution_time": 0.05,
        "motion_law": {
            "theta_deg": theta,
            "displacement": displacement,
            "velocity": velocity,
            "acceleration": acceleration,
        },
        "optimal_profiles": {
            "r_sun": r_sun,
            "r_planet": r_planet,
            "r_ring_inner": r_ring,
            "gear_ratio": gear_ratio,
            "optimal_solution": "larrick-stub",
            "instantaneous_ratio": ratio_wave,
            "journal_offset": journal_offset,
            "accumulated_planet_angle_deg": theta[-1],
            "force_transfer_efficiency": [0.86 for _ in theta],
            "power_transfer_efficiency": [0.83 for _ in theta],
            "thermal_efficiency_curve": [0.79 for _ in theta],
            "efficiency_analysis": {"mode": "stub", "source": "larrick_gui_bridge"},
        },
        "tooth_profiles": {"sun_teeth": [], "planet_teeth": [], "ring_teeth": []},
        "fea": {
            "analysis_summary": {
                "max_stress": 172.0,
                "natural_frequencies": [118.0, 242.0, 389.0],
                "fatigue_life": 850000.0,
            }
        },
        "diagnostics": {"source": "deterministic_stub"},
        "payload": {
            "summary": {
                "status": "success",
                "point_count": point_count,
                "gear_ratio": gear_ratio,
            }
        },
    }


def _build_stub_response(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic stub response for non-optimization modes."""
    if mode == "optimize":
        return _build_optimize_stub(payload)

    return {
        "status": "success",
        "mode": mode,
        "backend": "larrick-stub",
        "execution_time": 0.01,
        "payload": {
            "echo": payload,
            "summary": {
                "message": f"{mode} stub completed",
                "timestamp_epoch_s": int(time.time()),
            },
        },
        "diagnostics": {"source": "deterministic_stub"},
    }


def _run_real_orchestrate(payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """
    Execute a real orchestration run when explicitly requested.

    This path is opt-in only and intentionally tiny-budget by default so GUI
    flows stay responsive while proving adapter integration.
    """
    try:
        from larrak2.cli.run_workflows import run_orchestrate_workflow
    except ModuleNotFoundError as exc:
        return {
            "status": "unavailable",
            "mode": "orchestrate",
            "backend": "larrick-real",
            "execution_time": 0.0,
            "error": str(exc),
            "payload": {
                "summary": {
                    "message": "Real backend unavailable (missing monorepo module import).",
                    "hint": "Activate the larrick_multi environment (editable install) so larrak2 and externals are importable.",
                }
            },
            "diagnostics": {"source": "import_guard"},
        }
    except Exception as exc:
        return {
            "status": "failed",
            "mode": "orchestrate",
            "backend": "larrick-real",
            "execution_time": 0.0,
            "error": str(exc),
            "payload": {
                "summary": {
                    "message": "Real backend import failed.",
                    "hint": "This usually indicates missing runtime data files or import-time side effects in orchestration modules.",
                }
            },
            "diagnostics": {"source": "import_exception_guard"},
        }

    outdir = output_dir / "orchestrate_real"
    args = SimpleNamespace(
        outdir=str(outdir),
        rpm=float(payload.get("rpm", 3000.0)),
        torque=float(payload.get("torque", 200.0)),
        fidelity=int(payload.get("fidelity", 1)),
        bore_mm=float(payload.get("bore_mm", 80.0)),
        stroke_mm=float(payload.get("stroke_mm", 90.0)),
        intake_port_area_m2=float(payload.get("intake_port_area_m2", 4.0e-4)),
        exhaust_port_area_m2=float(payload.get("exhaust_port_area_m2", 4.0e-4)),
        p_manifold_pa=float(payload.get("p_manifold_pa", 101325.0)),
        p_back_pa=float(payload.get("p_back_pa", 101325.0)),
        compression_ratio=float(payload.get("compression_ratio", 10.0)),
        fuel_name=str(payload.get("fuel_name", "gasoline")),
        constraint_phase=str(payload.get("constraint_phase", "downselect")),
        enforce_contract_routing=bool(payload.get("enforce_contract_routing", False)),
        seed=int(payload.get("seed", 42)),
        sim_budget=int(payload.get("sim_budget", 2)),
        batch_size=int(payload.get("batch_size", 2)),
        max_iterations=int(payload.get("max_iterations", 1)),
        truth_dispatch_mode=str(payload.get("truth_dispatch_mode", "off")),
        truth_plan=str(payload.get("truth_plan", "")),
        truth_auto_top_k=int(payload.get("truth_auto_top_k", 1)),
        truth_auto_min_uncertainty=float(payload.get("truth_auto_min_uncertainty", 0.0)),
        truth_auto_min_feasibility=float(payload.get("truth_auto_min_feasibility", 0.0)),
        truth_auto_min_pred_quantile=float(payload.get("truth_auto_min_pred_quantile", 0.0)),
        truth_records_path=str(payload.get("truth_records_path", "")),
        openfoam_backend=str(payload.get("openfoam_backend", "docker")),
        openfoam_solver=str(payload.get("openfoam_solver", "larrakEngineFoam")),
        openfoam_template=str(payload.get("openfoam_template", "")),
        openfoam_docker_image=str(payload.get("openfoam_docker_image", "")),
        thermo_symbolic_mode=str(payload.get("thermo_symbolic_mode", "warn")),
        thermo_symbolic_artifact_path=str(payload.get("thermo_symbolic_artifact_path", "")),
        stack_model_path=str(payload.get("stack_model_path", "")),
        control_backend=str(payload.get("control_backend", "file")),
        provenance_backend=str(payload.get("provenance_backend", "jsonl")),
        control_run_id=str(payload.get("control_run_id", "gui")),
        control_file_path=str(payload.get("control_file_path", "")),
        provenance_file_path=str(payload.get("provenance_file_path", "")),
        control_redis_url=str(payload.get("control_redis_url", "")),
        control_redis_key=str(payload.get("control_redis_key", "")),
        provenance_weaviate_url=str(payload.get("provenance_weaviate_url", "")),
        provenance_weaviate_key=str(payload.get("provenance_weaviate_key", "")),
    )
    try:
        rc = run_orchestrate_workflow(args)
    except ModuleNotFoundError as exc:
        return {
            "status": "unavailable",
            "mode": "orchestrate",
            "backend": "larrick-real",
            "execution_time": 0.0,
            "error": str(exc),
            "payload": {
                "summary": {
                    "message": "Real backend unavailable (missing external dependency).",
                    "missing_module": str(exc),
                    "hint": "Install larrak-* externals from requirements-external.txt / pyproject pins.",
                }
            },
            "diagnostics": {"source": "external_import_guard"},
        }
    except Exception as exc:
        return {
            "status": "failed",
            "mode": "orchestrate",
            "backend": "larrick-real",
            "execution_time": 0.0,
            "error": str(exc),
            "payload": {
                "summary": {
                    "message": "Real orchestration execution failed.",
                    "hint": "Check LARRICK_MULTI_ROOT data paths and external runtime dependencies (thermo/CEM/OpenFOAM).",
                }
            },
            "diagnostics": {"source": "run_orchestrate_workflow_exception"},
        }
    status = "success" if rc == 0 else "failed"
    return {
        "status": status,
        "mode": "orchestrate",
        "backend": "larrick-real",
        "execution_time": 0.0,
        "payload": {
            "summary": {"return_code": rc, "outdir": str(outdir), "status": status},
        },
        "diagnostics": {"source": "run_orchestrate_workflow"},
    }


def main() -> int:
    """Parse bridge arguments, dispatch mode handler, and write JSON output."""
    parser = argparse.ArgumentParser(description="Larrick GUI bridge CLI")
    parser.add_argument("--mode", required=True, choices=["optimize", "orchestrate", "simulate", "analyze", "engine_eval"])
    parser.add_argument("--input", required=True, help="Input payload JSON")
    parser.add_argument("--output", required=True, help="Output response JSON")
    parser.add_argument("--output-dir", required=True, help="Workspace/output directory")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Allow real handler for mode when implemented (otherwise stub)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    try:
        payload = _read_json(input_path)
        if args.mode == "orchestrate" and args.real:
            response = _run_real_orchestrate(payload, output_dir)
        else:
            response = _build_stub_response(args.mode, payload)
        response["execution_time"] = float(response.get("execution_time", 0.0)) or round(
            time.time() - start,
            4,
        )
        _write_json(output_path, _sanitize_json(response))
        return 0
    except Exception as exc:  # pragma: no cover - defensive boundary
        error_payload = {
            "status": "failed",
            "mode": args.mode,
            "backend": "larrick-gui-bridge",
            "execution_time": round(time.time() - start, 4),
            "error": str(exc),
            "payload": {},
            "diagnostics": {"source": "exception_handler"},
        }
        _write_json(output_path, _sanitize_json(error_payload))
        # The GUI subprocess boundary treats the output file as the primary
        # contract surface; return 0 so callers that require exit==0 can still
        # parse and display structured failure information.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
