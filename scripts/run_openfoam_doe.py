#!/usr/bin/env python3
"""Checkpointed OpenFOAM DOE runner (Docker).

This script generates a dataset for training the OpenFOAM NN surrogate by:
1) Sampling a DOE space (rpm/torque/lambda + BC + timing + port geometry)
2) Cloning a template case into a per-sample run directory
3) Generating STLs (opposed-piston rotary-valve geometry)
4) Running OpenFOAM utilities + solver inside Docker
5) Parsing `solver.log` for required metrics and appending to JSONL

Output format: JSONL (one record per line) with an embedded `_meta` object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from larrak2.adapters.docker_openfoam import DockerOpenFoam
from larrak2.adapters.openfoam import OpenFoamRunner


def _sha_dir(dir_path: Path) -> str:
    items = []
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            items.append(f"{p.relative_to(dir_path)}:{p.stat().st_size}")
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _solver_completed_successfully(log_file: Path) -> bool:
    """Check if solver completed by looking for 'End' marker in final lines.

    OpenFOAM sometimes crashes during cleanup (exit 139) after successfully
    completing the simulation and printing results. This function detects
    such cases by checking for the 'End' marker.
    """
    if not log_file.exists():
        return False
    try:
        # Read last 2KB of log file efficiently
        with log_file.open("rb") as f:
            f.seek(0, 2)  # End of file
            size = f.tell()
            read_size = min(2048, size)
            f.seek(max(0, size - read_size))
            tail = f.read().decode("utf-8", errors="ignore")
        # Check for 'End' marker on its own line (typical OpenFOAM termination)
        return "\nEnd\n" in tail or tail.strip().endswith("End")
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="Run checkpointed OpenFOAM DOE via Docker")
    p.add_argument(
        "--template",
        type=str,
        default="openfoam_templates/opposed_piston_rotary_valve_sliding_case",
    )
    p.add_argument("--outdir", type=str, default="data/openfoam_doe")
    p.add_argument("--runs-root", type=str, default="runs/openfoam_doe")
    p.add_argument("--jsonl", type=str, default="data/openfoam_doe/results.jsonl")
    p.add_argument("--checkpoint", type=str, default="data/openfoam_doe/checkpoint.json")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=10)

    # Solver steps
    p.add_argument("--solver", type=str, default="rhoPimpleFoam")
    p.add_argument("--docker-timeout-s", type=int, default=1800)
    p.add_argument(
        "--snappy", action="store_true", help="Run blockMesh + snappyHexMesh before solver"
    )

    # DOE ranges (first usable NN)
    p.add_argument("--rpm-min", type=float, default=1000.0)
    p.add_argument("--rpm-max", type=float, default=7000.0)
    p.add_argument("--torque-min", type=float, default=50.0)
    p.add_argument("--torque-max", type=float, default=400.0)
    p.add_argument("--lambda-min", type=float, default=0.6)
    p.add_argument("--lambda-max", type=float, default=1.6)

    p.add_argument("--bore-mm", type=float, default=80.0)
    p.add_argument("--stroke-mm", type=float, default=90.0)
    p.add_argument("--intake-port-area-min", type=float, default=2e-4)
    p.add_argument("--intake-port-area-max", type=float, default=8e-4)
    p.add_argument("--exhaust-port-area-min", type=float, default=2e-4)
    p.add_argument("--exhaust-port-area-max", type=float, default=8e-4)

    p.add_argument("--p-manifold-min", type=float, default=30_000.0)
    p.add_argument("--p-manifold-max", type=float, default=250_000.0)
    p.add_argument("--p-back-min", type=float, default=80_000.0)
    p.add_argument("--p-back-max", type=float, default=200_000.0)

    p.add_argument("--overlap-min", type=float, default=0.0)
    p.add_argument("--overlap-max", type=float, default=80.0)
    p.add_argument("--intake-open-min", type=float, default=-60.0)
    p.add_argument("--intake-open-max", type=float, default=20.0)
    p.add_argument("--intake-close-min", type=float, default=20.0)
    p.add_argument("--intake-close-max", type=float, default=120.0)
    p.add_argument("--exhaust-open-min", type=float, default=-120.0)
    p.add_argument("--exhaust-open-max", type=float, default=-20.0)
    p.add_argument("--exhaust-close-min", type=float, default=-20.0)
    p.add_argument("--exhaust-close-max", type=float, default=60.0)

    # controlDict placeholders
    p.add_argument("--endTime", type=float, default=0.01)
    p.add_argument("--deltaT", type=float, default=1e-4)
    p.add_argument("--writeInterval", type=int, default=100)
    p.add_argument("--metricWriteInterval", type=int, default=1)

    args = p.parse_args()

    template_dir = Path(args.template)
    outdir = Path(args.outdir)
    runs_root = Path(args.runs_root)
    jsonl_path = Path(args.jsonl)
    checkpoint_path = Path(args.checkpoint)

    outdir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    docker = DockerOpenFoam()

    tpl_hash = _sha_dir(template_dir)

    rng = np.random.default_rng(args.seed)

    runner = OpenFoamRunner(
        template_dir=template_dir,
        solver_cmd=args.solver,
        backend="docker",
    )

    # Resume checkpoint
    done = 0
    if checkpoint_path.exists():
        ck = json.loads(checkpoint_path.read_text())
        done = int(ck.get("done", 0))

    for i in range(done, args.n):
        rpm = float(rng.uniform(args.rpm_min, args.rpm_max))
        torque = float(rng.uniform(args.torque_min, args.torque_max))
        lambda_af = float(rng.uniform(args.lambda_min, args.lambda_max))

        intake_port_area_m2 = float(
            rng.uniform(args.intake_port_area_min, args.intake_port_area_max)
        )
        exhaust_port_area_m2 = float(
            rng.uniform(args.exhaust_port_area_min, args.exhaust_port_area_max)
        )

        # Sample pressures with a physically plausible direction for scavenging:
        # intake manifold pressure >= back pressure (otherwise reverse flow dominates).
        p_back_Pa = float(rng.uniform(args.p_back_min, args.p_back_max))
        p_man_lo = max(float(args.p_manifold_min), p_back_Pa)
        p_manifold_Pa = float(rng.uniform(p_man_lo, args.p_manifold_max))

        overlap_deg = float(rng.uniform(args.overlap_min, args.overlap_max))
        intake_open_deg = float(rng.uniform(args.intake_open_min, args.intake_open_max))
        intake_close_deg = float(rng.uniform(args.intake_close_min, args.intake_close_max))
        exhaust_open_deg = float(rng.uniform(args.exhaust_open_min, args.exhaust_open_max))
        exhaust_close_deg = float(rng.uniform(args.exhaust_close_min, args.exhaust_close_max))

        # Compute overset mesh motion parameters
        omega_rad_s = rpm * 2 * math.pi / 60  # rad/s from rpm
        cylinder_length = args.stroke_mm / 1000.0 * 1.5  # meters
        valve_length = args.bore_mm / 1000.0 * 0.5
        intake_valve_z = cylinder_length / 2  # center
        exhaust_left_z = valve_length / 2  # near z=0
        exhaust_right_z = cylinder_length - valve_length / 2  # near z=max

        # Compute safe locations inside the valve component meshes (but outside the valve geometry itself)
        # Valve radius = 0.3 * bore. Box radius approx 0.75 * bore.
        # Point at (0, 0.5*bore, z) is safe.
        # Points for region selection (inside valve fluid region)
        y_loc = args.bore_mm / 1000.0 * 0.5
        p_intake = f"(0 {y_loc} {intake_valve_z})"
        p_exh_l = f"(0 {y_loc} {exhaust_left_z})"
        p_exh_r = f"(0 {y_loc} {exhaust_right_z})"

        # Format for locationsInMesh: ( point1 point2 ... )
        # Zone naming is handled by setSet
        valve_mesh_locations = f"( {p_intake} {p_exh_l} {p_exh_r} )"

        params = {
            "rpm": rpm,
            "torque": torque,
            "lambda_af": lambda_af,
            "bore_mm": float(args.bore_mm),
            "stroke_mm": float(args.stroke_mm),
            "intake_port_area_m2": intake_port_area_m2,
            "exhaust_port_area_m2": exhaust_port_area_m2,
            "p_manifold_Pa": p_manifold_Pa,
            "p_back_Pa": p_back_Pa,
            "overlap_deg": overlap_deg,
            "intake_open_deg": intake_open_deg,
            "intake_close_deg": intake_close_deg,
            "exhaust_open_deg": exhaust_open_deg,
            "exhaust_close_deg": exhaust_close_deg,
            "solver_name": args.solver,
            "endTime": float(args.endTime),
            "deltaT": float(args.deltaT),
            "writeInterval": int(args.writeInterval),
            "metricWriteInterval": int(args.metricWriteInterval),
            # Overset mesh motion parameters
            "omega_rad_s": omega_rad_s,
            "intake_valve_z": intake_valve_z,
            "exhaust_left_z": exhaust_left_z,
            "exhaust_right_z": exhaust_right_z,
            "valve_mesh_locations": valve_mesh_locations,
            "intake_mesh_point": p_intake,
            "exhaust_left_mesh_point": p_exh_l,
            "exhaust_right_mesh_point": p_exh_r,
            # Sliding Mesh Params
            "interface_radius_sel": args.bore_mm / 1000.0 * 0.6
            + 0.0005,  # approx radius matching geometry script
            "intake_valve_min_z": intake_valve_z
            - valve_length * 0.6,  # Slightly larger z-range to cover full valve
            "intake_valve_max_z": intake_valve_z + valve_length * 0.6,
            "exhaust_left_min_z": exhaust_left_z - valve_length * 0.6,
            "exhaust_left_max_z": exhaust_left_z + valve_length * 0.6,
            "exhaust_right_min_z": exhaust_right_z - valve_length * 0.6,
            "exhaust_right_max_z": exhaust_right_z + valve_length * 0.6,
        }

        run_dir = runs_root / f"case_{i:06d}"
        tri_dir = run_dir / "constant" / "triSurface"

        # Clone + placeholder replacement
        runner.setup_case(run_dir, params)

        # Generate geometry STLs
        tri_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "python",
                "scripts/openfoam_geometry/gen_opposed_piston_rotary_valve.py",
                "--outdir",
                str(tri_dir),
                "--bore-mm",
                str(args.bore_mm),
                "--stroke-mm",
                str(args.stroke_mm),
                "--intake-port-area-m2",
                str(intake_port_area_m2),
                "--exhaust-port-area-m2",
                str(exhaust_port_area_m2),
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

        # Mesh and Solve
        is_overset = (run_dir / "system" / "snappyHexMeshDict.background").exists()
        code_block = 0
        error_stage = ""
        ok = True

        if is_overset:
            # 1. Background Mesh
            c, _, _ = docker.run_utility(
                utility="blockMesh",
                case_dir=run_dir,
                log_file=run_dir / "blockMesh.log",
                timeout_s=600,
            )
            if c != 0:
                ok = False
                error_stage = "blockMesh"

            if ok:
                c, _, _ = docker.run_utility(
                    utility="snappyHexMesh",
                    args=["-dict", "system/snappyHexMeshDict.background", "-overwrite"],
                    case_dir=run_dir,
                    log_file=run_dir / "snappyHexMesh.background.log",
                    timeout_s=1200,
                )
                if c != 0:
                    ok = False
                    error_stage = "snappyHexMesh.background"

            if ok:
                # Save background
                if (run_dir / "constant/polyMesh.background").exists():
                    shutil.rmtree(run_dir / "constant/polyMesh.background")
                shutil.move(run_dir / "constant/polyMesh", run_dir / "constant/polyMesh.background")

                # 2. Valve Mesh
                c, _, _ = docker.run_utility(utility="blockMesh", case_dir=run_dir, timeout_s=600)
                c, _, _ = docker.run_utility(
                    utility="snappyHexMesh",
                    args=["-dict", "system/snappyHexMeshDict.valve", "-overwrite"],
                    case_dir=run_dir,
                    log_file=run_dir / "snappyHexMesh.valve.log",
                    timeout_s=1200,
                )
                if c != 0:
                    ok = False
                    error_stage = "snappyHexMesh.valve"

            if ok:
                # Save valve
                if (run_dir / "constant/polyMesh.valves").exists():
                    shutil.rmtree(run_dir / "constant/polyMesh.valves")
                shutil.move(run_dir / "constant/polyMesh", run_dir / "constant/polyMesh.valves")

                # Restore background as master
                shutil.move(run_dir / "constant/polyMesh.background", run_dir / "constant/polyMesh")

                # 3. Merge
                # mergeMeshes requires a full case structure (system/controlDict) for the source mesh
                valve_case_dir = run_dir / "valve_mesh_case"
                if valve_case_dir.exists():
                    shutil.rmtree(valve_case_dir)
                valve_case_dir.mkdir()
                (valve_case_dir / "constant").mkdir()
                shutil.copytree(run_dir / "system", valve_case_dir / "system")
                shutil.copytree(
                    run_dir / "constant/polyMesh.valves", valve_case_dir / "constant/polyMesh"
                )

                c, _, _ = docker.run_utility(
                    utility="mergeMeshes",
                    args=[".", "valve_mesh_case", "-overwrite"],
                    case_dir=run_dir,
                    log_file=run_dir / "mergeMeshes.log",
                    timeout_s=600,
                )
                if c != 0:
                    ok = False
                    error_stage = "mergeMeshes"

                # Cleanup temp case
                if valve_case_dir.exists():
                    shutil.rmtree(valve_case_dir)

            if ok:
                # 4. Zones
                # setSet is deprecated/missing in OpenFOAM 11 Foundation; uses topoSet
                c, _, _ = docker.run_utility(
                    utility="topoSet",
                    case_dir=run_dir,
                    log_file=run_dir / "topoSet.log",
                    timeout_s=600,
                )
                if c != 0:
                    ok = False
                    error_stage = "topoSet"

            if ok:
                # 5. Fields
                c, _, _ = docker.run_utility(
                    utility="setFields",
                    case_dir=run_dir,
                    log_file=run_dir / "setFields.log",
                    timeout_s=600,
                )
                if c != 0:
                    ok = False
                    error_stage = "setFields"

        else:
            # Standard blockMesh fallback (non-overset templates)
            c, _, _ = docker.run_utility(
                utility="blockMesh",
                case_dir=run_dir,
                timeout_s=300,
                log_file=run_dir / "blockMesh.log",
            )
            if c != 0:
                ok = False
                error_stage = "blockMesh"

        # Check for Sliding Mesh (Unified Snappy)
        is_sliding = (run_dir / "system" / "snappyHexMeshDict").exists() and not is_overset
        print(f"DEBUG: is_overset={is_overset}, is_sliding={is_sliding}, ok={ok}")

        if ok and is_sliding:
            # 1. BlockMesh (already ran above? No, only in else. Move blockMesh up?)
            # Actually, standard blockMesh logic is in 'else' block above.
            # If is_overset is False, we already ran blockMesh.
            pass

            # 2. Snappy (Unified)
            if ok:
                print("DEBUG: Starting snappyHexMesh...")
                c, _, _ = docker.run_utility(
                    utility="snappyHexMesh",
                    args=["-overwrite"],
                    case_dir=run_dir,
                    log_file=run_dir / "snappyHexMesh.log",
                    timeout_s=1200,
                )
                if c != 0:
                    ok = False
                    error_stage = "snappyHexMesh"

            # 3. TopoSet (Create CellZones)
            if ok:
                c, _, _ = docker.run_utility(
                    utility="topoSet", case_dir=run_dir, log_file=run_dir / "topoSet.log"
                )
                if c != 0:
                    ok = False
                    error_stage = "topoSet"

            # 4. CreateBaffles (Split Interfaces)
            if ok and (run_dir / "system/createBafflesDict").exists():
                c, _, _ = docker.run_utility(
                    utility="createBaffles",
                    args=["-overwrite"],
                    case_dir=run_dir,
                    log_file=run_dir / "createBaffles.log",
                )
                if c != 0:
                    ok = False
                    error_stage = "createBaffles"

            # 5. CreatePatch (Make AMI)
            if ok and (run_dir / "system/createPatchDict").exists():
                # createPatch usually requires -overwrite
                c, _, _ = docker.run_utility(
                    utility="createPatch",
                    args=["-overwrite"],
                    case_dir=run_dir,
                    log_file=run_dir / "createPatch.log",
                )
                if c != 0:
                    ok = False
                    error_stage = "createPatch"

            # 6. SetFields (Initialize)
            if ok:
                c, _, _ = docker.run_utility(
                    utility="setFields", case_dir=run_dir, log_file=run_dir / "setFields.log"
                )
                if c != 0:
                    ok = False
                    error_stage = "setFields"

        # Solver execution
        code_solve = 0
        solver_log_file = run_dir / f"{params['solver_name']}.log"
        if ok:
            code_solve, _, _ = docker.run_solver(
                solver=params["solver_name"],
                case_dir=run_dir,
                timeout_s=args.docker_timeout_s,
                log_file=solver_log_file,
            )
            if code_solve != 0:
                # Check if solver actually completed despite non-zero exit code
                # (OpenFOAM sometimes crashes during cleanup after writing "End")
                if _solver_completed_successfully(solver_log_file):
                    print(
                        f"[{i}] Solver exit code {code_solve} but 'End' found - treating as success"
                    )
                else:
                    ok = False
                    error_stage = "solver"

        if not ok:
            metrics = {"error": 1.0, "stage": error_stage}
            # ok is already False
            rec = {
                **params,
                "ok": False,
                "m_air_trapped": 0.0,
                "scavenging_efficiency": 0.0,
                "residual_fraction": 0.0,
                "trapped_o2_mass": 0.0,
                "_meta": {
                    "i": i,
                    "timestamp": time.time(),
                    "template_hash": tpl_hash,
                    "run_dir": str(run_dir),
                    "stage": error_stage,
                    "exit_code": int(
                        code_solve if error_stage == "solver" else code_block
                    ),  # simplistic code reporting
                },
            }
            _append_jsonl(jsonl_path, rec)
            continue

        if args.snappy:
            code_snappy, _, _ = docker.run_utility(
                utility="snappyHexMesh",
                case_dir=run_dir,
                args=["-overwrite"],
                timeout_s=1200,
                log_file=run_dir / "snappyHexMesh.log",
            )
            if code_snappy != 0:
                metrics = {"error": 1.0}
                ok = False
                rec = {
                    **params,
                    "ok": bool(ok),
                    "m_air_trapped": 0.0,
                    "scavenging_efficiency": 0.0,
                    "residual_fraction": 0.0,
                    "trapped_o2_mass": 0.0,
                    "_meta": {
                        "i": i,
                        "timestamp": time.time(),
                        "template_hash": tpl_hash,
                        "run_dir": str(run_dir),
                        "stage": "snappyHexMesh",
                        "exit_code": int(code_snappy),
                    },
                }
                _append_jsonl(jsonl_path, rec)
                continue

            code_check, _, _ = docker.run_utility(
                utility="checkMesh",
                case_dir=run_dir,
                timeout_s=300,
                log_file=run_dir / "checkMesh.log",
            )
            if code_check != 0:
                metrics = {"error": 1.0}
                ok = False
                rec = {
                    **params,
                    "ok": bool(ok),
                    "m_air_trapped": 0.0,
                    "scavenging_efficiency": 0.0,
                    "residual_fraction": 0.0,
                    "trapped_o2_mass": 0.0,
                    "_meta": {
                        "i": i,
                        "timestamp": time.time(),
                        "template_hash": tpl_hash,
                        "run_dir": str(run_dir),
                        "stage": "checkMesh",
                        "exit_code": int(code_check),
                    },
                }
                _append_jsonl(jsonl_path, rec)
                continue

        # Parse results
        metrics = runner.parse_results(run_dir, log_name=f"{params['solver_name']}.log")
        required = {"trapped_mass", "scavenging_efficiency", "residual_fraction", "trapped_o2_mass"}
        ok = ("error" not in metrics) and required.issubset(metrics.keys())

        rec = {
            **params,
            "ok": bool(ok),
            "m_air_trapped": float(metrics.get("trapped_mass", 0.0)),
            "scavenging_efficiency": float(metrics.get("scavenging_efficiency", 0.0)),
            "residual_fraction": float(metrics.get("residual_fraction", 0.0)),
            "trapped_o2_mass": float(metrics.get("trapped_o2_mass", 0.0)),
            "_meta": {
                "i": i,
                "timestamp": time.time(),
                "template_hash": tpl_hash,
                "run_dir": str(run_dir),
            },
        }

        _append_jsonl(jsonl_path, rec)

        if (i + 1) % args.checkpoint_every == 0:
            checkpoint_path.write_text(
                json.dumps(
                    {"done": i + 1, "n": args.n, "seed": args.seed, "template_hash": tpl_hash},
                    indent=2,
                )
            )

        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{args.n}] wrote {jsonl_path}")

    checkpoint_path.write_text(
        json.dumps(
            {"done": args.n, "n": args.n, "seed": args.seed, "template_hash": tpl_hash}, indent=2
        )
    )
    print(f"Completed DOE. Results: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
