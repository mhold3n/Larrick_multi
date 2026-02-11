"""Run workflows for optimization and simulation pipelines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from larrak2.adapters.openfoam import OpenFoamRunner
from larrak2.cli.run_pareto import main as run_pareto_main
from larrak2.core.encoding import decode_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext
from larrak2.pipelines.openfoam import OpenFoamPipeline
from larrak2.promote.staged import StagedWorkflow

# ---------------------------------------------------------------------------
# Pareto Grid
# ---------------------------------------------------------------------------


def run_pareto_grid_workflow(args: argparse.Namespace) -> int:
    """Run Pareto optimization over an (rpm, torque) grid."""

    if args.rpm_list:
        rpms = [float(x.strip()) for x in args.rpm_list.split(",") if x.strip()]
    else:
        rpms = [float(x) for x in np.linspace(args.rpm_min, args.rpm_max, args.rpm_n)]

    if args.torque_list:
        torques = [float(x.strip()) for x in args.torque_list.split(",") if x.strip()]
    else:
        torques = [float(x) for x in np.linspace(args.torque_min, args.torque_max, args.torque_n)]

    out_root = Path(args.outdir_root)
    out_root.mkdir(parents=True, exist_ok=True)

    grid_records: list[dict] = []
    idx = 0

    for rpm in rpms:
        for tq in torques:
            idx += 1
            point_out = out_root / f"rpm{int(round(rpm))}_tq{int(round(tq))}"
            point_out.mkdir(parents=True, exist_ok=True)

            seed_point = args.seed + idx
            argv = [
                "--pop",
                str(args.pop),
                "--gen",
                str(args.gen),
                "--rpm",
                str(rpm),
                "--torque",
                str(tq),
                "--fidelity",
                str(args.fidelity),
                "--seed",
                str(seed_point),
                "--outdir",
                str(point_out),
                "--bore-mm",
                str(args.bore_mm),
                "--stroke-mm",
                str(args.stroke_mm),
                "--intake-port-area-m2",
                str(args.intake_port_area_m2),
                "--exhaust-port-area-m2",
                str(args.exhaust_port_area_m2),
                "--p-manifold-pa",
                str(args.p_manifold_pa),
                "--p-back-pa",
                str(args.p_back_pa),
                "--overlap-deg",
                str(args.overlap_deg),
                "--intake-open-deg",
                str(args.intake_open_deg),
                "--intake-close-deg",
                str(args.intake_close_deg),
                "--exhaust-open-deg",
                str(args.exhaust_open_deg),
                "--exhaust-close-deg",
                str(args.exhaust_close_deg),
            ]
            if args.verbose:
                argv.append("--verbose")

            exit_code = run_pareto_main(argv)
            if exit_code != 0:
                grid_records.append(
                    {"rpm": rpm, "torque": tq, "outdir": str(point_out), "ok": False}
                )
                continue

            summary_path = point_out / "summary.json"
            if not summary_path.exists():
                grid_records.append(
                    {"rpm": rpm, "torque": tq, "outdir": str(point_out), "ok": False}
                )
                continue

            summary = json.loads(summary_path.read_text())
            grid_records.append(
                {
                    "rpm": rpm,
                    "torque": tq,
                    "outdir": str(point_out),
                    "ok": True,
                    "n_pareto": summary.get("n_pareto"),
                    "feasible_fraction": summary.get("feasible_fraction"),
                    "best_eta_comb": summary.get("best_eta_comb"),
                    "best_eta_exp": summary.get("best_eta_exp"),
                    "best_eta_gear": summary.get("best_eta_gear"),
                    "best_eta_total": summary.get("best_eta_total"),
                }
            )
            print(
                f"[{idx}/{len(rpms) * len(torques)}] rpm={rpm:.0f}, torque={tq:.0f} -> {point_out}"
            )

    out_summary = {
        "config": {
            "pop": args.pop,
            "gen": args.gen,
            "fidelity": args.fidelity,
            "seed_base": args.seed,
            "rpms": rpms,
            "torques": torques,
        },
        "points": grid_records,
    }
    (out_root / "grid_summary.json").write_text(json.dumps(out_summary, indent=2))
    print(f"\nWrote grid summary to: {out_root / 'grid_summary.json'}")
    return 0


# ---------------------------------------------------------------------------
# Pareto Staged
# ---------------------------------------------------------------------------


def run_pareto_staged_workflow(args: argparse.Namespace) -> int:
    """Run multi-fidelity staged Pareto optimization."""
    from larrak2.promote.staged import StagedWorkflow

    output_dir = Path(args.outdir)
    workflow = StagedWorkflow(outdir=output_dir, rpm=args.rpm, torque=args.torque, seed=args.seed)

    t0 = time.time()

    archive_s1 = workflow.run_stage1(args.pop, args.gen)
    archive_s2 = workflow.run_promotion(archive_s1, args.promote)
    archive_s3 = workflow.run_stage3(archive_s2, args.pop, args.gen)

    t_total = time.time() - t0

    s1_vals = archive_s1.to_arrays()[1]
    s3_vals = archive_s3.to_arrays()[1]

    summary = {
        "config": vars(args),
        "metrics": {
            "total_time_s": t_total,
            "stage1_n": len(s1_vals),
            "stage2_n": len(archive_s2.to_arrays()[1]),
            "stage3_n": len(s3_vals),
            "stage1_eff_max": float(-np.min(s1_vals[:, 0])) if len(s1_vals) > 0 else 0,
            "stage3_eff_max": float(-np.min(s3_vals[:, 0])) if len(s3_vals) > 0 else 0,
        },
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWorkflow complete in {t_total:.2f}s")
    print(f"Stage 1 Max Eff: {summary['metrics']['stage1_eff_max']:.1%}")
    print(f"Stage 3 Max Eff: {summary['metrics']['stage3_eff_max']:.1%}")
    return 0


# ---------------------------------------------------------------------------
# Active Learning
# ---------------------------------------------------------------------------


def run_active_learning_workflow(args: argparse.Namespace) -> int:
    """Active learning loop: staged opt → uncertainty → truth sims."""

    outdir = Path(args.outdir)
    workflow = StagedWorkflow(outdir, args.rpm, args.torque, seed=42)

    print("\n=== Phase 1: Exploration (Fidelity 1) ===")
    bundle_s1 = workflow.run_stage1(args.pop, args.gen)

    print("\n=== Phase 2: Promotion ===")
    bundle_s2 = workflow.run_promotion(bundle_s1, args.promote)

    print("\n=== Phase 3: Refinement (Fidelity 2) ===")
    bundle_s3 = workflow.run_stage3(bundle_s2, args.pop, args.gen)

    print(f"\nAnalyzing {len(bundle_s3)} candidates for uncertainty...")

    ctx = EvalContext(rpm=args.rpm, torque=args.torque, fidelity=2, seed=42)
    candidates = []
    for i, rec in enumerate(bundle_s3.records):
        res = evaluate_candidate(rec.x, ctx)
        unc = res.diag["versions"].get("uncertainty", {})
        u_gear = unc.get("gear", 0.0)
        u_scavenge = unc.get("scavenge", 0.0)
        candidates.append(
            {
                "index": i,
                "x": rec.x,
                "f": rec.f,
                "u_gear": u_gear,
                "u_scavenge": u_scavenge,
                "u_total": u_gear + u_scavenge,
            }
        )

    candidates.sort(key=lambda c: c["u_total"], reverse=True)
    selected = candidates[: args.n_truth]

    print(f"\nSelected {len(selected)} candidates for Truth Evaluation:")
    for c in selected:
        print(
            f"  Candidate {c['index']}: Unc={c['u_total']:.4f} (G={c['u_gear']:.4f}, S={c['u_scavenge']:.4f})"
        )

    if args.dry_run:
        print("\n[Dry Run] Saving plan to truth_plan.json")
        plan = [
            {"x": item["x"].tolist(), "metrics": {"u_total": item["u_total"]}} for item in selected
        ]
        with open(outdir / "truth_plan.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        return 0

    print("\n=== Phase 4: Truth Execution ===")
    # from larrak2.adapters.calculix import CalculiXRunner

    template_foam = Path("templates/openfoam_scavenge")
    template_dummy = not template_foam.exists()
    _foam_runner = OpenFoamRunner(template_dir=template_foam)
    _ccx_runner = None

    for i, item in enumerate(selected):
        x = item["x"]
        print(f"[{i + 1}/{len(selected)}] Running OpenFOAM...")
        try:
            _cand = decode_candidate(x)
            if template_dummy:
                print("  (Template missing, skipping real Run)")
        except Exception as e:
            print(f"  OpenFOAM Failed: {e}")

        print(f"[{i + 1}/{len(selected)}] Running CalculiX...")
        # Placeholder for real CalculiX execution

    print("\nActive Learning Loop Complete.")
    return 0


# ---------------------------------------------------------------------------
# OpenFOAM DOE
# ---------------------------------------------------------------------------


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


def run_openfoam_doe_workflow(args: argparse.Namespace) -> int:
    """Checkpointed OpenFOAM DOE runner."""

    template_dir = Path(args.template)
    runs_root = Path(args.runs_root)
    jsonl_path = Path(args.jsonl)
    checkpoint_path = Path(args.checkpoint)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    pipeline = OpenFoamPipeline(
        template_dir=template_dir,
        solver_cmd=args.solver,
        docker_timeout_s=args.docker_timeout_s,
    )
    tpl_hash = _sha_dir(template_dir)
    rng = np.random.default_rng(args.seed)

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
        p_back_Pa = float(rng.uniform(args.p_back_min, args.p_back_max))
        p_man_lo = max(float(args.p_manifold_min), p_back_Pa)
        p_manifold_Pa = float(rng.uniform(p_man_lo, args.p_manifold_max))
        overlap_deg = float(rng.uniform(args.overlap_min, args.overlap_max))
        intake_open_deg = float(rng.uniform(args.intake_open_min, args.intake_open_max))
        intake_close_deg = float(rng.uniform(args.intake_close_min, args.intake_close_max))
        exhaust_open_deg = float(rng.uniform(args.exhaust_open_min, args.exhaust_open_max))
        exhaust_close_deg = float(rng.uniform(args.exhaust_close_min, args.exhaust_close_max))

        omega_rad_s = rpm * 2 * math.pi / 60
        cylinder_length = args.stroke_mm / 1000.0 * 1.5
        valve_length = args.bore_mm / 1000.0 * 0.5
        intake_valve_z = cylinder_length / 2
        exhaust_left_z = valve_length / 2
        exhaust_right_z = cylinder_length - valve_length / 2
        y_loc = args.bore_mm / 1000.0 * 0.5
        p_intake = f"(0 {y_loc} {intake_valve_z})"
        p_exh_l = f"(0 {y_loc} {exhaust_left_z})"
        p_exh_r = f"(0 {y_loc} {exhaust_right_z})"
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
            "omega_rad_s": omega_rad_s,
            "intake_valve_z": intake_valve_z,
            "exhaust_left_z": exhaust_left_z,
            "exhaust_right_z": exhaust_right_z,
            "valve_mesh_locations": valve_mesh_locations,
            "intake_mesh_point": p_intake,
            "exhaust_left_mesh_point": p_exh_l,
            "exhaust_right_mesh_point": p_exh_r,
            "interface_radius_sel": args.bore_mm / 1000.0 * 0.6 + 0.0005,
            "intake_valve_min_z": intake_valve_z - valve_length * 0.6,
            "intake_valve_max_z": intake_valve_z + valve_length * 0.6,
            "exhaust_left_min_z": exhaust_left_z - valve_length * 0.6,
            "exhaust_left_max_z": exhaust_left_z + valve_length * 0.6,
            "exhaust_right_min_z": exhaust_right_z - valve_length * 0.6,
            "exhaust_right_max_z": exhaust_right_z + valve_length * 0.6,
        }

        run_dir = runs_root / f"case_{i:06d}"
        result = pipeline.execute(
            run_dir=run_dir,
            params=params,
            geometry_args={
                "bore_mm": args.bore_mm,
                "stroke_mm": args.stroke_mm,
                "intake_port_area_m2": intake_port_area_m2,
                "exhaust_port_area_m2": exhaust_port_area_m2,
            },
        )

        rec = {
            **params,
            "ok": result["ok"],
            "m_air_trapped": result.get("trapped_mass", 0.0),
            "scavenging_efficiency": result.get("scavenging_efficiency", 0.0),
            "residual_fraction": result.get("residual_fraction", 0.0),
            "trapped_o2_mass": result.get("trapped_o2_mass", 0.0),
            "_meta": {
                "i": i,
                "timestamp": time.time(),
                "template_hash": tpl_hash,
                "run_dir": str(run_dir),
                "stage": result.get("stage", "complete"),
                "exit_code": 0 if result["ok"] else 1,
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
