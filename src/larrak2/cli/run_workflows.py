"""Run workflows for optimization and simulation pipelines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from larrak_optimization.cli.run_pareto import main as run_pareto_main
from larrak_optimization.promote import select_strict_nsga3
from larrak_runtime.architecture.contracts import (
    CONTRACT_VERSION,
    active_contract_tracer,
    get_active_contract_tracer,
)
from larrak_runtime.core.artifact_paths import (
    DEFAULT_CALCULIX_NN_ARTIFACT,
    DEFAULT_GEAR_LOSS_NN_DIR,
    DEFAULT_HIFI_SURROGATE_DIR,
    DEFAULT_OPENFOAM_NN_ARTIFACT,
    assert_not_legacy_models_path,
    assert_not_legacy_models_write,
    resolve_stack_artifact_path,
    resolve_thermo_symbolic_artifact_path,
)
from larrak_runtime.core.constraints import (
    get_constraint_kinds_for_phase,
    get_constraint_names,
    get_constraint_reasons,
    get_constraint_scales,
    get_material_constraint_names,
)
from larrak_runtime.core.encoding import N_TOTAL, decode_candidate
from larrak_runtime.core.evaluator import evaluate_candidate
from larrak_runtime.core.types import BreathingConfig, EvalContext
from larrak_simulation.pipelines.openfoam import OpenFoamPipeline

from larrak2.adapters.calculix import CalculiXRunner
from larrak2.adapters.openfoam import OpenFoamRunner
from larrak2.promote.staged import StagedWorkflow

DEFAULT_READINESS_ROOT = Path("outputs/readiness")
DEFAULT_PIPELINE_READINESS_SUMMARY = DEFAULT_READINESS_ROOT / "pipeline_readiness_summary.md"
DEFAULT_F2_BLOCKERS = DEFAULT_READINESS_ROOT / "f2_blockers.json"
DEFAULT_ARTIFACT_CONTRACT_AUDIT = DEFAULT_READINESS_ROOT / "artifact_contract_audit.json"
ALT_PIPELINE_READINESS_SUMMARY = (
    DEFAULT_READINESS_ROOT / "architecture" / "pipeline_arch_readiness_summary.md"
)

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
                "--algorithm",
                str(getattr(args, "algorithm", "nsga3")),
                "--partitions",
                str(getattr(args, "partitions", 4)),
                "--nsga3-max-ref-dirs",
                str(getattr(args, "nsga3_max_ref_dirs", 192)),
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
                "--openfoam-model-path",
                str(getattr(args, "openfoam_model_path", "")),
                "--calculix-stress-mode",
                str(getattr(args, "calculix_stress_mode", "nn")),
                "--calculix-model-path",
                str(getattr(args, "calculix_model_path", "")),
                "--gear-loss-mode",
                str(getattr(args, "gear_loss_mode", "physics")),
                "--gear-loss-model-dir",
                str(getattr(args, "gear_loss_model_dir", "")),
                "--thermo-model",
                str(getattr(args, "thermo_model", "two_zone_eq_v1")),
                "--thermo-constants-path",
                str(getattr(args, "thermo_constants_path", "")),
                "--thermo-anchor-manifest",
                str(getattr(args, "thermo_anchor_manifest", "")),
                "--tribology-scuff-method",
                str(getattr(args, "tribology_scuff_method", "auto")),
            ]
            strict_trib = getattr(args, "strict_tribology_data", None)
            if strict_trib is True:
                argv.append("--strict-tribology-data")
            elif strict_trib is False:
                argv.append("--no-strict-tribology-data")
            if bool(getattr(args, "allow_nonproduction_paths", False)):
                argv.append("--allow-nonproduction-paths")
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
    from larrak_optimization.promote.staged import StagedWorkflow

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
# Dress Rehearsal
# ---------------------------------------------------------------------------


def _sweep_values(min_v: float, max_v: float, step_v: float) -> list[float]:
    """Create an inclusive coarse sweep list."""
    if step_v <= 0:
        raise ValueError(f"Sweep step must be > 0, got {step_v}")
    if max_v < min_v:
        raise ValueError(f"Sweep max must be >= min, got min={min_v}, max={max_v}")
    n = int(math.floor((max_v - min_v) / step_v + 1e-9)) + 1
    vals = [min_v + i * step_v for i in range(n)]
    if vals[-1] < max_v - 1e-9:
        vals.append(max_v)
    return [float(v) for v in vals]


def _build_operating_conditions(args: argparse.Namespace) -> list[tuple[float, float]]:
    """Return list of (rpm, torque) operating points for dress rehearsal."""
    if not bool(getattr(args, "condition_sweep", False)):
        return [(float(args.rpm), float(args.torque))]

    rpms = _sweep_values(float(args.rpm_min), float(args.rpm_max), float(args.rpm_step))
    torques = _sweep_values(
        float(args.torque_min),
        float(args.torque_max),
        float(args.torque_step),
    )
    return [(float(r), float(t)) for r in rpms for t in torques]


def _build_eval_context_from_args(
    args: argparse.Namespace, *, rpm: float | None = None, torque: float | None = None
) -> EvalContext:
    """Build EvalContext mirroring run_pareto workflow args."""
    breathing = BreathingConfig(
        bore_mm=float(args.bore_mm),
        stroke_mm=float(args.stroke_mm),
        intake_port_area_m2=float(args.intake_port_area_m2),
        exhaust_port_area_m2=float(args.exhaust_port_area_m2),
        p_manifold_Pa=float(args.p_manifold_pa),
        p_back_Pa=float(args.p_back_pa),
        overlap_deg=float(args.overlap_deg),
        intake_open_deg=float(args.intake_open_deg),
        intake_close_deg=float(args.intake_close_deg),
        exhaust_open_deg=float(args.exhaust_open_deg),
        exhaust_close_deg=float(args.exhaust_close_deg),
        compression_ratio=float(getattr(args, "compression_ratio", 10.0)),
        fuel_name=str(getattr(args, "fuel_name", "gasoline")),
        valve_timing_mode="candidate",
    )
    return EvalContext(
        rpm=float(args.rpm if rpm is None else rpm),
        torque=float(args.torque if torque is None else torque),
        fidelity=int(args.fidelity),
        seed=int(args.seed),
        breathing=breathing,
        constraint_phase=str(args.constraint_phase),
        tolerance_constraint_mode=str(args.tolerance_constraint_mode),
        tolerance_threshold_mm=float(args.tolerance_threshold_mm),
        openfoam_model_path=str(getattr(args, "openfoam_model_path", "")).strip() or None,
        calculix_stress_mode=str(getattr(args, "calculix_stress_mode", "nn")),
        calculix_model_path=str(getattr(args, "calculix_model_path", "")).strip() or None,
        gear_loss_mode=str(getattr(args, "gear_loss_mode", "physics")),
        gear_loss_model_dir=str(getattr(args, "gear_loss_model_dir", "")).strip() or None,
        strict_data=bool(getattr(args, "strict_data", True)),
        strict_tribology_data=getattr(args, "strict_tribology_data", None),
        tribology_scuff_method=str(getattr(args, "tribology_scuff_method", "auto")),
        surrogate_validation_mode=str(getattr(args, "surrogate_validation_mode", "strict")),
        thermo_constants_path=str(getattr(args, "thermo_constants_path", "")).strip() or None,
        thermo_anchor_manifest_path=str(getattr(args, "thermo_anchor_manifest", "")).strip()
        or None,
        thermo_chemistry_profile_path=str(
            getattr(args, "thermo_chemistry_profile_path", "")
        ).strip()
        or None,
        thermo_symbolic_mode=str(getattr(args, "thermo_symbolic_mode", "strict")),
        thermo_symbolic_artifact_path=str(
            getattr(args, "thermo_symbolic_artifact_path", "")
        ).strip()
        or None,
        production_profile="strict_prod",
        allow_nonproduction_paths=bool(getattr(args, "allow_nonproduction_paths", False)),
        machining_mode=str(getattr(args, "machining_mode", "nn")),
        machining_model_path=str(getattr(args, "machining_model_path", "")).strip() or None,
    )


def _compute_margin_profiles(
    X: np.ndarray,
    ctx: EvalContext | None = None,
    contexts: list[EvalContext] | None = None,
) -> list[dict[str, float]]:
    """Evaluate candidates and compute normalized margin profiles."""
    if ctx is None and contexts is None:
        raise ValueError("Either ctx or contexts must be provided")
    if contexts is not None and len(contexts) != int(X.shape[0]):
        raise ValueError(
            f"contexts length mismatch: expected {int(X.shape[0])}, got {len(contexts)}"
        )

    rows: list[dict[str, float]] = []
    for i, x in enumerate(X):
        ctx_i = contexts[i] if contexts is not None else ctx
        assert ctx_i is not None
        res = evaluate_candidate(x, ctx_i)
        recs = res.diag.get("constraints", [])

        if recs:
            scaled_raw = np.array(
                [float(r.get("scaled_raw", r.get("scaled", 0.0))) for r in recs], dtype=float
            )
            margin = -scaled_raw
            margin_min = float(np.min(margin))
            hard_margin_min = float(
                np.min(
                    [
                        -float(r.get("scaled_raw", r.get("scaled", 0.0)))
                        for r in recs
                        if str(r.get("kind", "hard")) == "hard"
                    ]
                    or [0.0]
                )
            )
            soft_violation_sum = float(np.sum([float(r.get("soft_violation", 0.0)) for r in recs]))
        else:
            margin_min = 0.0
            hard_margin_min = 0.0
            soft_violation_sum = 0.0

        rows.append(
            {
                "index": float(i),
                "margin_min": margin_min,
                "hard_margin_min": hard_margin_min,
                "soft_violation_sum": soft_violation_sum,
                "effective_feasible": float(np.all(res.G <= 0)),
            }
        )

    return rows


def _select_promotion_indices(
    X: np.ndarray,
    F: np.ndarray,
    ctx: EvalContext,
    k: int,
    margin_min: float,
    pool_mult: int,
    contexts: list[EvalContext] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select promotion set: Pareto-rank + margin floor + geometry diversity."""
    n = int(X.shape[0])
    if n == 0 or k <= 0:
        return np.zeros(0, dtype=int), {
            "n_candidates": n,
            "n_selected": 0,
            "margin_min_required": float(margin_min),
            "applied_margin_filter": 0,
        }

    margins = _compute_margin_profiles(X, ctx=ctx, contexts=contexts)

    margin_arr = np.array([m["margin_min"] for m in margins], dtype=float)
    hard_margin_arr = np.array([m["hard_margin_min"] for m in margins], dtype=float)
    soft_arr = np.array([m["soft_violation_sum"] for m in margins], dtype=float)
    # Canonical promotion selection lives in `larrak-optimization`. The pinned
    # public API exposes strict NSGA-III selection; we apply a margin filter and
    # use soft-violation sum as a deterministic secondary objective.
    eligible = np.where(hard_margin_arr >= float(margin_min))[0]
    if eligible.size == 0:
        return np.zeros(0, dtype=int), {
            "n_candidates": n,
            "n_selected": 0,
            "margin_min_required": float(margin_min),
            "applied_margin_filter": int(n),
        }

    F_elig = np.asarray(F, dtype=float)[eligible]
    soft_elig = soft_arr[eligible].reshape(-1, 1)
    F_aug = np.concatenate([F_elig, soft_elig], axis=1)
    hashes = [hashlib.sha256(X[int(i)].tobytes()).hexdigest() for i in eligible.tolist()]
    chosen_local = select_strict_nsga3(
        F_aug,
        hashes=hashes,
        k=int(min(k, eligible.size)),
    )
    selected_arr = eligible[np.asarray(chosen_local, dtype=int)]
    meta = {
        "n_candidates": n,
        "n_selected": int(np.asarray(selected_arr).size),
        "margin_min_required": float(margin_min),
        "applied_margin_filter": int(n - eligible.size),
        "selector": "larrak_optimization.promote.select_strict_nsga3",
        "pool_mult": int(pool_mult),
    }
    # Preserve legacy metadata fields for dashboards/debugging.
    meta.setdefault("margin_min", float(margin_min))
    meta.setdefault("margin_arr_preview", [float(x) for x in margin_arr[: min(8, n)]])
    return selected_arr, meta


def _build_constraint_policy_rows(
    fidelity: int,
    tolerance_mode: str,
    tolerance_threshold_mm: float,
) -> list[dict[str, object]]:
    """Build explicit explore/downselect policy rows for artifacts."""
    names = get_constraint_names(fidelity)
    scales = get_constraint_scales()
    reasons = get_constraint_reasons()
    material_names = set(get_material_constraint_names())
    rows: list[dict[str, object]] = []

    for phase in ("explore", "downselect"):
        kinds = get_constraint_kinds_for_phase(phase)
        for name in names:
            row = {
                "phase": phase,
                "constraint": name,
                "kind": kinds.get(name, "hard"),
                "scale": float(scales.get(name, 1.0)),
                "is_material_constraint": bool(name in material_names),
                "threshold_mode": tolerance_mode if name == "tol_budget" else "",
                "threshold_mm": float(tolerance_threshold_mm) if name == "tol_budget" else None,
                "rationale": reasons.get(name, ""),
            }
            rows.append(row)
    return rows


def _summarize_npz_dataset(path: Path) -> dict[str, object]:
    """Return lightweight shape summary for npz training files."""
    if not path.exists():
        return {"exists": False}

    if path.suffix.lower() != ".npz":
        return {"exists": True, "type": path.suffix.lower(), "size_bytes": int(path.stat().st_size)}

    summary: dict[str, object] = {
        "exists": True,
        "type": "npz",
        "size_bytes": int(path.stat().st_size),
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.keys())
            summary["keys"] = keys
            shapes = {
                k: tuple(int(x) for x in np.asarray(data[k]).shape)
                for k in keys
                if hasattr(data[k], "shape")
            }
            summary["shapes"] = shapes
            if "X" in shapes and len(shapes["X"]) > 0:
                summary["n_samples"] = int(shapes["X"][0])
    except Exception as e:  # pragma: no cover - best effort logging only
        summary["load_error"] = str(e)

    return summary


def _condition_ranges(
    operating_conditions: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    rpms = [float(r) for r, _ in operating_conditions]
    torques = [float(t) for _, t in operating_conditions]
    return min(rpms), max(rpms), min(torques), max(torques)


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _filter_openfoam_jsonl_success(raw_jsonl: Path, filtered_jsonl: Path) -> dict[str, int]:
    """Keep only solver-success OpenFOAM records with finite targets."""
    from larrak2.surrogate.openfoam_nn import DEFAULT_TARGET_KEYS

    n_total = 0
    n_kept = 0
    filtered_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with filtered_jsonl.open("w", encoding="utf-8") as f_out:
        for line in raw_jsonl.read_text().splitlines():
            row = line.strip()
            if not row:
                continue
            n_total += 1
            rec = json.loads(row)
            if not bool(rec.get("ok", False)):
                continue
            if any(not _is_finite_number(rec.get(k)) for k in DEFAULT_TARGET_KEYS):
                continue
            f_out.write(json.dumps(rec) + "\n")
            n_kept += 1
    return {"n_total": n_total, "n_kept": n_kept}


def _build_openfoam_training_data(
    args: argparse.Namespace,
    *,
    outdir: Path,
    operating_conditions: list[tuple[float, float]],
    log_fn,
) -> tuple[Path, dict[str, object]]:
    """Resolve OpenFOAM training data from provided file or solver DOE."""
    openfoam_data = Path(str(args.openfoam_data)) if str(args.openfoam_data).strip() else None
    if openfoam_data is not None:
        if not openfoam_data.exists():
            raise FileNotFoundError(f"OpenFOAM dataset not found: {openfoam_data}")
        return openfoam_data, {"source": "provided"}

    template_dir = (
        Path(str(args.openfoam_template)) if str(args.openfoam_template).strip() else None
    )
    if template_dir is None:
        raise FileNotFoundError(
            "OpenFOAM dataset path not provided. Set --openfoam-data or provide --openfoam-template "
            "to generate DOE-backed training data."
        )
    if not template_dir.exists():
        raise FileNotFoundError(f"OpenFOAM template directory not found: {template_dir}")

    rpm_min, rpm_max, tq_min, tq_max = _condition_ranges(operating_conditions)
    runs_per_condition = max(1, int(args.openfoam_runs_per_condition))
    n_cases = max(2, len(operating_conditions) * runs_per_condition)

    doe_outdir = outdir / "openfoam_doe"
    raw_jsonl = doe_outdir / "results_raw.jsonl"
    filtered_jsonl = doe_outdir / "results_train.jsonl"
    checkpoint = doe_outdir / "checkpoint.json"
    runs_root = doe_outdir / "runs"

    log_fn(
        "Generating OpenFOAM DOE for surrogate training",
        template=template_dir,
        n_cases=n_cases,
        runs_per_condition=runs_per_condition,
        rpm_min=rpm_min,
        rpm_max=rpm_max,
        torque_min=tq_min,
        torque_max=tq_max,
    )

    overlap_min = max(0.0, float(args.overlap_deg) - 20.0)
    overlap_max = min(80.0, float(args.overlap_deg) + 20.0)
    intake_open_min = float(args.intake_open_deg) - 10.0
    intake_open_max = float(args.intake_open_deg) + 10.0
    intake_close_min = float(args.intake_close_deg) - 20.0
    intake_close_max = float(args.intake_close_deg) + 20.0
    exhaust_open_min = float(args.exhaust_open_deg) - 20.0
    exhaust_open_max = float(args.exhaust_open_deg) + 20.0
    exhaust_close_min = float(args.exhaust_close_deg) - 20.0
    exhaust_close_max = float(args.exhaust_close_deg) + 20.0

    doe_args = argparse.Namespace(
        template=str(template_dir),
        outdir=str(doe_outdir),
        runs_root=str(runs_root),
        jsonl=str(raw_jsonl),
        checkpoint=str(checkpoint),
        n=int(n_cases),
        seed=int(args.seed),
        checkpoint_every=max(1, min(10, int(n_cases))),
        solver=str(args.openfoam_solver),
        docker_timeout_s=int(args.openfoam_docker_timeout_s),
        snappy=False,
        rpm_min=float(rpm_min),
        rpm_max=float(rpm_max),
        torque_min=float(tq_min),
        torque_max=float(tq_max),
        lambda_min=float(args.openfoam_lambda_min),
        lambda_max=float(args.openfoam_lambda_max),
        bore_mm=float(args.bore_mm),
        stroke_mm=float(args.stroke_mm),
        intake_port_area_min=max(1e-8, float(args.intake_port_area_m2) * 0.7),
        intake_port_area_max=max(1e-8, float(args.intake_port_area_m2) * 1.3),
        exhaust_port_area_min=max(1e-8, float(args.exhaust_port_area_m2) * 0.7),
        exhaust_port_area_max=max(1e-8, float(args.exhaust_port_area_m2) * 1.3),
        p_manifold_min=max(1.0, float(args.p_manifold_pa) * 0.7),
        p_manifold_max=max(1.0, float(args.p_manifold_pa) * 1.3),
        p_back_min=max(1.0, float(args.p_back_pa) * 0.7),
        p_back_max=max(1.0, float(args.p_back_pa) * 1.3),
        overlap_min=float(overlap_min),
        overlap_max=float(max(overlap_min, overlap_max)),
        intake_open_min=float(intake_open_min),
        intake_open_max=float(max(intake_open_min, intake_open_max)),
        intake_close_min=float(intake_close_min),
        intake_close_max=float(max(intake_close_min, intake_close_max)),
        exhaust_open_min=float(exhaust_open_min),
        exhaust_open_max=float(max(exhaust_open_min, exhaust_open_max)),
        exhaust_close_min=float(exhaust_close_min),
        exhaust_close_max=float(max(exhaust_close_min, exhaust_close_max)),
        endTime=0.01,
        deltaT=1e-4,
        writeInterval=100,
        metricWriteInterval=1,
    )

    rc = run_openfoam_doe_workflow(doe_args)
    if rc != 0:
        raise RuntimeError(f"OpenFOAM DOE failed with exit code {rc}")

    counts = _filter_openfoam_jsonl_success(raw_jsonl, filtered_jsonl)
    if counts["n_kept"] < 2:
        raise RuntimeError(
            "OpenFOAM DOE completed but produced insufficient successful samples "
            f"({counts['n_kept']}/{counts['n_total']})."
        )
    log_fn(
        "Prepared OpenFOAM training dataset",
        raw_jsonl=raw_jsonl,
        filtered_jsonl=filtered_jsonl,
        n_total=counts["n_total"],
        n_kept=counts["n_kept"],
    )
    return filtered_jsonl, {
        "source": "doe_generated",
        "doe_outdir": str(doe_outdir),
        "raw_jsonl": str(raw_jsonl),
        "filtered_jsonl": str(filtered_jsonl),
        "n_total_cases": int(counts["n_total"]),
        "n_success_cases": int(counts["n_kept"]),
    }


def _run_calculix_doe_dataset(
    *,
    template_path: Path,
    solver_cmd: str,
    n_cases: int,
    seed: int,
    rpm_min: float,
    rpm_max: float,
    torque_min: float,
    torque_max: float,
    outdir: Path,
) -> tuple[Path, dict[str, object]]:
    from larrak2.surrogate.calculix_nn import DEFAULT_FEATURE_KEYS as CCX_FEATURE_KEYS

    rng = np.random.default_rng(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    runs_root = outdir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "results.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    runner = CalculiXRunner(template_path=template_path, solver_cmd=solver_cmd)
    X_rows: list[list[float]] = []
    Y_rows: list[list[float]] = []

    def rand_between(a: float, b: float) -> float:
        if abs(a - b) < 1e-12:
            return float(a)
        return float(rng.uniform(a, b))

    for i in range(int(n_cases)):
        rpm = rand_between(rpm_min, rpm_max)
        torque = rand_between(torque_min, torque_max)
        base_radius_mm = rand_between(30.0, 120.0)
        face_width_mm = rand_between(8.0, 40.0)
        module_mm = rand_between(1.0, 6.0)
        pressure_angle_deg = rand_between(14.5, 30.0)
        helix_angle_deg = rand_between(-20.0, 20.0)
        profile_shift = rand_between(-0.5, 0.5)

        features = {
            "rpm": rpm,
            "torque": torque,
            "base_radius_mm": base_radius_mm,
            "face_width_mm": face_width_mm,
            "module_mm": module_mm,
            "pressure_angle_deg": pressure_angle_deg,
            "helix_angle_deg": helix_angle_deg,
            "profile_shift": profile_shift,
        }
        params = {
            **features,
            "base_radius": base_radius_mm,
            "face_width": face_width_mm,
            "module": module_mm,
        }
        run_dir = runs_root / f"case_{i:06d}"
        job_name = f"job_{i:06d}"

        result = runner.execute(run_dir=run_dir, job_name=job_name, params=params)
        max_stress = result.get("max_stress")
        ok = _is_finite_number(max_stress)

        rec = {
            **features,
            "ok": bool(ok),
            "max_stress": float(max_stress) if ok else None,
            "_meta": {"i": i, "run_dir": str(run_dir), "job_name": job_name},
        }
        _append_jsonl(jsonl_path, rec)
        if ok:
            X_rows.append([float(features[k]) for k in CCX_FEATURE_KEYS])
            Y_rows.append([float(max_stress)])

    if len(X_rows) < 2:
        raise RuntimeError(
            f"CalculiX DOE produced insufficient successful samples ({len(X_rows)}/{n_cases})."
        )

    train_path = outdir / "train.npz"
    np.savez(
        train_path,
        X=np.asarray(X_rows, dtype=np.float64),
        Y=np.asarray(Y_rows, dtype=np.float64),
    )
    return train_path, {
        "doe_outdir": str(outdir),
        "jsonl": str(jsonl_path),
        "n_total_cases": int(n_cases),
        "n_success_cases": int(len(X_rows)),
    }


def _build_calculix_training_data(
    args: argparse.Namespace,
    *,
    outdir: Path,
    operating_conditions: list[tuple[float, float]],
    log_fn,
) -> tuple[Path, dict[str, object]]:
    """Resolve CalculiX training data from provided file or solver DOE."""
    calculix_data = Path(str(args.calculix_data)) if str(args.calculix_data).strip() else None
    if calculix_data is not None:
        if not calculix_data.exists():
            raise FileNotFoundError(f"CalculiX dataset not found: {calculix_data}")
        return calculix_data, {"source": "provided"}

    template_path = (
        Path(str(args.calculix_template)) if str(args.calculix_template).strip() else None
    )
    if template_path is None:
        raise FileNotFoundError(
            "CalculiX dataset path not provided. Set --calculix-data or provide "
            "--calculix-template to generate DOE-backed training data."
        )
    if not template_path.exists():
        raise FileNotFoundError(f"CalculiX template file not found: {template_path}")

    rpm_min, rpm_max, tq_min, tq_max = _condition_ranges(operating_conditions)
    runs_per_condition = max(1, int(args.calculix_runs_per_condition))
    n_cases = max(2, len(operating_conditions) * runs_per_condition)
    doe_outdir = outdir / "calculix_doe"

    log_fn(
        "Generating CalculiX DOE for surrogate training",
        template=template_path,
        n_cases=n_cases,
        runs_per_condition=runs_per_condition,
        rpm_min=rpm_min,
        rpm_max=rpm_max,
        torque_min=tq_min,
        torque_max=tq_max,
    )
    train_path, meta = _run_calculix_doe_dataset(
        template_path=template_path,
        solver_cmd=str(args.calculix_solver),
        n_cases=n_cases,
        seed=int(args.seed),
        rpm_min=float(rpm_min),
        rpm_max=float(rpm_max),
        torque_min=float(tq_min),
        torque_max=float(tq_max),
        outdir=doe_outdir,
    )
    log_fn(
        "Prepared CalculiX training dataset",
        path=train_path,
        n_total=meta["n_total_cases"],
        n_kept=meta["n_success_cases"],
    )
    return train_path, {"source": "doe_generated", **meta}


def _resolve_path_from_arg_or_env(
    arg_value: object,
    *,
    env_key: str,
    default_path: str,
) -> Path:
    """Resolve a filesystem path from CLI arg, then env var, then default."""
    arg_str = str(arg_value).strip() if isinstance(arg_value, str) else ""
    if arg_str:
        return assert_not_legacy_models_path(arg_str, purpose=env_key)

    env_str = str(os.environ.get(env_key, "")).strip()
    if env_str:
        return assert_not_legacy_models_path(env_str, purpose=env_key)

    return assert_not_legacy_models_path(default_path, purpose=env_key)


def run_train_surrogates_workflow(args: argparse.Namespace) -> int:
    """Train OpenFOAM + CalculiX NN surrogates as a standalone pre-job."""
    from larrak_simulation.training.workflows import (
        train_calculix_workflow,
        train_openfoam_workflow,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "surrogate_training_manifest.json"
    log_path = outdir / "surrogate_training.log"
    log_path.write_text("")

    manifest: dict[str, object] = {
        "workflow": "train_surrogates",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "steps": {},
        "log_path": str(log_path),
    }

    def _log(message: str, *, level: str = "INFO", **fields: object) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} [{level}] {message}"
        if fields:
            rendered = ", ".join(f"{k}={fields[k]}" for k in sorted(fields))
            line = f"{line} | {rendered}"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    def _write_manifest() -> None:
        manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path.write_text(json.dumps(manifest, indent=2))

    _log("Surrogate training job started", outdir=outdir, seed=args.seed)

    try:
        operating_conditions = _build_operating_conditions(args)
    except Exception as e:
        manifest["steps"]["operating_conditions"] = {"ok": False, "error": str(e)}
        manifest["ready_for_dress_rehearsal"] = False
        _write_manifest()
        _log("Step failed: operating_conditions", level="ERROR", error=str(e))
        print(f"Surrogate training failed while preparing operating conditions: {e}")
        return 1

    manifest["operating_conditions"] = [
        {"rpm": float(r), "torque": float(t)} for r, t in operating_conditions
    ]
    _log(
        "Prepared operating-condition set",
        mode="sweep" if bool(getattr(args, "condition_sweep", False)) else "single",
        n_conditions=len(operating_conditions),
        rpm_min=min(r for r, _ in operating_conditions),
        rpm_max=max(r for r, _ in operating_conditions),
        torque_min=min(t for _, t in operating_conditions),
        torque_max=max(t for _, t in operating_conditions),
    )
    _write_manifest()

    t0 = time.perf_counter()
    openfoam_outdir = assert_not_legacy_models_write(
        args.openfoam_outdir,
        purpose="OpenFOAM surrogate artifact output",
    )
    openfoam_outdir.mkdir(parents=True, exist_ok=True)
    calculix_outdir = assert_not_legacy_models_write(
        args.calculix_outdir,
        purpose="CalculiX surrogate artifact output",
    )
    calculix_outdir.mkdir(parents=True, exist_ok=True)
    _log(
        "Step started: train_nn_surrogates",
        openfoam_outdir=openfoam_outdir,
        calculix_outdir=calculix_outdir,
    )
    try:
        openfoam_data, openfoam_data_meta = _build_openfoam_training_data(
            args,
            outdir=outdir,
            operating_conditions=operating_conditions,
            log_fn=_log,
        )
        effective_openfoam_outdir = openfoam_outdir
        if str(
            openfoam_data_meta.get("source", "")
        ).strip() == "doe_generated" and openfoam_outdir.resolve(
            strict=False
        ) == DEFAULT_OPENFOAM_NN_ARTIFACT.parent.resolve(strict=False):
            effective_openfoam_outdir = outdir / "openfoam_authority_artifact"
            effective_openfoam_outdir.mkdir(parents=True, exist_ok=True)
            _log(
                "Redirecting DOE-trained OpenFOAM artifact output to staged directory",
                requested_outdir=openfoam_outdir,
                effective_outdir=effective_openfoam_outdir,
            )
        _log("Dataset summary: openfoam", summary=_summarize_npz_dataset(openfoam_data))
        _log("Training OpenFOAM surrogate", data=openfoam_data, epochs=args.openfoam_epochs)
        t_train = time.perf_counter()
        openfoam_summary = train_openfoam_workflow(
            argparse.Namespace(
                data=str(openfoam_data),
                outdir=str(effective_openfoam_outdir),
                seed=args.seed,
                epochs=args.openfoam_epochs,
                lr=args.openfoam_lr,
                hidden=args.openfoam_hidden,
                weight_decay=args.openfoam_weight_decay,
                name=args.openfoam_name,
                data_provenance_kind=(
                    str(getattr(args, "openfoam_data_provenance_kind", "")).strip()
                    or (
                        "doe_generated"
                        if str(openfoam_data_meta.get("source", "")).strip() == "doe_generated"
                        else "synthetic_rehearsal"
                    )
                ),
                authoritative_for_strict_f2=bool(
                    getattr(args, "openfoam_authoritative_for_strict_f2", False)
                ),
                anchor_manifest=str(getattr(args, "openfoam_anchor_manifest", "")).strip(),
                truth_source_summary=(
                    str(getattr(args, "openfoam_truth_source_summary", "")).strip()
                    or json.dumps(openfoam_data_meta, sort_keys=True)
                ),
                authority_bundle_root=(
                    str(getattr(args, "openfoam_authority_bundle_root", "")).strip()
                    or "outputs/openfoam_authority"
                ),
                source_metadata_json=json.dumps(openfoam_data_meta, sort_keys=True),
                doe_template_path=str(getattr(args, "openfoam_template", "")).strip(),
            )
        )
        _log("Trained OpenFOAM surrogate", elapsed_s=round(time.perf_counter() - t_train, 3))
        openfoam_model = effective_openfoam_outdir / args.openfoam_name
        os.environ["LARRAK2_OPENFOAM_NN_PATH"] = str(openfoam_model)

        calculix_data, calculix_data_meta = _build_calculix_training_data(
            args,
            outdir=outdir,
            operating_conditions=operating_conditions,
            log_fn=_log,
        )
        _log("Dataset summary: calculix", summary=_summarize_npz_dataset(calculix_data))
        _log("Training CalculiX surrogate", data=calculix_data, epochs=args.calculix_epochs)
        t_train = time.perf_counter()
        train_calculix_workflow(
            argparse.Namespace(
                data=str(calculix_data),
                outdir=str(calculix_outdir),
                seed=args.seed,
                epochs=args.calculix_epochs,
                lr=args.calculix_lr,
                hidden=args.calculix_hidden,
                weight_decay=args.calculix_weight_decay,
                name=args.calculix_name,
            )
        )
        _log("Trained CalculiX surrogate", elapsed_s=round(time.perf_counter() - t_train, 3))
        calculix_model = calculix_outdir / args.calculix_name
        os.environ["LARRAK2_CALCULIX_NN_PATH"] = str(calculix_model)

        from larrak2.surrogate import calculix_nn as _calculix_nn
        from larrak2.surrogate import openfoam_nn as _openfoam_nn

        _openfoam_nn._OPENFOAM_SURROGATE.clear()
        _calculix_nn._CALCULIX_SURROGATE.clear()

        manifest["steps"]["train_nn_surrogates"] = {
            "ok": True,
            "elapsed_s": time.perf_counter() - t0,
            "openfoam_model": str(openfoam_model),
            "openfoam_data": str(openfoam_data),
            "openfoam_data_meta": openfoam_data_meta,
            "openfoam_authority_bundle": (
                dict(openfoam_summary.get("authority_bundle", {}))
                if isinstance(openfoam_summary, dict)
                else {}
            ),
            "calculix_model": str(calculix_model),
            "calculix_data": str(calculix_data),
            "calculix_data_meta": calculix_data_meta,
        }
        manifest["openfoam_strict_f2_promotable"] = bool(
            manifest["steps"]["train_nn_surrogates"]["openfoam_authority_bundle"].get(
                "promotable", False
            )  # type: ignore[index]
        )
        _write_manifest()
        _log(
            "Step completed: train_nn_surrogates",
            elapsed_s=round(float(manifest["steps"]["train_nn_surrogates"]["elapsed_s"]), 3),  # type: ignore[index]
            openfoam_model=openfoam_model,
            calculix_model=calculix_model,
        )
    except Exception as e:
        err_tb = traceback.format_exc()
        manifest["steps"]["train_nn_surrogates"] = {
            "ok": False,
            "elapsed_s": time.perf_counter() - t0,
            "error": str(e),
            "traceback": err_tb,
        }
        manifest["openfoam_strict_f2_promotable"] = False
        manifest["ready_for_dress_rehearsal"] = False
        _write_manifest()
        _log("Step failed: train_nn_surrogates", level="ERROR", error=str(e))
        _log(err_tb.rstrip(), level="ERROR")
        print(f"Surrogate training failed: {e}")
        return 1

    ready = bool(manifest["steps"]["train_nn_surrogates"]["ok"])  # type: ignore[index]
    manifest["ready_for_dress_rehearsal"] = ready
    _write_manifest()
    _log(
        "Surrogate training finished",
        gate_status="PASS" if ready else "FAIL",
        ready_for_dress_rehearsal=ready,
    )
    print(f"Surrogate training complete. Gate status: {'PASS' if ready else 'FAIL'}")
    print(f"Manifest: {manifest_path}")
    return 0 if ready else 2


def run_train_stack_surrogate_workflow(args: argparse.Namespace) -> int:
    """Train global stack surrogate artifact used by symbolic CasADi refinement."""
    from larrak_simulation.training.workflows import train_stack_surrogate_workflow

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "train_stack_surrogate_manifest.json"

    manifest: dict[str, object] = {
        "workflow": "train_stack_surrogate",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "ok": False,
    }

    try:
        summary = train_stack_surrogate_workflow(args)
        manifest["ok"] = True
        manifest["summary"] = summary
        manifest["artifact_path"] = str(summary.get("artifact_path", ""))
        print("Stack surrogate training complete.")
        print(f"Artifact: {manifest['artifact_path']}")
    except Exception as exc:
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        print(f"Stack surrogate training failed: {exc}")
        manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest_path}")
        return 1

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0


def run_train_thermo_symbolic_workflow(args: argparse.Namespace) -> int:
    """Train thermo symbolic surrogate artifact used by CasADi thermo overlay."""
    from larrak_simulation.training.workflows import train_thermo_symbolic_workflow

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "train_thermo_symbolic_manifest.json"
    manifest: dict[str, object] = {
        "workflow": "train_thermo_symbolic",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "ok": False,
    }
    try:
        summary = train_thermo_symbolic_workflow(args)
        manifest["ok"] = True
        manifest["summary"] = summary
        manifest["artifact_path"] = str(summary.get("artifact_path", ""))
        print("Thermo symbolic training complete.")
        print(f"Artifact: {manifest['artifact_path']}")
    except Exception as exc:
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        print(f"Thermo symbolic training failed: {exc}")
        manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest_path}")
        return 1

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0


def run_dress_rehearsal_workflow(args: argparse.Namespace) -> int:
    """Verify surrogate readiness, run optimization, and finish CEM validation."""
    from larrak2.cli.validate import format_report, validate_candidates

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "dress_rehearsal_manifest.json"
    log_path = outdir / "dress_rehearsal.log"
    log_path.write_text("")

    manifest: dict[str, object] = {
        "workflow": "dress_rehearsal",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "steps": {},
        "log_path": str(log_path),
    }

    def _log(message: str, *, level: str = "INFO", **fields: object) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} [{level}] {message}"
        if fields:
            rendered = ", ".join(f"{k}={fields[k]}" for k in sorted(fields))
            line = f"{line} | {rendered}"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

    def _write_manifest() -> None:
        manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path.write_text(json.dumps(manifest, indent=2))

    _log(
        "Dress rehearsal started",
        outdir=outdir,
        pop=args.pop,
        gen=args.gen,
        fidelity=args.fidelity,
        rpm=args.rpm,
        torque=args.torque,
        constraint_phase=args.constraint_phase,
    )
    try:
        operating_conditions = _build_operating_conditions(args)
    except Exception as e:
        manifest["ready_for_quality_analysis"] = False
        manifest["steps"]["operating_conditions"] = {"ok": False, "error": str(e)}
        _write_manifest()
        _log("Step failed: operating_conditions", level="ERROR", error=str(e))
        print(f"Dress rehearsal failed while preparing operating conditions: {e}")
        return 1

    manifest["operating_conditions"] = [
        {"rpm": float(r), "torque": float(t)} for r, t in operating_conditions
    ]
    _log(
        "Prepared operating-condition set",
        mode="sweep" if bool(getattr(args, "condition_sweep", False)) else "single",
        n_conditions=len(operating_conditions),
        rpm_min=min(r for r, _ in operating_conditions),
        rpm_max=max(r for r, _ in operating_conditions),
        torque_min=min(t for _, t in operating_conditions),
        torque_max=max(t for _, t in operating_conditions),
    )
    policy_rows = _build_constraint_policy_rows(
        fidelity=int(args.fidelity),
        tolerance_mode=str(args.tolerance_constraint_mode),
        tolerance_threshold_mm=float(args.tolerance_threshold_mm),
    )
    policy_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "constraint_phase_default": str(args.constraint_phase),
        "rows": policy_rows,
    }
    policy_path = outdir / "constraint_policy_table.json"
    policy_path.write_text(json.dumps(policy_payload, indent=2))
    manifest["constraint_policy_table"] = str(policy_path)
    _log("Wrote constraint policy table", path=policy_path, rows=len(policy_rows))
    _write_manifest()

    # 1) Verify required surrogate artifacts (training is a separate pre-job).
    t0 = time.perf_counter()
    _log(
        "Step started: verify_surrogates",
        fidelity=args.fidelity,
        calculix_stress_mode=args.calculix_stress_mode,
        gear_loss_mode=args.gear_loss_mode,
    )
    try:
        verify_payload: dict[str, object] = {
            "fidelity": int(args.fidelity),
            "calculix_stress_mode": str(args.calculix_stress_mode),
            "gear_loss_mode": str(args.gear_loss_mode),
        }

        if int(args.fidelity) >= 2:
            openfoam_model = _resolve_path_from_arg_or_env(
                getattr(args, "openfoam_model_path", ""),
                env_key="LARRAK2_OPENFOAM_NN_PATH",
                default_path=str(DEFAULT_OPENFOAM_NN_ARTIFACT),
            )
            if not openfoam_model.exists():
                raise FileNotFoundError(
                    "OpenFOAM NN surrogate is required for fidelity>=2 but was not found at "
                    f"'{openfoam_model}'. Run `larrak-run train-surrogates` first."
                )
            os.environ["LARRAK2_OPENFOAM_NN_PATH"] = str(openfoam_model)
            verify_payload["openfoam_model"] = str(openfoam_model)
        else:
            verify_payload["openfoam_model"] = "not_required_for_fidelity<2"

        calc_mode = str(args.calculix_stress_mode)
        if calc_mode == "nn":
            calculix_model = _resolve_path_from_arg_or_env(
                getattr(args, "calculix_model_path", ""),
                env_key="LARRAK2_CALCULIX_NN_PATH",
                default_path=str(DEFAULT_CALCULIX_NN_ARTIFACT),
            )
            if not calculix_model.exists():
                raise FileNotFoundError(
                    "CalculiX NN surrogate is required in nn mode but was not found at "
                    f"'{calculix_model}'. Run `larrak-run train-surrogates` first or use "
                    "`--calculix-stress-mode analytical`."
                )
            os.environ["LARRAK2_CALCULIX_NN_PATH"] = str(calculix_model)
            verify_payload["calculix_model"] = str(calculix_model)
        elif calc_mode == "analytical":
            verify_payload["calculix_model"] = "not_required_in_analytical_mode"
        else:
            raise ValueError(f"Unsupported calculix_stress_mode '{calc_mode}'")

        gear_loss_mode = str(args.gear_loss_mode)
        if gear_loss_mode == "nn":
            gear_loss_model_dir = _resolve_path_from_arg_or_env(
                getattr(args, "gear_loss_model_dir", ""),
                env_key="LARRAK2_GEAR_LOSS_NN_DIR",
                default_path=str(DEFAULT_GEAR_LOSS_NN_DIR),
            )
            if not gear_loss_model_dir.exists():
                raise FileNotFoundError(
                    "Gear-loss NN mode selected but model directory was not found at "
                    f"'{gear_loss_model_dir}'. Train the model first or set "
                    "`--gear-loss-mode physics`."
                )
            os.environ["LARRAK2_GEAR_LOSS_NN_DIR"] = str(gear_loss_model_dir)
            verify_payload["gear_loss_model_dir"] = str(gear_loss_model_dir)
        elif gear_loss_mode == "physics":
            verify_payload["gear_loss_model_dir"] = "not_required_in_physics_mode"
        else:
            raise ValueError(f"Unsupported gear_loss_mode '{gear_loss_mode}'")

        manifest["steps"]["verify_surrogates"] = {
            "ok": True,
            "elapsed_s": time.perf_counter() - t0,
            **verify_payload,
        }
        _write_manifest()
        _log(
            "Step completed: verify_surrogates",
            elapsed_s=round(float(manifest["steps"]["verify_surrogates"]["elapsed_s"]), 3),  # type: ignore[index]
            openfoam_model=verify_payload.get("openfoam_model", ""),
            calculix_model=verify_payload.get("calculix_model", ""),
            gear_loss_model_dir=verify_payload.get("gear_loss_model_dir", ""),
        )
    except Exception as e:
        err_tb = traceback.format_exc()
        manifest["steps"]["verify_surrogates"] = {
            "ok": False,
            "elapsed_s": time.perf_counter() - t0,
            "error": str(e),
            "traceback": err_tb,
        }
        manifest["ready_for_quality_analysis"] = False
        _write_manifest()
        _log("Step failed: verify_surrogates", level="ERROR", error=str(e))
        _log(err_tb.rstrip(), level="ERROR")
        print(f"Dress rehearsal failed during surrogate verification: {e}")
        return 1

    # 2) Run unit tests (required by dress-rehearsal contract unless skipped).
    t0 = time.perf_counter()
    if bool(getattr(args, "run_unit_tests", True)):
        pytest_cmd = [sys.executable, "-m", "pytest"]
        if str(getattr(args, "pytest_args", "")).strip():
            pytest_cmd.extend(shlex.split(str(args.pytest_args)))
        pytest_cmd.append(str(args.pytest_target))

        _log("Step started: unit_tests", command=" ".join(pytest_cmd))
        proc = subprocess.run(
            pytest_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout_tail = "\n".join(proc.stdout.splitlines()[-40:])
        stderr_tail = "\n".join(proc.stderr.splitlines()[-40:])
        manifest["steps"]["unit_tests"] = {
            "ok": proc.returncode == 0,
            "elapsed_s": time.perf_counter() - t0,
            "command": pytest_cmd,
            "return_code": int(proc.returncode),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_manifest()
        if proc.returncode != 0:
            manifest["ready_for_quality_analysis"] = False
            _write_manifest()
            _log(
                "Step failed: unit_tests",
                level="ERROR",
                return_code=proc.returncode,
            )
            if stdout_tail:
                _log(stdout_tail.rstrip(), level="ERROR")
            if stderr_tail:
                _log(stderr_tail.rstrip(), level="ERROR")
            print("Dress rehearsal failed during unit tests.")
            return 1

        _log(
            "Step completed: unit_tests",
            elapsed_s=round(float(manifest["steps"]["unit_tests"]["elapsed_s"]), 3),  # type: ignore[index]
            return_code=proc.returncode,
        )
    else:
        manifest["steps"]["unit_tests"] = {
            "ok": True,
            "skipped": True,
            "elapsed_s": time.perf_counter() - t0,
        }
        _write_manifest()
        _log("Step skipped: unit_tests", reason="--skip-unit-tests")

    # 3) Run optimization across operating conditions.
    t0 = time.perf_counter()
    opt_outdir = outdir / "optimization"
    opt_outdir.mkdir(parents=True, exist_ok=True)
    _log(
        "Step started: optimization",
        outdir=opt_outdir,
        pop=args.pop,
        gen=args.gen,
        fidelity=args.fidelity,
        constraint_phase=args.constraint_phase,
        calculix_stress_mode=args.calculix_stress_mode,
        gear_loss_mode=args.gear_loss_mode,
        openfoam_model_path=args.openfoam_model_path or "env/default",
        calculix_model_path=args.calculix_model_path or "env/default",
    )
    condition_results: list[dict[str, object]] = []
    pareto_X_chunks: list[np.ndarray] = []
    pareto_F_chunks: list[np.ndarray] = []
    pareto_G_chunks: list[np.ndarray] = []
    final_X_chunks: list[np.ndarray] = []
    final_F_chunks: list[np.ndarray] = []
    final_G_chunks: list[np.ndarray] = []
    pareto_contexts: list[EvalContext] = []
    final_contexts: list[EvalContext] = []
    pareto_condition_index: list[int] = []
    final_condition_index: list[int] = []
    objective_dim = 0

    for i, (rpm_i, torque_i) in enumerate(operating_conditions):
        cond_dir = opt_outdir / f"rpm{int(round(rpm_i))}_tq{int(round(torque_i))}"
        cond_dir.mkdir(parents=True, exist_ok=True)
        seed_i = int(args.seed) + i

        pareto_argv = [
            "--pop",
            str(args.pop),
            "--gen",
            str(args.gen),
            "--algorithm",
            str(getattr(args, "algorithm", "nsga3")),
            "--partitions",
            str(getattr(args, "partitions", 4)),
            "--nsga3-max-ref-dirs",
            str(getattr(args, "nsga3_max_ref_dirs", 192)),
            "--rpm",
            str(rpm_i),
            "--torque",
            str(torque_i),
            "--fidelity",
            str(args.fidelity),
            "--constraint-phase",
            str(args.constraint_phase),
            "--tolerance-constraint-mode",
            str(args.tolerance_constraint_mode),
            "--tolerance-threshold-mm",
            str(args.tolerance_threshold_mm),
            "--seed",
            str(seed_i),
            "--outdir",
            str(cond_dir),
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
            "--openfoam-model-path",
            str(args.openfoam_model_path),
            "--calculix-stress-mode",
            str(args.calculix_stress_mode),
            "--calculix-model-path",
            str(args.calculix_model_path),
            "--gear-loss-mode",
            str(args.gear_loss_mode),
            "--gear-loss-model-dir",
            str(args.gear_loss_model_dir),
        ]
        if args.verbose:
            pareto_argv.append("--verbose")
        if bool(getattr(args, "allow_nonproduction_paths", False)):
            pareto_argv.append("--allow-nonproduction-paths")
        _log(
            "Invoking run_pareto",
            index=i,
            rpm=rpm_i,
            torque=torque_i,
            argv=" ".join(pareto_argv),
        )

        opt_code = run_pareto_main(pareto_argv)
        if opt_code != 0:
            condition_results.append(
                {
                    "index": int(i),
                    "rpm": float(rpm_i),
                    "torque": float(torque_i),
                    "ok": False,
                    "error": f"run_pareto exit code {opt_code}",
                    "outdir": str(cond_dir),
                }
            )
            continue

        pX = np.load(cond_dir / "pareto_X.npy")
        pF = np.load(cond_dir / "pareto_F.npy")
        pG = np.load(cond_dir / "pareto_G.npy")
        fX = np.load(cond_dir / "final_pop_X.npy")
        fF = np.load(cond_dir / "final_pop_F.npy")
        fG = np.load(cond_dir / "final_pop_G.npy")
        objective_dim = max(objective_dim, int(pF.shape[1] if pF.ndim == 2 else 0))
        objective_dim = max(objective_dim, int(fF.shape[1] if fF.ndim == 2 else 0))

        cond_ctx = _build_eval_context_from_args(args, rpm=float(rpm_i), torque=float(torque_i))
        if pX.size > 0:
            pareto_X_chunks.append(pX)
            pareto_F_chunks.append(pF)
            pareto_G_chunks.append(pG)
            pareto_contexts.extend([cond_ctx] * int(pX.shape[0]))
            pareto_condition_index.extend([int(i)] * int(pX.shape[0]))
        if fX.size > 0:
            final_X_chunks.append(fX)
            final_F_chunks.append(fF)
            final_G_chunks.append(fG)
            final_contexts.extend([cond_ctx] * int(fX.shape[0]))
            final_condition_index.extend([int(i)] * int(fX.shape[0]))

        cond_summary_path = cond_dir / "summary.json"
        cond_summary = (
            json.loads(cond_summary_path.read_text()) if cond_summary_path.exists() else {}
        )
        condition_results.append(
            {
                "index": int(i),
                "rpm": float(rpm_i),
                "torque": float(torque_i),
                "ok": True,
                "outdir": str(cond_dir),
                "n_pareto": int(pX.shape[0]),
                "n_final_pop": int(fX.shape[0]),
                "summary": cond_summary,
            }
        )

    failed_conditions = [r for r in condition_results if not bool(r.get("ok", False))]
    if failed_conditions:
        manifest["steps"]["optimization"] = {
            "ok": False,
            "elapsed_s": time.perf_counter() - t0,
            "outdir": str(opt_outdir),
            "conditions": condition_results,
            "n_failed_conditions": len(failed_conditions),
        }
        manifest["ready_for_quality_analysis"] = False
        _write_manifest()
        _log(
            "Step failed: optimization",
            level="ERROR",
            n_failed_conditions=len(failed_conditions),
        )
        print("Dress rehearsal failed during optimization condition sweep.")
        return 1

    if objective_dim <= 0:
        objective_dim = 6

    X = (
        np.vstack(pareto_X_chunks)
        if pareto_X_chunks
        else np.array([], dtype=np.float64).reshape(0, N_TOTAL)
    )
    F = (
        np.vstack(pareto_F_chunks)
        if pareto_F_chunks
        else np.array([], dtype=np.float64).reshape(0, objective_dim)
    )
    G = (
        np.vstack(pareto_G_chunks)
        if pareto_G_chunks
        else np.array([], dtype=np.float64).reshape(0, 0)
    )
    final_pop_X = (
        np.vstack(final_X_chunks)
        if final_X_chunks
        else np.array([], dtype=np.float64).reshape(0, N_TOTAL)
    )
    final_pop_F = (
        np.vstack(final_F_chunks)
        if final_F_chunks
        else np.array([], dtype=np.float64).reshape(0, objective_dim)
    )
    final_pop_G = (
        np.vstack(final_G_chunks)
        if final_G_chunks
        else np.array([], dtype=np.float64).reshape(0, G.shape[1] if G.ndim == 2 else 0)
    )

    np.save(opt_outdir / "pareto_X.npy", X)
    np.save(opt_outdir / "pareto_F.npy", F)
    np.save(opt_outdir / "pareto_G.npy", G)
    np.save(opt_outdir / "final_pop_X.npy", final_pop_X)
    np.save(opt_outdir / "final_pop_F.npy", final_pop_F)
    np.save(opt_outdir / "final_pop_G.npy", final_pop_G)
    (opt_outdir / "pareto_condition_index.json").write_text(json.dumps(pareto_condition_index))
    (opt_outdir / "final_condition_index.json").write_text(json.dumps(final_condition_index))

    n_pareto = int(X.shape[0])
    manifest["steps"]["optimization"] = {
        "ok": True,
        "elapsed_s": time.perf_counter() - t0,
        "outdir": str(opt_outdir),
        "n_pareto": n_pareto,
        "n_final_pop": int(final_pop_X.shape[0]),
        "n_conditions": len(operating_conditions),
        "conditions": condition_results,
    }
    _log(
        "Step completed: optimization",
        elapsed_s=round(float(manifest["steps"]["optimization"]["elapsed_s"]), 3),  # type: ignore[index]
        n_pareto=n_pareto,
        n_final_pop=int(final_pop_X.shape[0]),
        n_conditions=len(operating_conditions),
    )
    _write_manifest()

    # 4) CEM validation gate.
    t0 = time.perf_counter()
    source = "pareto"
    X_source = X
    F_source = F
    source_contexts = pareto_contexts
    if n_pareto == 0 and bool(getattr(args, "allow_nonproduction_paths", False)):
        source = "final_population_fallback"
        X_source = final_pop_X
        F_source = final_pop_F
        source_contexts = final_contexts

    if X_source.shape[0] == 0:
        manifest["steps"]["cem_validation"] = {
            "ok": False,
            "elapsed_s": time.perf_counter() - t0,
            "error": "No candidates available from Pareto or final population.",
            "source": source,
        }
        manifest["ready_for_quality_analysis"] = False
        _write_manifest()
        _log("Step failed: cem_validation", level="ERROR", source=source, reason="no candidates")
        print("Dress rehearsal failed: no candidates available for CEM validation.")
        return 2

    _log(
        "Step started: cem_validation",
        source=source,
        n_source=int(X_source.shape[0]),
        cem_top=args.cem_top,
        cem_min_feasible=args.cem_min_feasible,
    )

    if args.cem_top > 0:
        sel_ctx = _build_eval_context_from_args(args)
        idx, promotion_meta = _select_promotion_indices(
            X_source,
            F_source,
            ctx=sel_ctx,
            k=min(int(args.cem_top), int(X_source.shape[0])),
            margin_min=float(args.promotion_margin_min),
            pool_mult=int(args.promotion_pool_mult),
            contexts=source_contexts if source_contexts else None,
        )
        X_val = X_source[idx]
        F_val = F_source[idx]
    else:
        idx = np.arange(int(X_source.shape[0]), dtype=int)
        promotion_meta = {
            "n_candidates": int(X_source.shape[0]),
            "n_selected": int(X_source.shape[0]),
            "strategy": "all_candidates",
        }
        X_val = X_source
        F_val = F_source

    _log(
        "Promotion selection complete",
        n_selected=int(X_val.shape[0]),
        strategy=promotion_meta.get("strategy", "rank_margin_diversity"),
        margin_min=promotion_meta.get("margin_min_required", ""),
    )

    report = validate_candidates(X_val, F_val)
    report.archive_path = str(opt_outdir)
    report.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    report_text = format_report(report)

    cem_report_txt = outdir / "cem_validation_report.txt"
    cem_report_json = outdir / "cem_validation_report.json"
    cem_report_txt.write_text(report_text)
    cem_payload = {
        "n_candidates": report.n_candidates,
        "n_feasible": report.n_feasible,
        "timestamp": report.timestamp,
        "archive_path": report.archive_path,
        "encoding_version": report.encoding_version,
        "promotion": promotion_meta,
        "candidates": [
            {
                "index": c.index,
                "is_feasible": c.is_feasible,
                "feasibility_score": c.feasibility_score,
                "cem_lambda_min": c.cem_lambda_min,
                "cem_scuff_margin_C": c.cem_scuff_margin_C,
                "cem_micropitting_sf": c.cem_micropitting_sf,
                "cem_material_temp_margin_C": c.cem_material_temp_margin_C,
                "cem_cost_index": c.cem_cost_index,
                "cem_lube_regime": c.cem_lube_regime,
                "surr_lambda_min": c.surr_lambda_min,
                "surr_scuff_margin_C": c.surr_scuff_margin_C,
                "drift_lambda_pct": c.drift_lambda_pct,
                "drift_scuff_pct": c.drift_scuff_pct,
                "recommendation": c.recommendation,
            }
            for c in report.ranked_candidates
        ],
    }
    cem_report_json.write_text(json.dumps(cem_payload, indent=2))

    constraint_phase = str(args.constraint_phase)
    if constraint_phase == "explore":
        # Explore mode: optimization must proceed even when CEM feasibility is low.
        # Gate only requires that validation executed and produced candidates.
        cem_gate_mode = "report_only"
        cem_gate_ok = report.n_candidates > 0
        cem_gate_target = 1
    else:
        # Downselect mode: enforce explicit CEM feasibility minimum.
        cem_gate_mode = "feasible_min"
        cem_gate_ok = report.n_feasible >= args.cem_min_feasible
        cem_gate_target = int(args.cem_min_feasible)

    if constraint_phase == "explore":
        # Explore mode: optimizer health gate is execution-level, not strict Pareto size.
        pareto_gate_mode = "report_only"
        pareto_gate_ok = int(final_pop_X.shape[0]) > 0
        pareto_gate_target = 1
    else:
        pareto_gate_mode = "pareto_min"
        pareto_gate_ok = n_pareto >= args.min_pareto
        pareto_gate_target = int(args.min_pareto)

    manifest["steps"]["cem_validation"] = {
        "ok": cem_gate_ok,
        "elapsed_s": time.perf_counter() - t0,
        "source": source,
        "promotion": promotion_meta,
        "pareto_gate_mode": pareto_gate_mode,
        "pareto_gate_target": pareto_gate_target,
        "pareto_gate_ok": pareto_gate_ok,
        "gate_mode": cem_gate_mode,
        "gate_target": cem_gate_target,
        "constraint_phase": constraint_phase,
        "n_candidates": report.n_candidates,
        "n_feasible": report.n_feasible,
        "n_pareto": n_pareto,
        "min_pareto_required": args.min_pareto,
        "cem_min_feasible_required": args.cem_min_feasible,
        "report_text": str(cem_report_txt),
        "report_json": str(cem_report_json),
    }
    _log(
        "Step completed: cem_validation",
        elapsed_s=round(float(manifest["steps"]["cem_validation"]["elapsed_s"]), 3),  # type: ignore[index]
        n_candidates=report.n_candidates,
        n_feasible=report.n_feasible,
        gate_mode=cem_gate_mode,
        gate_target=cem_gate_target,
        pareto_gate_mode=pareto_gate_mode,
        pareto_gate_target=pareto_gate_target,
        pareto_gate_ok=pareto_gate_ok,
        cem_gate_ok=cem_gate_ok,
    )
    _write_manifest()

    ready = bool(
        manifest["steps"]["verify_surrogates"]["ok"]  # type: ignore[index]
        and manifest["steps"]["unit_tests"]["ok"]  # type: ignore[index]
        and manifest["steps"]["optimization"]["ok"]  # type: ignore[index]
        and pareto_gate_ok
        and cem_gate_ok
    )
    manifest["ready_for_quality_analysis"] = ready
    _write_manifest()
    _log(
        "Dress rehearsal finished",
        gate_status="PASS" if ready else "FAIL",
        ready_for_quality_analysis=ready,
        pareto_gate_ok=pareto_gate_ok,
        cem_gate_ok=cem_gate_ok,
    )

    if not ready and args.fail_on_gate:
        print(f"Dress rehearsal finished, but gate failed. See {manifest_path}")
        return 2

    print(f"Dress rehearsal complete. Gate status: {'PASS' if ready else 'FAIL'}")
    print(f"Manifest: {manifest_path}")
    return 0


# ---------------------------------------------------------------------------
# Explore -> Exploit (Pareto + CasADi refinement + Tribology)
# ---------------------------------------------------------------------------


def _parse_float_list(raw: str) -> np.ndarray | None:
    text = str(raw).strip()
    if not text:
        return None
    vals = [float(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not vals:
        return None
    return np.asarray(vals, dtype=np.float64)


def _parse_int_list(raw: str) -> list[int] | None:
    text = str(raw).strip()
    if not text:
        return None
    vals = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
    return vals or None


def _ensure_casadi_available(*, purpose: str) -> None:
    try:
        import casadi as _ca  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "CasADi import failed in active runtime "
            f"({sys.executable}) for {purpose}: {type(exc).__name__}: {exc}. "
            "Install optional dependency in this interpreter: pip install -e '.[casadi]'"
        ) from exc


def _resolve_default_hifi_quality_contract_artifacts() -> list[str]:
    summary = DEFAULT_PIPELINE_READINESS_SUMMARY
    if not summary.exists() and ALT_PIPELINE_READINESS_SUMMARY.exists():
        summary = ALT_PIPELINE_READINESS_SUMMARY
    return [
        str(summary.resolve(strict=False)),
        str(DEFAULT_F2_BLOCKERS.resolve(strict=False)),
        str(DEFAULT_ARTIFACT_CONTRACT_AUDIT.resolve(strict=False)),
    ]


def _extract_exception_payload(exc: BaseException | None) -> dict[str, Any]:
    if exc is None:
        return {}
    payload = getattr(exc, "payload", None)
    if isinstance(payload, dict) and payload:
        return payload
    for nested in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        nested_payload = _extract_exception_payload(nested)
        if nested_payload:
            return nested_payload
    return {}


def run_explore_exploit_workflow(args: argparse.Namespace) -> int:
    """Two-stage pipeline: explore Pareto set then exploit selected slices via CasADi."""
    from larrak_optimization.pipelines.principles_frontier import synthesize_principles_frontier

    from larrak2.pipelines.explore_exploit import run_two_stage_pipeline

    source_mode = str(getattr(args, "explore_source", "principles")).strip().lower()
    pareto_dir = Path(args.pareto_dir)
    if source_mode == "archive" and bool(args.run_explore):
        pareto_dir.mkdir(parents=True, exist_ok=True)
        explore_argv = [
            "--pop",
            str(args.pop),
            "--gen",
            str(args.gen),
            "--algorithm",
            str(getattr(args, "algorithm", "nsga3")),
            "--partitions",
            str(getattr(args, "partitions", 4)),
            "--nsga3-max-ref-dirs",
            str(getattr(args, "nsga3_max_ref_dirs", 192)),
            "--rpm",
            str(args.rpm),
            "--torque",
            str(args.torque),
            "--fidelity",
            str(args.explore_fidelity),
            "--seed",
            str(args.seed),
            "--output",
            str(pareto_dir),
            "--thermo-model",
            str(getattr(args, "thermo_model", "two_zone_eq_v1")),
            "--thermo-constants-path",
            str(getattr(args, "thermo_constants_path", "")),
            "--thermo-anchor-manifest",
            str(getattr(args, "thermo_anchor_manifest", "")),
            "--tribology-scuff-method",
            str(getattr(args, "tribology_scuff_method", "auto")),
        ]
        strict_trib = getattr(args, "strict_tribology_data", None)
        if strict_trib is True:
            explore_argv.append("--strict-tribology-data")
        elif strict_trib is False:
            explore_argv.append("--no-strict-tribology-data")
        if bool(getattr(args, "allow_nonproduction_paths", False)):
            explore_argv.append("--allow-nonproduction-paths")
        if bool(args.verbose):
            explore_argv.append("--verbose")
        code = run_pareto_main(explore_argv)
        if code != 0:
            print("Explore stage failed: run_pareto returned non-zero exit code")
            return int(code)

    if source_mode == "archive":
        for name in ("pareto_X.npy", "pareto_F.npy", "pareto_G.npy"):
            if not (pareto_dir / name).exists():
                print(
                    f"Missing Pareto artifact '{name}' in {pareto_dir}. "
                    "Run with --run-explore or point --pareto-dir to a completed archive."
                )
                return 1

    backend_mode = str(getattr(args, "backend", "casadi"))
    if backend_mode == "casadi":
        try:
            _ensure_casadi_available(purpose="explore-exploit casadi backend")
        except Exception as exc:
            print(str(exc))
            if bool(getattr(args, "architecture_probe_mode", False)):
                outdir = Path(args.outdir)
                outdir.mkdir(parents=True, exist_ok=True)
                trace_path = outdir / "contract_trace.jsonl"
                summary_path = outdir / "contract_summary.json"
                failure_manifest = {
                    "workflow": "explore_exploit",
                    "selected_indices": [],
                    "pareto_source": str(Path(args.pareto_dir)),
                    "explore_source": str(getattr(args, "explore_source", "principles")),
                    "contract_version": CONTRACT_VERSION,
                    "contract_trace_file": str(trace_path),
                    "contract_summary_file": str(summary_path),
                    "contract_summary": {},
                    "release_readiness": {
                        "release_ready": False,
                        "strict_data": True,
                        "constraint_phase": str(
                            getattr(args, "hifi_constraint_phase", "downselect")
                        ),
                        "reasons": ["architecture_probe_runtime_exception"],
                    },
                    "failure": {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                }
                (outdir / "explore_exploit_manifest.json").write_text(
                    json.dumps(failure_manifest, indent=2),
                    encoding="utf-8",
                )
            return 1
    thermo_symbolic_mode = str(getattr(args, "thermo_symbolic_mode", "strict"))
    explicit_thermo_symbolic_path = (
        str(getattr(args, "thermo_symbolic_artifact_path", "")).strip() or None
    )
    explicit_stack_model_path = str(getattr(args, "stack_model_path", "")).strip() or None
    try:
        resolved_hifi_thermo_symbolic_path = str(
            resolve_thermo_symbolic_artifact_path(
                fidelity=int(args.hifi_fidelity),
                explicit_path=explicit_thermo_symbolic_path,
                must_exist=(backend_mode == "casadi" and thermo_symbolic_mode.lower() == "strict"),
            )
        )
    except Exception as exc:
        print(str(exc))
        return 1
    try:
        resolved_stack_model_path = str(
            resolve_stack_artifact_path(
                fidelity=int(args.hifi_fidelity),
                explicit_path=explicit_stack_model_path,
                must_exist=backend_mode == "casadi",
            )
        )
    except Exception as exc:
        print(str(exc))
        return 1

    breathing = BreathingConfig(
        bore_mm=float(getattr(args, "bore_mm", 80.0)),
        stroke_mm=float(getattr(args, "stroke_mm", 90.0)),
        intake_port_area_m2=float(getattr(args, "intake_port_area_m2", 4.0e-4)),
        exhaust_port_area_m2=float(getattr(args, "exhaust_port_area_m2", 4.0e-4)),
        p_manifold_Pa=float(getattr(args, "p_manifold_pa", 101325.0)),
        p_back_Pa=float(getattr(args, "p_back_pa", 101325.0)),
        overlap_deg=float(getattr(args, "overlap_deg", 0.0)),
        intake_open_deg=float(getattr(args, "intake_open_deg", 0.0)),
        intake_close_deg=float(getattr(args, "intake_close_deg", 0.0)),
        exhaust_open_deg=float(getattr(args, "exhaust_open_deg", 0.0)),
        exhaust_close_deg=float(getattr(args, "exhaust_close_deg", 0.0)),
        compression_ratio=float(getattr(args, "compression_ratio", 10.0)),
        fuel_name=str(getattr(args, "fuel_name", "gasoline")),
        valve_timing_mode="candidate",
    )

    lowfi_ctx = EvalContext(
        rpm=float(args.rpm),
        torque=float(args.torque),
        fidelity=int(args.explore_fidelity),
        seed=int(args.seed),
        breathing=breathing,
        constraint_phase="explore",
        openfoam_model_path=str(getattr(args, "openfoam_model_path", "")).strip() or None,
        calculix_stress_mode=str(getattr(args, "calculix_stress_mode", "nn")),
        calculix_model_path=str(getattr(args, "calculix_model_path", "")).strip() or None,
        gear_loss_mode=str(getattr(args, "gear_loss_mode", "physics")),
        gear_loss_model_dir=str(getattr(args, "gear_loss_model_dir", "")).strip() or None,
        thermo_model=str(getattr(args, "thermo_model", "two_zone_eq_v1")),
        thermo_constants_path=str(getattr(args, "thermo_constants_path", "")).strip() or None,
        thermo_anchor_manifest_path=str(getattr(args, "thermo_anchor_manifest", "")).strip()
        or None,
        thermo_chemistry_profile_path=str(
            getattr(args, "thermo_chemistry_profile_path", "")
        ).strip()
        or None,
        thermo_symbolic_mode=thermo_symbolic_mode,
        thermo_symbolic_artifact_path=explicit_thermo_symbolic_path,
        production_profile="strict_prod",
        allow_nonproduction_paths=bool(getattr(args, "allow_nonproduction_paths", False)),
        strict_data=bool(getattr(args, "strict_data", True)),
        strict_tribology_data=getattr(args, "strict_tribology_data", None),
        tribology_scuff_method=str(getattr(args, "tribology_scuff_method", "auto")),
        surrogate_validation_mode=str(getattr(args, "surrogate_validation_mode", "strict")),
        machining_mode=str(getattr(args, "machining_mode", "nn")),
        machining_model_path=str(getattr(args, "machining_model_path", "")).strip() or None,
    )
    hifi_ctx = EvalContext(
        rpm=float(args.rpm),
        torque=float(args.torque),
        fidelity=int(args.hifi_fidelity),
        seed=int(args.seed),
        breathing=breathing,
        constraint_phase=str(getattr(args, "hifi_constraint_phase", "downselect")),
        openfoam_model_path=str(getattr(args, "openfoam_model_path", "")).strip() or None,
        calculix_stress_mode=str(getattr(args, "calculix_stress_mode", "nn")),
        calculix_model_path=str(getattr(args, "calculix_model_path", "")).strip() or None,
        gear_loss_mode=str(getattr(args, "gear_loss_mode", "physics")),
        gear_loss_model_dir=str(getattr(args, "gear_loss_model_dir", "")).strip() or None,
        thermo_model=str(getattr(args, "thermo_model", "two_zone_eq_v1")),
        thermo_constants_path=str(getattr(args, "thermo_constants_path", "")).strip() or None,
        thermo_anchor_manifest_path=str(getattr(args, "thermo_anchor_manifest", "")).strip()
        or None,
        thermo_chemistry_profile_path=str(
            getattr(args, "thermo_chemistry_profile_path", "")
        ).strip()
        or None,
        thermo_symbolic_mode=thermo_symbolic_mode,
        thermo_symbolic_artifact_path=resolved_hifi_thermo_symbolic_path,
        production_profile="strict_prod",
        allow_nonproduction_paths=bool(getattr(args, "allow_nonproduction_paths", False)),
        strict_data=bool(getattr(args, "strict_data", True)),
        strict_tribology_data=getattr(args, "strict_tribology_data", None),
        tribology_scuff_method=str(getattr(args, "tribology_scuff_method", "auto")),
        surrogate_validation_mode=str(getattr(args, "surrogate_validation_mode", "strict")),
        machining_mode=str(getattr(args, "machining_mode", "nn")),
        machining_model_path=str(getattr(args, "machining_model_path", "")).strip() or None,
    )

    weights = _parse_float_list(str(args.rank_weights))
    refine_indices = _parse_int_list(str(args.refine_indices))

    ipopt_options = {
        k: v
        for k, v in {
            "max_iter": args.ipopt_max_iter,
            "tol": args.ipopt_tol,
            "linear_solver": args.ipopt_linear_solver,
        }.items()
        if v is not None
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    principles_result = None
    trace_path = outdir / "contract_trace.jsonl"
    summary_path = outdir / "contract_summary.json"
    tracer_fidelity = (
        int(hifi_ctx.fidelity) if int(hifi_ctx.fidelity) == int(lowfi_ctx.fidelity) else None
    )
    with active_contract_tracer(
        trace_path=trace_path,
        summary_path=summary_path,
        fidelity=tracer_fidelity,
        enforce_routing=bool(getattr(args, "enforce_contract_routing", False)),
    ):
        candidate_store = None
        effective_pareto_source = pareto_dir
        if source_mode == "principles":
            export_archive = str(getattr(args, "principles_export_archive_dir", "")).strip() or str(
                outdir / "principles_pareto"
            )
            principles_result = synthesize_principles_frontier(
                outdir=outdir,
                ctx=lowfi_ctx,
                profile_name=str(getattr(args, "principles_profile", "iso_litvin_v2")),
                seed=int(args.seed),
                seed_count=int(getattr(args, "principles_seed_count", 64)),
                min_frontier_size=int(getattr(args, "principles_region_min_size", 12)),
                root_max_iter=int(getattr(args, "principles_root_max_iter", 80)),
                export_archive_dir=export_archive,
                contract_version=CONTRACT_VERSION,
                allow_nonproduction_paths=bool(getattr(args, "allow_nonproduction_paths", False)),
                alignment_mode=str(getattr(args, "principles_alignment_mode", "blend")),
                alignment_fidelity=int(getattr(args, "principles_canonical_alignment_fidelity", 1)),
            )
            candidate_store = principles_result.store
            effective_pareto_source = Path(principles_result.pareto_source)

        try:
            manifest = run_two_stage_pipeline(
                pareto_dir=effective_pareto_source,
                candidate_store=candidate_store,
                outdir=outdir,
                lowfi_ctx=lowfi_ctx,
                hifi_ctx=hifi_ctx,
                top_k=int(args.top_k),
                candidate_index=None
                if int(args.candidate_index) < 0
                else int(args.candidate_index),
                rank_weights=weights,
                refine_indices=refine_indices,
                mode=str(args.mode),
                backend=str(args.backend),
                active_k=args.active_k,
                min_per_group=int(args.min_per_group),
                slice_method=str(args.slice_method),
                trust_radius=args.trust_radius,
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                eps_margin=float(args.eps_margin),
                run_tribology_stage=not bool(args.skip_tribology),
                ipopt_options=ipopt_options or None,
                stack_model_path=(resolved_stack_model_path if backend_mode == "casadi" else None),
                contract_version=CONTRACT_VERSION,
                contract_trace_file=str(trace_path),
                contract_summary_file=str(summary_path),
                architecture_probe_mode=bool(getattr(args, "architecture_probe_mode", False)),
                explore_source=source_mode,
                principles_profile=(
                    str(principles_result.profile_name)
                    if principles_result is not None
                    else str(getattr(args, "principles_profile", ""))
                ),
                principles_frontier_gate=(
                    dict(principles_result.gate) if principles_result is not None else {}
                ),
                principles_artifacts=(
                    dict(principles_result.artifacts) if principles_result is not None else {}
                ),
                principles_region_summary=(
                    dict(principles_result.region_summary) if principles_result is not None else {}
                ),
                principles_proxy_vs_canonical=(
                    dict(principles_result.proxy_vs_canonical)
                    if principles_result is not None
                    else {}
                ),
                principles_diagnosis=(
                    dict(principles_result.diagnosis) if principles_result is not None else {}
                ),
                production_profile="strict_prod",
                allow_nonproduction_paths=bool(getattr(args, "allow_nonproduction_paths", False)),
            )
        except Exception:
            manifest_path = outdir / "explore_exploit_manifest.json"
            if not manifest_path.exists():
                exc_type, exc_value, _ = sys.exc_info()
                tracer = get_active_contract_tracer()
                contract_summary = tracer.summary_payload() if tracer is not None else {}
                failure_details = _extract_exception_payload(exc_value)
                failure_manifest = {
                    "workflow": "explore_exploit",
                    "selected_indices": [],
                    "pareto_source": str(effective_pareto_source),
                    "explore_source": source_mode,
                    "principles_profile": (
                        str(principles_result.profile_name)
                        if principles_result is not None
                        else str(getattr(args, "principles_profile", ""))
                    ),
                    "principles_frontier_gate": (
                        dict(principles_result.gate) if principles_result is not None else {}
                    ),
                    "principles_artifacts": (
                        dict(principles_result.artifacts) if principles_result is not None else {}
                    ),
                    "principles_region_summary": (
                        dict(principles_result.region_summary)
                        if principles_result is not None
                        else {}
                    ),
                    "principles_proxy_vs_canonical": (
                        dict(principles_result.proxy_vs_canonical)
                        if principles_result is not None
                        else {}
                    ),
                    "principles_problem": {
                        "profile": (
                            str(principles_result.profile_name)
                            if principles_result is not None
                            else str(getattr(args, "principles_profile", ""))
                        ),
                        "alignment": (
                            dict(
                                (principles_result.region_summary or {}).get(
                                    "canonical_alignment", {}
                                )
                            )
                            if principles_result is not None
                            else {}
                        ),
                        "normalization_scales": (
                            dict(
                                (principles_result.region_summary or {}).get(
                                    "normalization_scales", {}
                                )
                            )
                            if principles_result is not None
                            else {}
                        ),
                    },
                    "reduced_core": (
                        dict((principles_result.region_summary or {}).get("reduced_core", {}))
                        if principles_result is not None
                        else {}
                    ),
                    "expansion_policy": (
                        dict((principles_result.region_summary or {}).get("expansion_policy", {}))
                        if principles_result is not None
                        else {}
                    ),
                    "diagnosis_classification": str(
                        (principles_result.diagnosis or {}).get("classification", "")
                    )
                    if principles_result is not None
                    else "",
                    "source_region_pass": bool(
                        (principles_result.diagnosis or {}).get("source_region_pass", False)
                    )
                    if principles_result is not None
                    else False,
                    "optimization_pass": False,
                    "contract_version": CONTRACT_VERSION,
                    "contract_trace_file": str(trace_path),
                    "contract_summary_file": str(summary_path),
                    "contract_summary": contract_summary,
                    "release_readiness": {
                        "release_ready": False,
                        "strict_data": bool(hifi_ctx.strict_data),
                        "constraint_phase": str(hifi_ctx.constraint_phase),
                        "reasons": ["runtime_exception_before_manifest"],
                    },
                    "failure": {
                        "error_type": str(exc_type.__name__) if exc_type else "",
                        "error_message": str(exc_value) if exc_value else "",
                        "details": failure_details,
                    },
                }
                manifest_path.write_text(
                    json.dumps(failure_manifest, indent=2),
                    encoding="utf-8",
                )
            raise

    manifest_path = Path(args.outdir) / "explore_exploit_manifest.json"
    print("Explore->Exploit pipeline complete.")
    print(f"Selected candidates: {manifest.get('selected_indices', [])}")
    print(f"Manifest: {manifest_path}")
    return 0


# ---------------------------------------------------------------------------
# Backend Orchestration (No GUI)
# ---------------------------------------------------------------------------


def _load_truth_plan_tokens(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        items = payload.get("candidates", payload.get("plan", []))
        if isinstance(items, list):
            return [str(item).strip() for item in items if str(item).strip()]
    raise ValueError(
        "truth plan must be a JSON list of candidate tokens or a dict with 'candidates' list"
    )


def run_orchestrate_workflow(args: argparse.Namespace) -> int:
    """Run backend-only orchestration merge path (Wave 2)."""
    from larrak_orchestration.legacy_loop.backends import (
        FileControlBackend,
        JSONLProvenanceBackend,
        RedisControlBackend,
        WeaviateProvenanceBackend,
    )
    from larrak_simulation.pipelines.openfoam import OpenFoamPipeline

    from larrak2.adapters.openfoam import OpenFoamRunner
    from larrak2.orchestration import OrchestrationConfig, Orchestrator
    from larrak2.orchestration.adapters import (
        CasadiSolverAdapter,
        CEMAdapter,
        HifiSurrogateAdapter,
        PhysicsSimulationAdapter,
    )
    from larrak2.orchestration.multi_start import optimize_with_multistart

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        _ensure_casadi_available(purpose="orchestrate casadi solver adapter")
    except Exception as exc:
        print(str(exc))
        return 1

    thermo_symbolic_mode = str(getattr(args, "thermo_symbolic_mode", "strict"))
    explicit_thermo_symbolic_path = (
        str(getattr(args, "thermo_symbolic_artifact_path", "")).strip() or None
    )
    explicit_stack_model_path = str(getattr(args, "stack_model_path", "")).strip() or None
    try:
        resolved_thermo_symbolic_path = str(
            resolve_thermo_symbolic_artifact_path(
                fidelity=int(getattr(args, "fidelity", 2)),
                explicit_path=explicit_thermo_symbolic_path,
                must_exist=thermo_symbolic_mode.lower() == "strict",
            )
        )
    except Exception as exc:
        print(str(exc))
        return 1
    try:
        resolved_stack_model_path = str(
            resolve_stack_artifact_path(
                fidelity=int(getattr(args, "fidelity", 2)),
                explicit_path=explicit_stack_model_path,
                must_exist=True,
            )
        )
    except Exception as exc:
        print(str(exc))
        return 1
    ipopt_options = {
        k: v
        for k, v in {
            "max_iter": getattr(args, "ipopt_max_iter", None),
            "tol": getattr(args, "ipopt_tol", None),
            "linear_solver": getattr(args, "ipopt_linear_solver", None),
        }.items()
        if v is not None
    }

    truth_tokens: list[str] | None = None
    if str(args.truth_dispatch_mode) == "manual":
        if not str(args.truth_plan).strip():
            print("--truth-plan is required when --truth-dispatch-mode=manual")
            return 1
        try:
            truth_tokens = _load_truth_plan_tokens(str(args.truth_plan))
        except Exception as exc:
            print(f"Failed to load truth plan '{args.truth_plan}': {exc}")
            return 1

    if str(args.control_backend) == "redis":
        control_backend = RedisControlBackend()
    else:
        control_backend = FileControlBackend(path=outdir / "control_signal.json")

    if str(args.provenance_backend) == "off":
        provenance_backend = None
        use_provenance = False
    elif str(args.provenance_backend) == "weaviate":
        provenance_backend = WeaviateProvenanceBackend(
            mirror_jsonl=outdir / "provenance_events.jsonl"
        )
        use_provenance = True
    else:
        provenance_backend = JSONLProvenanceBackend(path=outdir / "provenance_events.jsonl")
        use_provenance = True

    config = OrchestrationConfig(
        total_sim_budget=int(args.sim_budget),
        batch_size=int(args.batch_size),
        max_iterations=int(args.max_iterations),
        seed=int(args.seed),
        rpm=float(args.rpm),
        torque=float(args.torque),
        fidelity=int(getattr(args, "fidelity", 2)),
        bore_mm=float(getattr(args, "bore_mm", 80.0)),
        stroke_mm=float(getattr(args, "stroke_mm", 90.0)),
        intake_port_area_m2=float(getattr(args, "intake_port_area_m2", 4.0e-4)),
        exhaust_port_area_m2=float(getattr(args, "exhaust_port_area_m2", 4.0e-4)),
        p_manifold_pa=float(getattr(args, "p_manifold_pa", 101325.0)),
        p_back_pa=float(getattr(args, "p_back_pa", 101325.0)),
        compression_ratio=float(getattr(args, "compression_ratio", 10.0)),
        fuel_name=str(getattr(args, "fuel_name", "gasoline")),
        constraint_phase=str(getattr(args, "constraint_phase", "downselect")),
        enforce_contract_routing=bool(getattr(args, "enforce_contract_routing", False)),
        truth_dispatch_mode=str(args.truth_dispatch_mode),
        truth_plan=truth_tokens,
        truth_auto_top_k=int(getattr(args, "truth_auto_top_k", 2)),
        truth_auto_min_uncertainty=float(getattr(args, "truth_auto_min_uncertainty", 0.0)),
        truth_auto_min_feasibility=float(getattr(args, "truth_auto_min_feasibility", 0.0)),
        truth_auto_min_pred_quantile=float(getattr(args, "truth_auto_min_pred_quantile", 0.0)),
        truth_records_path=str(getattr(args, "truth_records_path", "")).strip() or None,
        outdir=outdir,
        cache_path=str(args.cache_path).strip() or None,
        use_provenance=use_provenance,
        strict_data=bool(getattr(args, "strict_data", True)),
        strict_tribology_data=getattr(args, "strict_tribology_data", None),
        tribology_scuff_method=str(getattr(args, "tribology_scuff_method", "auto")),
        surrogate_validation_mode=str(getattr(args, "surrogate_validation_mode", "strict")),
        thermo_symbolic_mode=thermo_symbolic_mode,
        thermo_symbolic_artifact_path=resolved_thermo_symbolic_path,
        thermo_constants_path=str(getattr(args, "thermo_constants_path", "")).strip() or None,
        thermo_anchor_manifest_path=str(getattr(args, "thermo_anchor_manifest", "")).strip()
        or None,
        thermo_chemistry_profile_path=str(
            getattr(args, "thermo_chemistry_profile_path", "")
        ).strip()
        or None,
        machining_mode=str(getattr(args, "machining_mode", "nn")),
        machining_model_path=str(getattr(args, "machining_model_path", "")).strip() or None,
        stack_model_path=resolved_stack_model_path,
        allow_nonproduction_paths=bool(getattr(args, "allow_nonproduction_paths", False)),
        production_profile="strict_prod",
        ipopt_options=ipopt_options,
    )

    cem = CEMAdapter()
    surrogate_model_dir = str(getattr(args, "hifi_model_dir", "")).strip() or str(
        DEFAULT_HIFI_SURROGATE_DIR
    )
    surrogate = HifiSurrogateAdapter(
        model_dir=surrogate_model_dir,
        default_rpm=float(args.rpm),
        default_torque=float(args.torque),
        allow_heuristic_fallback=bool(getattr(args, "allow_heuristic_surrogate_fallback", False)),
        validation_mode=str(getattr(args, "surrogate_validation_mode", "strict")),
        required_quality_artifacts=(
            _resolve_default_hifi_quality_contract_artifacts()
            if bool(getattr(args, "strict_data", True))
            else []
        ),
    )
    solver = CasadiSolverAdapter(
        backend="casadi",
        mode="weighted_sum",
        stack_model_path=str(config.stack_model_path).strip() or None,
        ipopt_options=config.ipopt_options,
        trust_radius=None,
    )
    truth_run_openfoam = bool(getattr(args, "truth_run_openfoam", False))
    openfoam_runner = None
    if truth_run_openfoam and str(args.truth_dispatch_mode) != "off":
        template_dir = Path(
            str(getattr(args, "openfoam_template", "")).strip()
            or "openfoam_templates/opposed_piston_rotary_valve_sliding_case"
        )
        if not template_dir.exists():
            print(f"OpenFOAM template directory not found: {template_dir}")
            return 1
        openfoam_backend = str(getattr(args, "openfoam_backend", "docker"))
        openfoam_docker_image = str(getattr(args, "openfoam_docker_image", "")).strip() or None
        if openfoam_backend == "docker":
            openfoam_runner = OpenFoamPipeline(
                template_dir=template_dir,
                solver_cmd=str(getattr(args, "openfoam_solver", "larrakEngineFoam")),
                docker_image=openfoam_docker_image,
            )
            docker_ready = bool(getattr(openfoam_runner, "docker", None).check_availability())
        else:
            openfoam_runner = OpenFoamRunner(
                template_dir=template_dir,
                solver_cmd=str(getattr(args, "openfoam_solver", "larrakEngineFoam")),
                backend=openfoam_backend,
                docker_image=openfoam_docker_image,
            )
            docker_ready = True
        if openfoam_backend == "docker" and not docker_ready:
            print(
                "OpenFOAM truth dispatch requested but Docker/OpenFOAM is unavailable. "
                "Verify Docker is running and the configured image is pullable."
            )
            return 1

    simulation = PhysicsSimulationAdapter(
        work_dir=outdir / "truth_runs",
        openfoam_runner=openfoam_runner,
        run_openfoam=truth_run_openfoam,
    )

    orchestrator = Orchestrator(
        cem=cem,
        surrogate=surrogate,
        solver=solver,
        simulation=simulation,
        config=config,
        control_backend=control_backend,
        provenance_backend=provenance_backend,
    )

    initial_params = {
        "rpm": float(args.rpm),
        "torque": float(args.torque),
        "seed": int(args.seed),
    }

    if bool(args.multi_start):
        result = optimize_with_multistart(
            orchestrator, initial_params, n_starts=3, seed=int(args.seed)
        )
    else:
        result = orchestrator.optimize(initial_params=initial_params)

    print("Orchestration run complete.")
    print(f"Best objective: {result.best_objective:.6f} ({result.best_source})")
    print(f"Manifest: {result.manifest_path}")
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

    if template_dir.name == "opposed_piston_rotary_valve_sliding_case" and done == 0 and args.n > 0:
        smoke_intake_port_area_m2 = float(
            0.5 * (float(args.intake_port_area_min) + float(args.intake_port_area_max))
        )
        smoke_exhaust_port_area_m2 = float(
            0.5 * (float(args.exhaust_port_area_min) + float(args.exhaust_port_area_max))
        )
        smoke_p_back = float(0.5 * (float(args.p_back_min) + float(args.p_back_max)))
        smoke_p_man_lo = max(float(args.p_manifold_min), smoke_p_back)
        smoke_p_manifold = float(0.5 * (smoke_p_man_lo + float(args.p_manifold_max)))
        smoke_params = {
            "rpm": float(0.5 * (float(args.rpm_min) + float(args.rpm_max))),
            "torque": float(0.5 * (float(args.torque_min) + float(args.torque_max))),
            "lambda_af": float(0.5 * (float(args.lambda_min) + float(args.lambda_max))),
            "bore_mm": float(args.bore_mm),
            "stroke_mm": float(args.stroke_mm),
            "intake_port_area_m2": smoke_intake_port_area_m2,
            "exhaust_port_area_m2": smoke_exhaust_port_area_m2,
            "p_manifold_Pa": smoke_p_manifold,
            "p_back_Pa": smoke_p_back,
            "T_intake_K": 300.0,
            "T_residual_K": 900.0,
            "engine_start_angle_deg": -180.0,
            "engine_end_angle_deg": 180.0,
            "residual_fraction_seed": 0.08,
            "solver_name": args.solver,
            "endTime": float(args.endTime),
            "deltaT": float(args.deltaT),
            "writeInterval": int(args.writeInterval),
            "metricWriteInterval": int(args.metricWriteInterval),
        }
        pipeline.engine_smoke_gate(
            Path(args.outdir) / "engine_smoke_gate",
            params=smoke_params,
            geometry_args={
                "bore_mm": float(args.bore_mm),
                "stroke_mm": float(args.stroke_mm),
                "intake_port_area_m2": smoke_intake_port_area_m2,
                "exhaust_port_area_m2": smoke_exhaust_port_area_m2,
            },
        )

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
            "T_intake_K": 300.0,
            "T_residual_K": 900.0,
            "overlap_deg": overlap_deg,
            "intake_open_deg": intake_open_deg,
            "intake_close_deg": intake_close_deg,
            "exhaust_open_deg": exhaust_open_deg,
            "exhaust_close_deg": exhaust_close_deg,
            "engine_start_angle_deg": -180.0,
            "engine_end_angle_deg": 180.0,
            "residual_fraction_seed": 0.08,
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


def run_promote_openfoam_artifact_workflow(args: argparse.Namespace) -> int:
    """Promote a staged OpenFOAM authority bundle to the canonical runtime path."""
    from larrak2.surrogate.openfoam_authority import (
        promote_openfoam_artifact,
        resolve_latest_openfoam_authority_dir,
    )

    staged_dir = str(getattr(args, "staged_dir", "")).strip()
    bundle_root = str(getattr(args, "bundle_root", "")).strip() or "outputs/openfoam_authority"
    staged_path = (
        Path(staged_dir) if staged_dir else resolve_latest_openfoam_authority_dir(bundle_root)
    )

    result = promote_openfoam_artifact(
        staged_dir=staged_path,
        canonical_dir=str(getattr(args, "canonical_dir", "")).strip()
        or str(DEFAULT_OPENFOAM_NN_ARTIFACT.parent),
        backup_root=str(getattr(args, "backup_root", "")).strip()
        or "outputs/artifacts/surrogates/openfoam_nn/archive",
    )
    print("Promoted OpenFOAM artifact")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_overnight_f2_nn_campaign_workflow(args: argparse.Namespace) -> int:
    """Run staged overnight DOE/training/promotion campaign for F2 NN artifacts."""
    from larrak_simulation.training.overnight_campaign import (
        CampaignError,
        run_overnight_f2_nn_campaign,
    )

    run_id = str(getattr(args, "run_id", "")).strip() or time.strftime("%Y%m%d_%H%M%S")
    outdir_root = Path(str(getattr(args, "outdir_root", "outputs/overnight_f2")))
    outdir = outdir_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "overnight_f2_campaign_manifest.json"
    log_path = outdir / "overnight_f2_campaign.log"
    log_path.write_text("", encoding="utf-8")

    manifest: dict[str, Any] = {
        "workflow": "overnight_f2_nn_campaign",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "log_path": str(log_path),
        "ok": False,
    }

    def _log(message: str, *, level: str = "INFO", **fields: object) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} [{level}] {message}"
        if fields:
            rendered = ", ".join(f"{k}={fields[k]}" for k in sorted(fields))
            line = f"{line} | {rendered}"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line)

    try:
        summary = run_overnight_f2_nn_campaign(args, log_fn=_log)
        manifest["ok"] = True
        manifest["summary"] = summary
        _log("Overnight F2 NN campaign completed", outdir=outdir)
    except CampaignError as exc:
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        _log("Overnight F2 NN campaign failed", level="ERROR", error=str(exc))
        manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest_path}")
        return 2
    except Exception as exc:
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        _log("Overnight F2 NN campaign failed", level="ERROR", error=str(exc))
        manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest_path}")
        return 1

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0
