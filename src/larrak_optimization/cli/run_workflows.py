"""Optimization-only workflow runners for the standalone package."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from larrak_optimization.cli.run_pareto import main as run_pareto_main
from larrak_optimization.pipelines.explore_exploit import run_two_stage_pipeline
from larrak_optimization.pipelines.principles_frontier import synthesize_principles_frontier
from larrak_optimization.promote.staged import StagedWorkflow
from larrak_runtime.architecture.contracts import (
    CONTRACT_VERSION,
    active_contract_tracer,
    get_active_contract_tracer,
)
from larrak_runtime.core.artifact_paths import (
    resolve_stack_artifact_path,
    resolve_thermo_symbolic_artifact_path,
)
from larrak_runtime.core.types import BreathingConfig, EvalContext


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

    grid_records: list[dict[str, Any]] = []
    idx = 0
    for rpm in rpms:
        for tq in torques:
            idx += 1
            point_out = out_root / f"rpm{int(round(rpm))}_tq{int(round(tq))}"
            point_out.mkdir(parents=True, exist_ok=True)
            seed_point = int(args.seed) + idx
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
                "--compression-ratio",
                str(args.compression_ratio),
                "--fuel-name",
                str(args.fuel_name),
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
                "--thermo-chemistry-profile-path",
                str(getattr(args, "thermo_chemistry_profile_path", "")),
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
            if bool(args.verbose):
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

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
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
    summary_payload = {
        "config": {
            "pop": int(args.pop),
            "gen": int(args.gen),
            "fidelity": int(args.fidelity),
            "seed_base": int(args.seed),
            "rpms": rpms,
            "torques": torques,
        },
        "points": grid_records,
    }
    (out_root / "grid_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    return 0


def run_pareto_staged_workflow(args: argparse.Namespace) -> int:
    """Run multi-fidelity staged Pareto optimization."""
    output_dir = Path(args.outdir)
    workflow = StagedWorkflow(
        outdir=output_dir,
        rpm=float(args.rpm),
        torque=float(args.torque),
        seed=int(args.seed),
    )
    t0 = time.time()
    archive_s1 = workflow.run_stage1(int(args.pop), int(args.gen))
    archive_s2 = workflow.run_promotion(archive_s1, int(args.promote))
    archive_s3 = workflow.run_stage3(archive_s2, int(args.pop), int(args.gen))
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
            "stage1_eff_max": float(-np.min(s1_vals[:, 0])) if len(s1_vals) > 0 else 0.0,
            "stage3_eff_max": float(-np.min(s3_vals[:, 0])) if len(s3_vals) > 0 else 0.0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def _parse_float_list(raw: str) -> np.ndarray | None:
    text = str(raw).strip()
    if not text:
        return None
    vals = [float(tok.strip()) for tok in text.split(",") if tok.strip()]
    return np.asarray(vals, dtype=np.float64) if vals else None


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
            "--thermo-chemistry-profile-path",
            str(getattr(args, "thermo_chemistry_profile_path", "")),
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
                    "failure": {"error_type": type(exc).__name__, "error_message": str(exc)},
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
    ctx_kwargs = {
        "rpm": float(args.rpm),
        "torque": float(args.torque),
        "seed": int(args.seed),
        "breathing": breathing,
        "openfoam_model_path": str(getattr(args, "openfoam_model_path", "")).strip() or None,
        "calculix_stress_mode": str(getattr(args, "calculix_stress_mode", "nn")),
        "calculix_model_path": str(getattr(args, "calculix_model_path", "")).strip() or None,
        "gear_loss_mode": str(getattr(args, "gear_loss_mode", "physics")),
        "gear_loss_model_dir": str(getattr(args, "gear_loss_model_dir", "")).strip() or None,
        "thermo_model": str(getattr(args, "thermo_model", "two_zone_eq_v1")),
        "thermo_constants_path": str(getattr(args, "thermo_constants_path", "")).strip() or None,
        "thermo_anchor_manifest_path": str(getattr(args, "thermo_anchor_manifest", "")).strip()
        or None,
        "thermo_chemistry_profile_path": str(
            getattr(args, "thermo_chemistry_profile_path", "")
        ).strip()
        or None,
        "thermo_symbolic_mode": thermo_symbolic_mode,
        "production_profile": "strict_prod",
        "allow_nonproduction_paths": bool(getattr(args, "allow_nonproduction_paths", False)),
        "strict_data": bool(getattr(args, "strict_data", True)),
        "strict_tribology_data": getattr(args, "strict_tribology_data", None),
        "tribology_scuff_method": str(getattr(args, "tribology_scuff_method", "auto")),
        "surrogate_validation_mode": str(getattr(args, "surrogate_validation_mode", "strict")),
        "machining_mode": str(getattr(args, "machining_mode", "nn")),
        "machining_model_path": str(getattr(args, "machining_model_path", "")).strip() or None,
    }
    lowfi_ctx = EvalContext(
        fidelity=int(args.explore_fidelity),
        constraint_phase="explore",
        thermo_symbolic_artifact_path=explicit_thermo_symbolic_path,
        **ctx_kwargs,
    )
    hifi_ctx = EvalContext(
        fidelity=int(args.hifi_fidelity),
        constraint_phase=str(getattr(args, "hifi_constraint_phase", "downselect")),
        thermo_symbolic_artifact_path=resolved_hifi_thermo_symbolic_path,
        **ctx_kwargs,
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
                manifest_path.write_text(json.dumps(failure_manifest, indent=2), encoding="utf-8")
            raise

    manifest_path = Path(args.outdir) / "explore_exploit_manifest.json"
    print("Explore->Exploit pipeline complete.")
    print(f"Selected candidates: {manifest.get('selected_indices', [])}")
    print(f"Manifest: {manifest_path}")
    return 0
