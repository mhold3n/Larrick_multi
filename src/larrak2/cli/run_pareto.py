"""Pareto optimization CLI runner.

Usage:
    python -m larrak2.cli.run_pareto --pop 64 --gen 50 --rpm 3000 --torque 200
    python -m larrak2.cli.run_pareto --fidelity 1 --seed 123 --outdir ./results

Outputs:
    pareto_X.npy  - Decision vectors of Pareto front
    pareto_F.npy  - Objective values of Pareto front
    pareto_G.npy  - Constraint values of Pareto front
    summary.json  - Run metadata and statistics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..architecture.contracts import CONTRACT_VERSION, active_contract_tracer
from ..core.archive_io import save_archive
from ..core.constants import MODEL_VERSION_GEAR_V1, MODEL_VERSION_THERMO_V1
from ..core.constraints import get_constraint_names, get_constraint_scales
from ..core.encoding import ENCODING_VERSION
from ..core.types import BreathingConfig
from ..optimization.production_gate import (
    STRICT_PRODUCTION_PROFILE,
    evaluate_production_gate,
)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _deterministic_order(F: np.ndarray, X: np.ndarray) -> np.ndarray:
    if F.ndim != 2 or X.ndim != 2 or F.shape[0] == 0 or X.shape[0] == 0:
        return np.arange(int(F.shape[0] if F.ndim == 2 else 0), dtype=int)
    key = np.hstack([np.asarray(F, dtype=np.float64), np.asarray(X, dtype=np.float64)])
    return np.lexsort(key.T[::-1]).astype(int)


def main(argv: list[str] | None = None) -> int:
    """Run Pareto optimization.

    Args:
        argv: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(description="Run multi-objective Pareto optimization")
    parser.add_argument("--pop", type=int, default=64, help="Population size")
    parser.add_argument("--gen", type=int, default=100, help="Number of generations")
    parser.add_argument("--rpm", type=float, default=3000.0, help="Engine speed (rpm)")
    parser.add_argument("--torque", type=float, default=200.0, help="Torque demand (Nm)")
    parser.add_argument(
        "--fidelity", type=int, default=2, choices=[0, 1, 2], help="Model fidelity (0=toy, 1=v1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", "--outdir", type=str, default=".", dest="output", help="Output directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="nsga3",
        choices=["nsga2", "nsga3"],
        help="Optimization algorithm",
    )
    parser.add_argument(
        "--constraint-phase",
        type=str,
        default="downselect",
        choices=["explore", "downselect"],
        help="Constraint enforcement phase",
    )
    parser.add_argument(
        "--tolerance-constraint-mode",
        type=str,
        default="capability_floor",
        choices=["capability_floor", "stack_budget_max"],
        help="tol_budget interpretation: capability floor or max stack budget",
    )
    parser.add_argument(
        "--tolerance-threshold-mm",
        type=float,
        default=0.24,
        help="Tolerance threshold in mm used by tol_budget",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=4,
        help="Reference direction partitions (NSGA-III only)",
    )
    parser.add_argument(
        "--nsga3-max-ref-dirs",
        type=int,
        default=192,
        help="Maximum reference directions allowed for NSGA-III",
    )

    # Breathing/BC/timing inputs for OpenFOAM NN (fidelity>=2)
    parser.add_argument("--bore-mm", type=float, default=80.0)
    parser.add_argument("--stroke-mm", type=float, default=90.0)
    parser.add_argument("--intake-port-area-m2", type=float, default=4.0e-4)
    parser.add_argument("--exhaust-port-area-m2", type=float, default=4.0e-4)
    parser.add_argument("--p-manifold-pa", type=float, default=101325.0)
    parser.add_argument("--p-back-pa", type=float, default=101325.0)
    parser.add_argument("--overlap-deg", type=float, default=0.0)
    parser.add_argument("--intake-open-deg", type=float, default=0.0)
    parser.add_argument("--intake-close-deg", type=float, default=0.0)
    parser.add_argument("--exhaust-open-deg", type=float, default=0.0)
    parser.add_argument("--exhaust-close-deg", type=float, default=0.0)
    parser.add_argument("--compression-ratio", type=float, default=10.0)
    parser.add_argument(
        "--fuel-name",
        type=str,
        default="gasoline",
        choices=["gasoline", "ethanol", "methanol"],
    )
    parser.add_argument(
        "--openfoam-model-path",
        type=str,
        default="",
        help="Path to OpenFOAM NN artifact (required for fidelity=2 if env var not set)",
    )
    parser.add_argument(
        "--calculix-stress-mode",
        type=str,
        default="nn",
        choices=["nn", "analytical"],
        help="CalculiX stress evaluation mode",
    )
    parser.add_argument(
        "--calculix-model-path",
        type=str,
        default="",
        help="Path to CalculiX NN artifact when --calculix-stress-mode=nn",
    )
    parser.add_argument(
        "--gear-loss-mode",
        type=str,
        default="physics",
        choices=["physics", "nn"],
        help="Gear-loss evaluation mode (first-principles physics or NN)",
    )
    parser.add_argument(
        "--gear-loss-model-dir",
        type=str,
        default="",
        help="Gear-loss NN model directory when --gear-loss-mode=nn",
    )
    parser.add_argument(
        "--thermo-model",
        type=str,
        default="two_zone_eq_v1",
        choices=["two_zone_eq_v1"],
        help="Thermo backend model",
    )
    parser.add_argument(
        "--thermo-constants-path",
        type=str,
        default="",
        help="Override path to thermo literature constants pack",
    )
    parser.add_argument(
        "--thermo-anchor-manifest",
        type=str,
        default="",
        help="Override path to thermo benchmark anchor manifest",
    )
    parser.add_argument(
        "--thermo-chemistry-profile-path",
        type=str,
        default="",
        help="Override path to hybrid chemistry profile",
    )
    parser.add_argument(
        "--surrogate-validation-mode",
        type=str,
        default="strict",
        choices=["strict", "warn", "off"],
    )
    parser.add_argument(
        "--tribology-scuff-method",
        type=str,
        default="auto",
        choices=["auto", "flash", "integral"],
    )
    parser.add_argument("--strict-data", dest="strict_data", action="store_true")
    parser.add_argument("--no-strict-data", dest="strict_data", action="store_false")
    parser.add_argument(
        "--strict-tribology-data", dest="strict_tribology_data", action="store_true"
    )
    parser.add_argument(
        "--no-strict-tribology-data",
        dest="strict_tribology_data",
        action="store_false",
    )
    parser.set_defaults(strict_data=True)
    parser.set_defaults(strict_tribology_data=None)
    parser.add_argument("--machining-mode", type=str, default="nn", choices=["nn", "analytical"])
    parser.add_argument("--machining-model-path", type=str, default="")
    parser.add_argument(
        "--enforce-contract-routing",
        action="store_true",
        help="Fail if observed edge engine_mode violates the fidelity routing policy",
    )
    parser.add_argument(
        "--allow-nonproduction-paths",
        action="store_true",
        help="Allow non-production fallback paths and mark run as non-release",
    )

    args = parser.parse_args(argv)

    # Import here to avoid loading pymoo at module level
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions

    from larrak2.adapters.pymoo_problem import ParetoProblem
    from larrak2.core.types import EvalContext

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    breathing = BreathingConfig(
        bore_mm=args.bore_mm,
        stroke_mm=args.stroke_mm,
        intake_port_area_m2=args.intake_port_area_m2,
        exhaust_port_area_m2=args.exhaust_port_area_m2,
        p_manifold_Pa=args.p_manifold_pa,
        p_back_Pa=args.p_back_pa,
        overlap_deg=args.overlap_deg,
        intake_open_deg=args.intake_open_deg,
        intake_close_deg=args.intake_close_deg,
        exhaust_open_deg=args.exhaust_open_deg,
        exhaust_close_deg=args.exhaust_close_deg,
        compression_ratio=float(args.compression_ratio),
        fuel_name=str(args.fuel_name),
        valve_timing_mode="candidate",
    )

    ctx = EvalContext(
        rpm=args.rpm,
        torque=args.torque,
        fidelity=args.fidelity,
        seed=args.seed,
        breathing=breathing,
        constraint_phase=args.constraint_phase,
        tolerance_constraint_mode=args.tolerance_constraint_mode,
        tolerance_threshold_mm=args.tolerance_threshold_mm,
        openfoam_model_path=str(args.openfoam_model_path).strip() or None,
        calculix_stress_mode=str(args.calculix_stress_mode),
        calculix_model_path=str(args.calculix_model_path).strip() or None,
        gear_loss_mode=str(args.gear_loss_mode),
        gear_loss_model_dir=str(args.gear_loss_model_dir).strip() or None,
        thermo_model=str(args.thermo_model),
        thermo_constants_path=str(args.thermo_constants_path).strip() or None,
        thermo_anchor_manifest_path=str(args.thermo_anchor_manifest).strip() or None,
        thermo_chemistry_profile_path=str(args.thermo_chemistry_profile_path).strip() or None,
        strict_data=bool(args.strict_data),
        strict_tribology_data=args.strict_tribology_data,
        tribology_scuff_method=str(args.tribology_scuff_method),
        surrogate_validation_mode=str(args.surrogate_validation_mode),
        machining_mode=str(args.machining_mode),
        machining_model_path=str(args.machining_model_path).strip() or None,
        production_profile=STRICT_PRODUCTION_PROFILE,
        allow_nonproduction_paths=bool(args.allow_nonproduction_paths),
    )

    contract_trace_path = output_dir / "contract_trace.jsonl"
    contract_summary_path = output_dir / "contract_summary.json"

    with active_contract_tracer(
        trace_path=contract_trace_path,
        summary_path=contract_summary_path,
        fidelity=int(args.fidelity),
        enforce_routing=bool(args.enforce_contract_routing),
    ):
        problem = ParetoProblem(ctx=ctx)
        requested_pop = int(args.pop)
        requested_partitions = int(args.partitions)
        effective_pop = int(requested_pop)
        effective_partitions = int(requested_partitions)
        n_ref_dirs = 0
        algorithm_used = str(args.algorithm)
        ref_dirs = None

        if args.algorithm == "nsga3":
            max_ref_dirs = int(max(1, args.nsga3_max_ref_dirs))
            chosen_ref_dirs = None
            chosen_partitions = None
            for part in range(int(max(1, requested_partitions)), 0, -1):
                rd = get_reference_directions("das-dennis", problem.n_obj, n_partitions=part)
                if int(len(rd)) <= max_ref_dirs:
                    chosen_ref_dirs = rd
                    chosen_partitions = part
                    break
            if chosen_ref_dirs is None or chosen_partitions is None:
                raise RuntimeError(
                    "Unable to satisfy NSGA-III reference direction cap. "
                    f"Try smaller --partitions or larger --nsga3-max-ref-dirs "
                    f"(requested partitions={requested_partitions}, cap={max_ref_dirs})."
                )

            ref_dirs = np.asarray(chosen_ref_dirs, dtype=np.float64)
            n_ref_dirs = int(ref_dirs.shape[0])
            effective_partitions = int(chosen_partitions)
            effective_pop = int(max(requested_pop, n_ref_dirs))

            if args.verbose:
                print(f"Reference directions: {n_ref_dirs} points")

            algorithm = NSGA3(
                pop_size=effective_pop,
                ref_dirs=ref_dirs,
                prob_neighbor_mating=0.7,
            )
        else:
            algorithm = NSGA2(pop_size=effective_pop)
        termination = get_termination("n_gen", args.gen)

        if args.verbose:
            print(f"Starting {args.algorithm.upper()}: pop={effective_pop}, gen={args.gen}")
            if args.algorithm == "nsga3":
                print(
                    "NSGA-III settings: "
                    f"requested_pop={requested_pop}, effective_pop={effective_pop}, "
                    f"requested_partitions={requested_partitions}, effective_partitions={effective_partitions}, "
                    f"n_ref_dirs={n_ref_dirs}, ref_dir_cap={int(args.nsga3_max_ref_dirs)}"
                )
            print(f"Context: rpm={args.rpm}, torque={args.torque}, fidelity={args.fidelity}")
            print(
                "Constraint phase: "
                f"{args.constraint_phase}, tol mode: {args.tolerance_constraint_mode}, "
                f"tol threshold: {args.tolerance_threshold_mm:.3f} mm"
            )
            print(
                "Surrogate modes: "
                f"calculix={args.calculix_stress_mode}, gear_loss={args.gear_loss_mode}, "
                f"openfoam_model={args.openfoam_model_path or 'env/default'}"
            )

        # Run optimization
        t_start = time.perf_counter()

        result = minimize(
            problem,
            algorithm,
            termination,
            seed=args.seed,
            verbose=args.verbose,
        )

        t_elapsed = time.perf_counter() - t_start

        # Extract Pareto front (may be None if no feasible solutions)
        X = result.X
        F = result.F
        G_result = getattr(result, "G", None)
        n_eval_errors = int(getattr(problem, "n_eval_errors", 0))
        eval_error_signatures = dict(getattr(problem, "eval_error_signatures", {}))
        X_pop = result.pop.get("X") if result.pop is not None else None
        F_pop = result.pop.get("F") if result.pop is not None else None
        G_pop = result.pop.get("G") if result.pop is not None else None

        # Save final population arrays for downstream fallback workflows.
        if X_pop is None:
            np.save(output_dir / "final_pop_X.npy", np.array([]).reshape(0, problem.n_var))
        else:
            np.save(output_dir / "final_pop_X.npy", X_pop)
        if F_pop is None:
            np.save(output_dir / "final_pop_F.npy", np.array([]).reshape(0, problem.n_obj))
        else:
            np.save(output_dir / "final_pop_F.npy", F_pop)
        if G_pop is None:
            np.save(output_dir / "final_pop_G.npy", np.array([]).reshape(0, problem.N_CONSTR))
        else:
            np.save(output_dir / "final_pop_G.npy", G_pop)

        # Handle case where no solutions found
        if X is None or len(X) == 0:
            n_pareto = 0
            G = np.array([]).reshape(0, problem.N_CONSTR)
            X = np.array([]).reshape(0, problem.n_var)
            F = np.array([]).reshape(0, problem.n_obj)
            # Save empty arrays
            np.save(output_dir / "pareto_X.npy", X)
            np.save(output_dir / "pareto_F.npy", F)
            np.save(output_dir / "pareto_G.npy", G)
        else:
            # Get constraint values for Pareto solutions (reuse result.G if present)
            X = np.asarray(X, dtype=np.float64)
            F = np.asarray(F, dtype=np.float64)
            n_pareto = int(X.shape[0])
            if G_result is not None and getattr(G_result, "shape", (0,))[0] == n_pareto:
                G = np.asarray(G_result, dtype=np.float64)
            else:
                from ..core.evaluator import evaluate_candidate

                G = np.zeros((n_pareto, problem.N_CONSTR), dtype=np.float64)
                for i, x in enumerate(X):
                    try:
                        res = evaluate_candidate(x, ctx)
                        G[i] = res.G
                    except Exception:
                        G[i] = np.full(problem.N_CONSTR, 1.0e3, dtype=np.float64)

            order = _deterministic_order(F=F, X=X)
            if order.size == n_pareto and n_pareto > 0:
                X = X[order]
                F = F[order]
                G = G[order]

            # Save results
            np.save(output_dir / "pareto_X.npy", X)
            np.save(output_dir / "pareto_F.npy", F)
            np.save(output_dir / "pareto_G.npy", G)

    # Summary
    if n_pareto > 0 and F is not None:
        f_min = F.min(axis=0).tolist()
        f_max = F.max(axis=0).tolist()
        if F.shape[1] >= 3:
            best_eta_comb = float(1.0 - F[:, 0].min())
            best_eta_exp = float(1.0 - F[:, 1].min())
            best_eta_gear = float(1.0 - F[:, 2].min())
            eta_total = (1.0 - F[:, 0]) * (1.0 - F[:, 1]) * (1.0 - F[:, 2])
            best_eta_total = float(np.max(eta_total))
        else:
            best_eta_comb = 0.0
            best_eta_exp = 0.0
            best_eta_gear = 0.0
            best_eta_total = 0.0
    else:
        f_min = [0.0] * int(problem.n_obj)
        f_max = [0.0] * int(problem.n_obj)
        best_eta_comb = 0.0
        best_eta_exp = 0.0
        best_eta_gear = 0.0
        best_eta_total = 0.0

    contract_summary: dict[str, object] = {}
    if contract_summary_path.exists():
        try:
            contract_summary = json.loads(contract_summary_path.read_text(encoding="utf-8"))
        except Exception:
            contract_summary = {}

    summary = {
        "n_pareto": n_pareto,
        "n_final_pop": int(X_pop.shape[0]) if X_pop is not None else 0,
        "n_evals": problem.n_evals,
        "n_eval_errors": int(n_eval_errors),
        "eval_error_signatures": eval_error_signatures,
        "elapsed_s": t_elapsed,
        "pop_size": int(effective_pop),
        "requested_pop": int(requested_pop),
        "effective_pop": int(effective_pop),
        "requested_partitions": int(requested_partitions),
        "effective_partitions": int(effective_partitions),
        "n_ref_dirs": int(n_ref_dirs),
        "nsga3_max_ref_dirs": int(args.nsga3_max_ref_dirs),
        "algorithm": str(algorithm_used),
        "n_gen": args.gen,
        "rpm": args.rpm,
        "torque": args.torque,
        "fidelity": args.fidelity,
        "constraint_phase": args.constraint_phase,
        "tolerance_constraint_mode": args.tolerance_constraint_mode,
        "tolerance_threshold_mm": args.tolerance_threshold_mm,
        "seed": args.seed,
        "openfoam_model_path": args.openfoam_model_path,
        "calculix_stress_mode": args.calculix_stress_mode,
        "calculix_model_path": args.calculix_model_path,
        "gear_loss_mode": args.gear_loss_mode,
        "gear_loss_model_dir": args.gear_loss_model_dir,
        "tribology_scuff_method": str(args.tribology_scuff_method),
        "strict_data": bool(args.strict_data),
        "strict_tribology_data": args.strict_tribology_data,
        "production_profile": STRICT_PRODUCTION_PROFILE,
        "allow_nonproduction_paths": bool(args.allow_nonproduction_paths),
        "surrogate_validation_mode": str(args.surrogate_validation_mode),
        "machining_mode": str(args.machining_mode),
        "encoding_version": ENCODING_VERSION,
        "model_versions": {
            "thermo_v1": MODEL_VERSION_THERMO_V1,
            "gear_v1": MODEL_VERSION_GEAR_V1,
        },
        "n_obj": problem.n_obj,
        "objective_names": [
            "eta_comb_gap",
            "eta_exp_gap",
            "eta_gear_gap",
            "motion_law_penalty",
            "life_damage_penalty",
            "material_risk_penalty",
        ][: int(problem.n_obj)],
        "n_constr": problem.N_CONSTR,
        "constraint_names": get_constraint_names(args.fidelity),
        "constraint_scales": get_constraint_scales(),
        "F_min": f_min,
        "F_max": f_max,
        "n_feasible": int(np.sum(np.all(G <= 0, axis=1))) if len(G) > 0 else 0,
        "feasible_fraction": float(np.sum(np.all(G <= 0, axis=1)) / n_pareto)
        if n_pareto > 0
        else 0.0,
        "best_eta_comb": best_eta_comb,
        "best_eta_exp": best_eta_exp,
        "best_eta_gear": best_eta_gear,
        "best_eta_total": best_eta_total,
        "contract_version": CONTRACT_VERSION,
        "contract_trace_file": str(contract_trace_path),
        "contract_summary_file": str(contract_summary_path),
        "contract_summary": contract_summary,
    }

    fallback_paths_used: list[str] = []
    if int(n_pareto) == 0:
        fallback_paths_used.append("empty_pareto_front")
    if int(n_eval_errors) > 0:
        fallback_paths_used.append("candidate_eval_error_penalties")

    nonproduction_overrides: list[str] = []
    if bool(args.allow_nonproduction_paths):
        nonproduction_overrides.append("allow_nonproduction_paths")

    production_gate = evaluate_production_gate(
        production_profile=STRICT_PRODUCTION_PROFILE,
        require_pareto_metrics=True,
        allow_nonproduction_paths=bool(args.allow_nonproduction_paths),
        fallback_paths_used=fallback_paths_used,
        nonproduction_overrides=nonproduction_overrides,
        n_pareto=int(n_pareto),
        effective_pop=int(effective_pop),
        feasible_fraction=float(summary["feasible_fraction"]),
        n_eval_errors=int(n_eval_errors),
        algorithm_used=str(algorithm_used),
        fidelity=int(args.fidelity),
        constraint_phase=str(args.constraint_phase),
    )
    summary["production_gate"] = production_gate
    summary["production_gate_pass"] = bool(production_gate.get("production_gate_pass", False))
    summary["production_gate_failures"] = list(production_gate.get("production_gate_failures", []))
    summary["fallback_paths_used"] = list(production_gate.get("fallback_paths_used", []))
    summary["nonproduction_overrides"] = list(production_gate.get("nonproduction_overrides", []))
    summary["n_eval_errors"] = int(production_gate.get("n_eval_errors", n_eval_errors))
    summary["algorithm_used"] = str(production_gate.get("algorithm_used", algorithm_used))
    summary["constraint_phase"] = str(
        production_gate.get("constraint_phase", summary.get("constraint_phase", ""))
    )
    summary["fidelity"] = int(
        production_gate.get("fidelity", summary.get("fidelity", args.fidelity))
    )
    summary["production_profile"] = str(
        production_gate.get("production_profile", summary.get("production_profile", ""))
    )

    X_archive = (
        np.asarray(X, dtype=np.float64)
        if X is not None
        else np.zeros((0, int(problem.n_var)), dtype=np.float64)
    )
    F_archive = (
        np.asarray(F, dtype=np.float64)
        if F is not None
        else np.zeros((0, int(problem.n_obj)), dtype=np.float64)
    )
    if X_archive.ndim == 1:
        X_archive = X_archive.reshape(0, int(problem.n_var))
    if F_archive.ndim == 1:
        F_archive = F_archive.reshape(0, int(problem.n_obj))

    save_archive(output_dir, X_archive, F_archive, np.asarray(G, dtype=np.float64), summary)

    artifact_hashes = {
        "pareto_X_sha256": _file_sha256(output_dir / "pareto_X.npy"),
        "pareto_F_sha256": _file_sha256(output_dir / "pareto_F.npy"),
        "pareto_G_sha256": _file_sha256(output_dir / "pareto_G.npy"),
        "final_pop_X_sha256": _file_sha256(output_dir / "final_pop_X.npy"),
        "final_pop_F_sha256": _file_sha256(output_dir / "final_pop_F.npy"),
        "final_pop_G_sha256": _file_sha256(output_dir / "final_pop_G.npy"),
    }
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        try:
            persisted = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            persisted = {}
        persisted["artifact_hashes"] = artifact_hashes
        persisted["pareto_ordering"] = "lexicographic_F_then_X"
        summary_path.write_text(json.dumps(persisted, indent=2), encoding="utf-8")
        summary = persisted

    if args.verbose:
        print(f"\nCompleted in {t_elapsed:.1f}s")
        print(f"Pareto front: {n_pareto} solutions")
        print(f"Feasible: {summary['n_feasible']}")
        print(
            "Best (component) efficiencies: "
            f"η_comb={summary['best_eta_comb']:.3f}, "
            f"η_exp={summary['best_eta_exp']:.3f}, "
            f"η_gear={summary['best_eta_gear']:.3f}, "
            f"η_total={summary['best_eta_total']:.3f}"
        )
        if not bool(summary.get("production_gate_pass", False)):
            print(
                "Production gate failed: "
                f"{summary.get('production_gate', {}).get('production_gate_failures', [])}"
            )
        print(f"Results saved to: {output_dir}")

    if not bool(summary.get("production_gate_pass", False)) and not bool(
        args.allow_nonproduction_paths
    ):
        return 2
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
