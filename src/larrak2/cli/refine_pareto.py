"""Pareto refinement CLI with slice-aware CasADi/Ipopt backend."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ..core.artifact_paths import (
    resolve_stack_artifact_path,
    resolve_thermo_symbolic_artifact_path,
)
from ..core.constants import MODEL_VERSION_GEAR_V1, MODEL_VERSION_THERMO_V1
from ..core.constraints import get_constraint_names, get_constraint_scales
from ..core.encoding import ENCODING_VERSION, N_TOTAL


def _load_freeze_mask(path: str | None) -> np.ndarray | None:
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Freeze-mask path not found: {p}")

    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "freeze_mask" in payload:
                arr = np.asarray(payload["freeze_mask"])
            elif "freeze_indices" in payload:
                arr = np.zeros(N_TOTAL, dtype=bool)
                for idx in payload["freeze_indices"]:
                    arr[int(idx)] = True
            else:
                raise ValueError("JSON freeze mask must contain 'freeze_mask' or 'freeze_indices'")
        else:
            arr = np.asarray(payload)
    else:
        text = p.read_text(encoding="utf-8")
        tokens = [t for t in re.split(r"[\s,]+", text.strip()) if t]
        arr = np.zeros(N_TOTAL, dtype=bool)
        for tok in tokens:
            idx = int(tok)
            if idx < 0 or idx >= N_TOTAL:
                raise ValueError(f"freeze index out of range: {idx}")
            arr[idx] = True

    arr = np.asarray(arr)
    if arr.dtype == bool and arr.size == N_TOTAL:
        return arr

    if arr.ndim == 1 and arr.size == N_TOTAL and np.isin(arr, [0, 1]).all():
        return arr.astype(bool)

    # Treat as explicit list of frozen indices.
    mask = np.zeros(N_TOTAL, dtype=bool)
    for idx in arr.reshape(-1):
        ii = int(idx)
        if ii < 0 or ii >= N_TOTAL:
            raise ValueError(f"freeze index out of range: {ii}")
        mask[ii] = True
    return mask


def main(argv: list[str] | None = None) -> int:
    """Refine top-K Pareto candidates."""
    parser = argparse.ArgumentParser(description="Refine Pareto front candidates")
    parser.add_argument("--input", type=str, default=".", help="Input directory with pareto_*.npy")
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory (default: input)"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to refine")
    parser.add_argument(
        "--mode",
        type=str,
        default="eps_constraint",
        choices=["weighted_sum", "eps_constraint"],
        help="Refinement mode",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="casadi",
        choices=["casadi", "scipy"],
        help="Refinement backend",
    )
    parser.add_argument(
        "--slice-method",
        type=str,
        default="sensitivity",
        choices=["sensitivity"],
        help="Active-set slice selection method",
    )
    parser.add_argument(
        "--active-k",
        type=int,
        default=None,
        help="Number of active variables for local slice (default: max(6, ceil(0.25*N_TOTAL)))",
    )
    parser.add_argument(
        "--min-per-group",
        type=int,
        default=1,
        help="Minimum active variables per variable group",
    )
    parser.add_argument(
        "--freeze-mask-path",
        type=str,
        default=None,
        help="Optional path to bool freeze mask or frozen index list",
    )
    parser.add_argument("--ipopt-max-iter", type=int, default=None)
    parser.add_argument("--ipopt-tol", type=float, default=None)
    parser.add_argument("--ipopt-linear-solver", type=str, default=None)
    parser.add_argument(
        "--stack-model-path",
        type=str,
        default="",
        help="Global stack surrogate artifact path for casadi backend",
    )
    parser.add_argument("--fidelity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument(
        "--thermo-model",
        type=str,
        default="two_zone_eq_v1",
        choices=["two_zone_eq_v1"],
    )
    parser.add_argument(
        "--thermo-constants-path",
        type=str,
        default="",
        help="Override thermo literature constants path",
    )
    parser.add_argument(
        "--thermo-anchor-manifest",
        type=str,
        default="",
        help="Override thermo benchmark anchor manifest path",
    )
    parser.add_argument(
        "--trust-radius",
        type=float,
        default=None,
        help="Optional L-infinity trust region radius for active variables",
    )
    parser.add_argument(
        "--thermo-symbolic-mode",
        type=str,
        default="strict",
        choices=["strict", "warn", "off"],
        help="Thermo symbolic overlay mode for CasADi refinement",
    )
    parser.add_argument(
        "--thermo-symbolic-artifact-path",
        type=str,
        default="",
        help="Thermo symbolic artifact path for CasADi overlay",
    )

    parser.add_argument("--rpm", type=float, default=3000.0, help="Engine speed (rpm)")
    parser.add_argument("--torque", type=float, default=200.0, help="Torque demand (Nm)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args(argv)

    from ..adapters.casadi_refine import RefinementMode, refine_candidate
    from ..core.types import EvalContext

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir

    X = np.load(input_dir / "pareto_X.npy")
    F = np.load(input_dir / "pareto_F.npy")

    if args.verbose:
        print(f"Loaded {X.shape[0]} Pareto solutions")

    indices = np.argsort(F[:, 0])[: args.top_k]
    try:
        resolved_thermo_symbolic_path = resolve_thermo_symbolic_artifact_path(
            fidelity=int(args.fidelity),
            explicit_path=str(args.thermo_symbolic_artifact_path).strip() or None,
            must_exist=(
                args.backend == "casadi"
                and str(args.thermo_symbolic_mode).strip().lower() == "strict"
            ),
        )
    except Exception as exc:
        print(str(exc))
        return 1

    ctx = EvalContext(
        rpm=args.rpm,
        torque=args.torque,
        fidelity=int(args.fidelity),
        seed=42,
        thermo_model=str(args.thermo_model),
        thermo_constants_path=str(args.thermo_constants_path).strip() or None,
        thermo_anchor_manifest_path=str(args.thermo_anchor_manifest).strip() or None,
        thermo_symbolic_mode=str(args.thermo_symbolic_mode),
        thermo_symbolic_artifact_path=str(resolved_thermo_symbolic_path),
    )
    mode = RefinementMode(args.mode)

    freeze_mask = _load_freeze_mask(args.freeze_mask_path)
    try:
        stack_model_path = str(
            resolve_stack_artifact_path(
                fidelity=int(args.fidelity),
                explicit_path=str(args.stack_model_path).strip() or None,
                must_exist=args.backend == "casadi",
            )
        )
    except Exception as exc:
        print(str(exc))
        return 1

    ipopt_options = {
        k: v
        for k, v in {
            "max_iter": args.ipopt_max_iter,
            "tol": args.ipopt_tol,
            "linear_solver": args.ipopt_linear_solver,
        }.items()
        if v is not None
    }

    refined_X = []
    refined_F = []
    refined_G = []
    results_diag = []
    had_casadi_failures = False

    for i, idx in enumerate(indices):
        x0 = X[idx]

        if args.verbose:
            print(f"Refining candidate {i + 1}/{len(indices)} (original F={F[idx]})")

        result = refine_candidate(
            x0,
            ctx,
            mode=mode,
            backend=args.backend,
            active_k=args.active_k,
            min_per_group=args.min_per_group,
            slice_method=args.slice_method,
            freeze_mask=freeze_mask,
            ipopt_options=ipopt_options or None,
            trust_radius=args.trust_radius,
            surrogate_stack_path=stack_model_path if args.backend == "casadi" else None,
            fidelity=int(args.fidelity),
        )

        refined_X.append(result.x_refined)
        refined_F.append(result.F_refined)
        refined_G.append(result.G_refined)

        candidate_diag = {
            "original_idx": int(idx),
            "original_F": F[idx].tolist(),
            "refined_F": result.F_refined.tolist(),
            "success": result.success,
            "message": result.message,
            "backend_used": result.backend_used,
            "ipopt_status": result.ipopt_status,
            "active_indices": result.diag.get("active_indices", []),
            "frozen_indices": result.diag.get("frozen_indices", []),
            "slice_scores": result.diag.get("slice_scores", []),
            "nlp_formulation": result.diag.get("nlp_formulation", ""),
            "surrogate_stack_version": result.diag.get("surrogate_stack_version", ""),
            "validation_attempts": result.diag.get("validation_attempts", 0),
            "trust_radius_final": result.diag.get("trust_radius_final", args.trust_radius),
            "thermo_symbolic_mode": result.diag.get("thermo_symbolic_mode", ""),
            "thermo_symbolic_used": bool(result.diag.get("thermo_symbolic_used", False)),
            "thermo_symbolic_version": result.diag.get("thermo_symbolic_version", ""),
            "thermo_symbolic_path": result.diag.get("thermo_symbolic_path", ""),
            "thermo_symbolic_overlay_objectives": result.diag.get(
                "thermo_symbolic_overlay_objectives", []
            ),
            "thermo_symbolic_overlay_constraints": result.diag.get(
                "thermo_symbolic_overlay_constraints", []
            ),
            "thermo_symbolic_error": result.diag.get("thermo_symbolic_error", ""),
        }
        results_diag.append(candidate_diag)
        if args.backend == "casadi" and not bool(result.success):
            had_casadi_failures = True

        if args.verbose:
            print(
                "  -> refined F="
                f"{result.F_refined}, success={result.success}, backend={result.backend_used}"
            )

    np.save(output_dir / "refined_X.npy", np.array(refined_X, dtype=np.float64))
    np.save(output_dir / "refined_F.npy", np.array(refined_F, dtype=np.float64))
    np.save(output_dir / "refined_G.npy", np.array(refined_G, dtype=np.float64))

    with open(output_dir / "refinement_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "backend": args.backend,
                "slice_method": args.slice_method,
                "active_k": args.active_k,
                "min_per_group": args.min_per_group,
                "freeze_mask_path": args.freeze_mask_path,
                "ipopt_options": ipopt_options,
                "stack_model_path": stack_model_path if args.backend == "casadi" else None,
                "trust_radius": args.trust_radius,
                "rpm": args.rpm,
                "torque": args.torque,
                "fidelity": ctx.fidelity,
                "thermo_model": ctx.thermo_model,
                "thermo_constants_path": ctx.thermo_constants_path,
                "thermo_anchor_manifest_path": ctx.thermo_anchor_manifest_path,
                "thermo_symbolic_mode": ctx.thermo_symbolic_mode,
                "thermo_symbolic_artifact_path": ctx.thermo_symbolic_artifact_path,
                "seed": ctx.seed,
                "encoding_version": ENCODING_VERSION,
                "model_versions": {
                    "thermo_v1": MODEL_VERSION_THERMO_V1,
                    "gear_v1": MODEL_VERSION_GEAR_V1,
                },
                "constraint_names": get_constraint_names(ctx.fidelity),
                "constraint_scales": get_constraint_scales(),
                "n_refined": len(refined_X),
                "results": results_diag,
            },
            f,
            indent=2,
        )

    if args.verbose:
        print(f"Refined {len(refined_X)} candidates")
        print(f"Results saved to: {output_dir}")
        if args.backend == "casadi" and had_casadi_failures:
            print("CasADi strict mode: one or more candidates failed.")

    if args.backend == "casadi" and had_casadi_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
