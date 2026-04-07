"""CEM validation for Pareto-optimal candidates.

Post-optimization command that validates Pareto-optimal candidates using the
full CEM (no surrogates). Produces a ranked report with recommendations.

Usage:
    larrak-validate --archive results/pareto_front.npz
    larrak-validate --archive results/pareto_front.npz --top 5 --output report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CandidateValidation:
    """Validation result for a single Pareto candidate."""

    index: int
    # Surrogate predictions (from optimization)
    surr_lambda_min: float = 0.0
    surr_scuff_margin_C: float = 0.0
    surr_micropitting_sf: float = 0.0
    # CEM predictions (authoritative)
    cem_lambda_min: float = 0.0
    cem_scuff_margin_C: float = 0.0
    cem_micropitting_sf: float = 0.0
    cem_material_temp_margin_C: float = 0.0
    cem_cost_index: float = 0.0
    cem_lube_regime: str = ""
    # Drift (surrogate vs CEM)
    drift_lambda_pct: float = 0.0
    drift_scuff_pct: float = 0.0
    # Feasibility
    is_feasible: bool = False
    feasibility_score: float = 0.0
    # Recommendation
    recommendation: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a Pareto front."""

    n_candidates: int
    n_feasible: int
    ranked_candidates: list[CandidateValidation]
    timestamp: str = ""
    archive_path: str = ""
    encoding_version: str = ""


def validate_candidates(
    X: np.ndarray,
    F: np.ndarray | None = None,
) -> ValidationReport:
    """Validate an array of Pareto-optimal candidate vectors.

    Args:
        X: Decision vectors, shape (n, 22).
        F: Optional objective values, shape (n, n_obj).

    Returns:
        ValidationReport with ranked candidates.
    """
    from larrak_runtime.core.encoding import ENCODING_VERSION, N_TOTAL, decode_candidate

    from ..cem.evaluator import CEMEvalParams, evaluate_cem
    from ..cem.lubrication import LubricationParams, mode_from_level
    from ..cem.post_processing import coating_from_level
    from ..cem.surface_finish import tier_from_level
    from ..realworld.surrogates import (
        RealWorldSurrogateParams,
        _material_from_level,
        evaluate_realworld_surrogates,
    )

    n = X.shape[0]
    results: list[CandidateValidation] = []

    for i in range(n):
        x = X[i]
        if len(x) != N_TOTAL:
            logger.warning("Candidate %d has %d vars (expected %d), skipping", i, len(x), N_TOTAL)
            continue

        candidate = decode_candidate(x)
        rw = candidate.realworld

        # --- Surrogate evaluation (for comparison) ---
        surr_params = RealWorldSurrogateParams(
            surface_finish_level=rw.surface_finish_level,
            lube_mode_level=rw.lube_mode_level,
            material_quality_level=rw.material_quality_level,
            coating_level=rw.coating_level,
            oil_flow_level=rw.oil_flow_level,
            oil_supply_temp_level=rw.oil_supply_temp_level,
            evacuation_level=rw.evacuation_level,
        )
        surr_result = evaluate_realworld_surrogates(surr_params)

        # --- Full CEM evaluation ---
        lube_mode = mode_from_level(rw.lube_mode_level)
        flow_rate = 0.5 + rw.oil_flow_level * 9.5
        supply_temp = 40.0 + rw.oil_supply_temp_level * 80.0

        cem_params = CEMEvalParams(
            material=_material_from_level(rw.material_quality_level),
            surface_finish=tier_from_level(rw.surface_finish_level),
            lubrication=LubricationParams(
                mode=lube_mode,
                supply_temp_C=supply_temp,
                flow_rate_L_min=flow_rate,
            ),
            coating=coating_from_level(rw.coating_level),
        )
        cem_result = evaluate_cem(cem_params)

        # --- Drift calculation ---
        drift_lambda = 0.0
        if abs(surr_result.lambda_min) > 1e-6:
            drift_lambda = (
                (cem_result.lambda_min - surr_result.lambda_min) / surr_result.lambda_min * 100.0
            )

        drift_scuff = 0.0
        if abs(surr_result.scuff_margin_C) > 1e-6:
            drift_scuff = (
                (cem_result.scuff_margin_C - surr_result.scuff_margin_C)
                / abs(surr_result.scuff_margin_C)
                * 100.0
            )

        # --- Feasibility ---
        feasible_lambda = cem_result.lambda_min >= 1.0
        feasible_scuff = cem_result.scuff_margin_C > 0
        feasible_micro = cem_result.micropitting_safety >= 1.0
        feasible_temp = cem_result.details.get("material", {}).get("temp_margin_C", 0) > 0
        is_feasible = feasible_lambda and feasible_scuff and feasible_micro and feasible_temp

        # Composite feasibility score (higher = better)
        score = (
            min(cem_result.lambda_min, 3.0) / 3.0 * 25.0  # λ contribution
            + min(max(cem_result.scuff_margin_C, 0) / 200.0, 1.0) * 25.0  # scuff contribution
            + min(cem_result.micropitting_safety, 5.0) / 5.0 * 25.0  # micropitting contribution
            + (25.0 if feasible_temp else 0.0)
        )

        # --- Recommendation ---
        ranking = cem_result.recommendation_ranking
        rec_parts = [f"{r[0]}" for r in ranking[:4]]
        recommendation = " > ".join(rec_parts) if rec_parts else "no recommendation"

        results.append(
            CandidateValidation(
                index=i,
                surr_lambda_min=surr_result.lambda_min,
                surr_scuff_margin_C=surr_result.scuff_margin_C,
                surr_micropitting_sf=surr_result.micropitting_safety,
                cem_lambda_min=cem_result.lambda_min,
                cem_scuff_margin_C=cem_result.scuff_margin_C,
                cem_micropitting_sf=cem_result.micropitting_safety,
                cem_material_temp_margin_C=float(
                    cem_result.details.get("material", {}).get("temp_margin_C", 0)
                ),
                cem_cost_index=cem_result.total_cost_index,
                cem_lube_regime=cem_result.lube_regime,
                drift_lambda_pct=drift_lambda,
                drift_scuff_pct=drift_scuff,
                is_feasible=is_feasible,
                feasibility_score=score,
                recommendation=recommendation,
                details=cem_result.details,
            )
        )

    # Rank by feasibility score (descending)
    results.sort(key=lambda r: r.feasibility_score, reverse=True)

    return ValidationReport(
        n_candidates=n,
        n_feasible=sum(1 for r in results if r.is_feasible),
        ranked_candidates=results,
        encoding_version=ENCODING_VERSION,
    )


def format_report(report: ValidationReport) -> str:
    """Format a validation report as human-readable text."""
    lines = [
        "CEM Validation Report",
        f"{'=' * 60}",
        f"Candidates: {report.n_candidates}  |  Feasible: {report.n_feasible}",
        f"Encoding: v{report.encoding_version}",
        "",
    ]

    for rank, cand in enumerate(report.ranked_candidates, 1):
        status = "✓" if cand.is_feasible else "✗"
        lines.append(
            f"Candidate #{cand.index} (rank {rank}) {status}  [score: {cand.feasibility_score:.1f}/100]"
        )
        lines.append(
            f"  λ_min = {cand.cem_lambda_min:.3f} (surrogate: {cand.surr_lambda_min:.3f}, drift: {cand.drift_lambda_pct:+.1f}%)"
        )
        lines.append(
            f"  Scuff margin = {cand.cem_scuff_margin_C:.1f}°C (surrogate: {cand.surr_scuff_margin_C:.1f}°C)"
        )
        lines.append(f"  Micropitting S_λ = {cand.cem_micropitting_sf:.2f}")
        lines.append(f"  Lube regime: {cand.cem_lube_regime}")
        lines.append(f"  Recommendation: {cand.recommendation}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for larrak-validate."""
    parser = argparse.ArgumentParser(
        prog="larrak-validate",
        description="Validate Pareto-optimal candidates with full CEM evaluation.",
    )
    parser.add_argument(
        "--archive",
        required=True,
        type=str,
        help="Path to Pareto archive (.npz or .json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Only validate top N candidates (0 = all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to save JSON report (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load archive
    archive_path = Path(args.archive)
    if not archive_path.exists():
        logger.error("Archive not found: %s", archive_path)
        return 1

    if archive_path.suffix == ".npz":
        data = np.load(str(archive_path), allow_pickle=True)
        X = data["X"]
        F = data.get("F")
    elif archive_path.suffix == ".json":
        with open(archive_path) as f:
            jdata = json.load(f)
        X = np.array(jdata["X"])
        F = np.array(jdata["F"]) if "F" in jdata else None
    else:
        logger.error("Unsupported archive format: %s", archive_path.suffix)
        return 1

    if args.top > 0:
        X = X[: args.top]
        if F is not None:
            F = F[: args.top]

    logger.info("Validating %d candidates from %s", X.shape[0], archive_path)
    t0 = time.perf_counter()

    report = validate_candidates(X, F)
    report.archive_path = str(archive_path)
    report.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    elapsed = time.perf_counter() - t0
    logger.info("Validation complete in %.2f s", elapsed)

    # Print report
    print(format_report(report))

    # Save JSON if requested
    if args.output:
        out_path = Path(args.output)
        report_dict = {
            "n_candidates": report.n_candidates,
            "n_feasible": report.n_feasible,
            "timestamp": report.timestamp,
            "archive_path": report.archive_path,
            "encoding_version": report.encoding_version,
            "candidates": [
                {
                    "index": c.index,
                    "cem_lambda_min": c.cem_lambda_min,
                    "cem_scuff_margin_C": c.cem_scuff_margin_C,
                    "cem_micropitting_sf": c.cem_micropitting_sf,
                    "cem_material_temp_margin_C": c.cem_material_temp_margin_C,
                    "cem_cost_index": c.cem_cost_index,
                    "cem_lube_regime": c.cem_lube_regime,
                    "surr_lambda_min": c.surr_lambda_min,
                    "drift_lambda_pct": c.drift_lambda_pct,
                    "is_feasible": c.is_feasible,
                    "feasibility_score": c.feasibility_score,
                    "recommendation": c.recommendation,
                }
                for c in report.ranked_candidates
            ],
        }
        out_path.write_text(json.dumps(report_dict, indent=2))
        logger.info("Report saved to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
