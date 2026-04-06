"""CLI entry point for canonical-regime simulation validation.

Commands:
    larrak-validate-sim chemistry
    larrak-validate-sim spray
    larrak-validate-sim reacting-flow
    larrak-validate-sim closed-cylinder
    larrak-validate-sim full-handoff
    larrak-validate-sim runtime-chemistry-table
    larrak-validate-sim engine-restart-benchmark
    larrak-validate-sim coverage-corpus-analysis
    larrak-validate-sim restart-regression-analysis
    larrak-validate-sim tuning-characterization --mode ingest|propose|run-batch
    larrak-validate-sim chemistry-cache
    larrak-validate-sim flame-speed-compare
    larrak-validate-sim suite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_regime_config(config_path: Path) -> dict[str, Any]:
    """Load a regime configuration JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _extract_chemistry_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the chemistry regime config from either a regime or suite config."""
    if "regimes" in config:
        chemistry_cfg = dict(config.get("regimes", {}).get("chemistry", {}) or {})
        if not chemistry_cfg:
            raise ValueError("Suite config does not define a chemistry regime")
        return chemistry_cfg
    return config


def _build_dataset_and_case(
    config: dict[str, Any],
    regime: str,
) -> tuple:
    """Build ValidationDatasetManifest and ValidationCaseSpec from config dict."""
    from larrak2.simulation_validation.models import (
        ComparisonMode,
        SourceType,
        ValidationCaseSpec,
        ValidationDatasetManifest,
        ValidationMetricSpec,
    )

    ds_cfg = config.get("dataset", {})
    metrics = []
    for m in ds_cfg.get("metrics", []):
        metrics.append(
            ValidationMetricSpec(
                metric_id=str(m["metric_id"]),
                units=str(m.get("units", "")),
                comparison_mode=ComparisonMode(m.get("comparison_mode", "absolute")),
                tolerance_band=float(m.get("tolerance_band", 0.0)),
                source_type=SourceType(m.get("source_type", "measured")),
                required=bool(m.get("required", True)),
                description=str(m.get("description", "")),
            )
        )

    dataset = ValidationDatasetManifest(
        dataset_id=str(ds_cfg.get("dataset_id", f"{regime}_default")),
        regime=regime,
        fuel_family=str(ds_cfg.get("fuel_family", "gasoline")),
        source_type=SourceType(ds_cfg.get("source_type", "measured")),
        provenance=dict(ds_cfg.get("provenance", {})),
        operating_bounds=dict(ds_cfg.get("operating_bounds", {})),
        metrics=metrics,
        measured_anchor_ids=list(ds_cfg.get("measured_anchor_ids", [])),
        governing_basis=str(ds_cfg.get("governing_basis", "")),
        literature_reference=str(ds_cfg.get("literature_reference", "")),
        standard_reference=str(ds_cfg.get("standard_reference", "")),
    )

    cs_cfg = config.get("case_spec", {})
    case_spec = ValidationCaseSpec(
        case_id=str(cs_cfg.get("case_id", f"{regime}_case_1")),
        regime=regime,
        operating_point=dict(cs_cfg.get("operating_point", {})),
        geometry_revision=str(cs_cfg.get("geometry_revision", "")),
        motion_profile_revision=str(cs_cfg.get("motion_profile_revision", "")),
        solver_config=dict(cs_cfg.get("solver_config", {})),
        dataset_binding=str(cs_cfg.get("dataset_binding", "")),
    )

    return dataset, case_spec


def _build_suite_profile(config: dict[str, Any]):
    """Build a ValidationSuiteProfile from a suite config dict."""
    from larrak2.simulation_validation.models import ValidationSuiteProfile
    from larrak2.simulation_validation.regimes import (
        CanonicalRegime,
        canonical_prerequisite_names,
    )

    regime_order = list(config.get("regime_order", []) or [])
    if not regime_order:
        regime_order = CanonicalRegime.ordered_names()

    prerequisites_raw = dict(config.get("prerequisites", {}) or {})
    prerequisites = {
        str(regime_name): [str(dep) for dep in deps]
        for regime_name, deps in prerequisites_raw.items()
    }
    if not prerequisites:
        prerequisites = canonical_prerequisite_names()

    return ValidationSuiteProfile(
        suite_id=str(config.get("suite_id", "canonical_v1")),
        regime_order=regime_order,
        prerequisites=prerequisites,
        description=str(config.get("description", "")),
    )


def _write_regime_artifacts(
    *,
    regime_name: str,
    run_manifest,
    outdir: Path,
    elapsed: float,
) -> tuple[Path, Path]:
    from larrak2.simulation_validation.plotting import (
        generate_error_summary_plot,
        generate_metric_comparison_plot,
    )

    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = outdir / f"{regime_name}_run_manifest.json"
    manifest_dict = {
        "regime": run_manifest.regime,
        "status": run_manifest.status.value,
        "metrics": [
            {
                "metric_id": r.metric_id,
                "measured_value": r.measured_value,
                "simulated_value": r.simulated_value,
                "error": r.error,
                "tolerance_used": r.tolerance_used,
                "passed": r.passed,
                "source_type": r.source_type.value,
                "units": r.units,
            }
            for r in run_manifest.metric_results
        ],
        "solver_artifacts": run_manifest.solver_artifacts,
        "messages": run_manifest.messages,
    }
    manifest_path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
    logger.info("Manifest: %s", manifest_path)

    if run_manifest.metric_results:
        generate_metric_comparison_plot(
            run_manifest.metric_results,
            f"{regime_name} — Metric Comparison",
            outdir / f"{regime_name}_comparison.html",
        )
        generate_error_summary_plot(
            run_manifest.metric_results,
            f"{regime_name} — Error vs Tolerance",
            outdir / f"{regime_name}_errors.html",
        )

    summary_lines = [
        f"# {regime_name.replace('_', ' ').title()} Validation Report",
        "",
        f"**Status:** {run_manifest.status.value}",
        f"**Elapsed:** {elapsed:.2f}s",
        "",
    ]
    if run_manifest.metric_results:
        summary_lines.append("| Metric | Measured | Simulated | Error | Tol | Pass |")
        summary_lines.append("|--------|----------|-----------|-------|-----|------|")
        for r in run_manifest.metric_results:
            icon = "✅" if r.passed else "❌"
            summary_lines.append(
                f"| {r.metric_id} | {r.measured_value:.4g} | "
                f"{r.simulated_value:.4g} | {r.error:.4g} | "
                f"{r.tolerance_used:.4g} | {icon} |"
            )
        summary_lines.append("")

    md_path = outdir / f"{regime_name}_summary.md"
    md_path.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Summary: %s", md_path)
    return manifest_path, md_path


def _run_single_regime_cmd(args: argparse.Namespace, regime_name: str) -> int:
    """Execute a single regime validation from CLI args."""
    from larrak2.simulation_validation.regimes import CanonicalRegime
    from larrak2.simulation_validation.suite import (
        run_single_regime,
    )

    config_path = Path(args.config)
    config = _load_regime_config(config_path)

    dataset, case_spec = _build_dataset_and_case(config, regime_name)
    simulation_data = config.get("simulation_data", {})

    regime = CanonicalRegime(regime_name)

    logger.info("Running %s validation...", regime_name)
    t0 = time.perf_counter()

    run_manifest = run_single_regime(regime, dataset, case_spec, simulation_data)

    elapsed = time.perf_counter() - t0
    logger.info("Completed in %.2fs: %s", elapsed, run_manifest.status.value)

    # Output artifacts
    outdir = Path(args.outdir)
    _write_regime_artifacts(
        regime_name=regime_name,
        run_manifest=run_manifest,
        outdir=outdir,
        elapsed=elapsed,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  {regime_name.upper()} VALIDATION: {run_manifest.status.value}")
    print(
        f"  Metrics: {sum(1 for r in run_manifest.metric_results if r.passed)}/"
        f"{len(run_manifest.metric_results)} passed"
    )
    print(f"{'=' * 60}\n")

    return 0 if run_manifest.passed else 1


def _run_chemistry_cache_cmd(args: argparse.Namespace) -> int:
    """Build or refresh the persistent offline chemistry-results cache."""
    from larrak2.simulation_validation.regimes import CanonicalRegime
    from larrak2.simulation_validation.suite import run_single_regime

    config_path = Path(args.config)
    config = _extract_chemistry_config(_load_regime_config(config_path))

    case_cfg = dict(config.get("case_spec", {}) or {})
    solver_cfg = dict(case_cfg.get("solver_config", {}) or {})
    adapter_cfg = dict(solver_cfg.get("simulation_adapter", {}) or {})
    if not adapter_cfg:
        raise ValueError(
            "Chemistry cache build requires case_spec.solver_config.simulation_adapter"
        )

    adapter_cfg["backend"] = "native_cantera"
    adapter_cfg["offline_results_only"] = False
    adapter_cfg["refresh_offline_results"] = bool(args.refresh)
    if str(getattr(args, "output", "")).strip():
        adapter_cfg["offline_results_path"] = str(args.output).strip()
    elif not str(adapter_cfg.get("offline_results_path", "")).strip():
        adapter_cfg["offline_results_path"] = str(
            Path("outputs")
            / "validation_runtime"
            / "chemistry_cache"
            / "chemistry_offline_results.json"
        )
    solver_cfg["simulation_adapter"] = adapter_cfg
    case_cfg["solver_config"] = solver_cfg
    config["case_spec"] = case_cfg

    dataset, case_spec = _build_dataset_and_case(config, "chemistry")
    simulation_data = dict(config.get("simulation_data", {}) or {})

    logger.info("Building chemistry offline cache...")
    t0 = time.perf_counter()
    run_manifest = run_single_regime(
        CanonicalRegime.CHEMISTRY,
        dataset,
        case_spec,
        simulation_data,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Completed in %.2fs: %s", elapsed, run_manifest.status.value)

    outdir = Path(args.outdir)
    _write_regime_artifacts(
        regime_name="chemistry",
        run_manifest=run_manifest,
        outdir=outdir,
        elapsed=elapsed,
    )

    cache_path = run_manifest.solver_artifacts.get("chemistry_offline_results", "")
    if cache_path:
        print(f"Offline chemistry cache: {cache_path}")
    print(f"\n{'=' * 60}")
    print(f"  CHEMISTRY CACHE BUILD: {run_manifest.status.value}")
    print(
        f"  Metrics: {sum(1 for r in run_manifest.metric_results if r.passed)}/"
        f"{len(run_manifest.metric_results)} passed"
    )
    print(f"{'=' * 60}\n")
    return 0 if run_manifest.passed else 1


def _run_suite_cmd(args: argparse.Namespace) -> int:
    """Execute the full validation suite from CLI args."""
    from larrak2.simulation_validation.suite import (
        run_suite,
        suite_to_json,
        suite_to_markdown,
    )

    config_path = Path(args.config)
    config = _load_regime_config(config_path)
    suite_profile = _build_suite_profile(config)

    # Build per-regime configs
    regime_configs: dict[str, dict[str, Any]] = {}
    for regime_name, regime_cfg in config.get("regimes", {}).items():
        dataset, case_spec = _build_dataset_and_case(regime_cfg, regime_name)
        regime_configs[regime_name] = {
            "dataset": dataset,
            "case_spec": case_spec,
            "simulation_data": regime_cfg.get("simulation_data", {}),
        }

    logger.info("Running full validation suite (%d regimes configured)", len(regime_configs))
    t0 = time.perf_counter()

    suite_manifest = run_suite(regime_configs, suite_profile=suite_profile)

    elapsed = time.perf_counter() - t0
    logger.info("Suite completed in %.2fs", elapsed)

    # Output artifacts
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    suite_to_json(suite_manifest, outdir / "suite_manifest.json")
    md = suite_to_markdown(suite_manifest)
    (outdir / "suite_summary.md").write_text(md, encoding="utf-8")

    print(md)

    return 0 if suite_manifest.overall_passed else 1


def _run_combustion_truth_cmd(args: argparse.Namespace) -> int:
    """Execute the gas-combustion truth workflow over the DOE core corridor."""
    from larrak2.simulation_validation.combustion_truth import (
        DEFAULT_PROFILE_PATH,
        run_combustion_truth_workflow,
    )

    summary = run_combustion_truth_workflow(
        suite_config_path=args.config,
        outdir=args.outdir,
        profile_path=getattr(args, "profile", str(DEFAULT_PROFILE_PATH)),
        max_points=getattr(args, "max_points", None),
    )
    print(json.dumps(summary, indent=2))
    return 0 if int(summary.get("n_points", 0)) == int(summary.get("n_passed", 0)) else 1


def _run_flame_speed_compare_cmd(args: argparse.Namespace) -> int:
    """Compare flame-speed tractability across mechanism candidates."""
    from larrak2.simulation_validation.flame_speed_comparison import (
        compare_flame_speed_mechanisms,
    )

    summary = compare_flame_speed_mechanisms(
        config_path=args.config,
        outdir=args.outdir,
        refresh=bool(getattr(args, "refresh", False)),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _run_runtime_chemistry_table_cmd(args: argparse.Namespace) -> int:
    from larrak2.simulation_validation.runtime_chemistry_table import build_runtime_chemistry_table

    manifest = build_runtime_chemistry_table(
        config_path=args.config,
        refresh=bool(args.refresh),
        repo_root=Path.cwd(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _run_coverage_corpus_analysis_cmd(args: argparse.Namespace) -> int:
    from larrak2.simulation_validation.coverage_corpus_analysis import (
        analyze_coverage_corpus_vs_targets,
    )

    extra = [str(p).strip() for p in (getattr(args, "extra_corpus", []) or []) if str(p).strip()]
    miss = str(getattr(args, "authority_miss", "") or "").strip()
    result = analyze_coverage_corpus_vs_targets(
        table_config_path=args.config,
        repo_root=Path.cwd(),
        extra_corpus_paths=extra or None,
        authority_miss_path=miss or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_engine_restart_benchmark_cmd(args: argparse.Namespace) -> int:
    from larrak2.simulation_validation.engine_restart_benchmark import (
        benchmark_engine_restart_profiles,
    )

    summary = benchmark_engine_restart_profiles(
        run_dir=args.run_dir,
        tuned_params_path=args.tuned_params,
        handoff_artifact_path=args.handoff_artifact,
        outdir=args.outdir,
        profiles=list(args.profiles),
        window_angle_deg=float(args.window_angle_deg),
        solver_name=str(args.solver_name),
        docker_timeout_s=int(args.docker_timeout_s),
        runtime_strategy_config=args.runtime_strategy_config,
        package_label=str(args.package_label),
        docker_image=args.docker_image,
        docker_bin=args.docker_bin,
        refresh_runtime_tables=bool(args.refresh_runtime_tables),
        continue_across_remaining_stages=bool(args.continue_across_stages),
        refresh_custom_solver=bool(args.refresh_custom_solver),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_tuning_characterization_cmd(args: argparse.Namespace) -> int:
    from larrak2.simulation_validation.tuning_characterization_study import (
        STRATEGY_REGISTRY,
        append_experiments_jsonl,
        ingest_benchmark_run_directory,
        load_experiments_jsonl,
        load_knob_schema,
        resolve_run_directories,
        run_tuning_batch,
    )

    mode = str(args.tc_mode).strip().lower()
    if mode == "ingest":
        if (
            not list(getattr(args, "runs", []) or [])
            and not str(getattr(args, "glob", "") or "").strip()
        ):
            logger.error("ingest mode requires --runs and/or --glob")
            return 2
        resolved = resolve_run_directories(
            runs=list(getattr(args, "runs", []) or []),
            glob_pattern=str(getattr(args, "glob", "") or ""),
            latest=getattr(args, "latest", None),
        )
        profile = str(getattr(args, "profile_name", "") or "").strip() or None
        records = [ingest_benchmark_run_directory(p, profile_name=profile) for p in resolved]
        out_path = Path(str(args.experiments_jsonl))
        append_experiments_jsonl(out_path, records)
        print(json.dumps({"ingested": len(records), "path": str(out_path.resolve())}, indent=2))
        return 0

    if mode == "propose":
        schema = load_knob_schema(args.knob_schema)
        experiments = (
            load_experiments_jsonl(args.experiments_jsonl)
            if Path(args.experiments_jsonl).exists()
            else []
        )
        strat_name = str(args.strategy).strip().lower()
        if strat_name not in STRATEGY_REGISTRY:
            logger.error("Unknown strategy %s", strat_name)
            return 2
        import numpy as np

        rng = np.random.default_rng(int(args.rng_seed))
        proposals = STRATEGY_REGISTRY[strat_name]().propose(
            knob_schema=schema,
            experiments=experiments,
            n=int(args.n_proposals),
            rng=rng,
        )
        print(json.dumps({"proposals": proposals}, indent=2, sort_keys=True))
        return 0

    if mode == "run-batch":
        summary = run_tuning_batch(
            knob_schema_path=args.knob_schema,
            base_table_config_path=args.base_table_config,
            strategy_config_path=str(args.strategy_config).strip() or None,
            experiments_jsonl_path=args.experiments_jsonl,
            study_outdir=args.study_outdir,
            search_strategy=str(args.strategy).strip().lower(),
            n_trials=int(args.n_trials),
            refresh_table=bool(args.refresh_table),
            run_benchmark=bool(args.run_benchmark),
            benchmark_run_dir=str(args.benchmark_run_dir).strip() or None,
            tuned_params_path=str(args.tuned_params).strip() or None,
            handoff_artifact_path=str(args.handoff_artifact).strip() or None,
            runtime_strategy_config=str(args.runtime_strategy_config).strip() or None,
            benchmark_profiles=list(args.benchmark_profiles or []) or None,
            window_angle_deg=float(args.window_angle_deg),
            docker_timeout_s=int(args.docker_timeout_s),
            refresh_custom_solver=bool(args.refresh_custom_solver),
            max_benchmarks=int(args.max_benchmarks),
            dry_run=bool(args.dry_run),
            profile_name=str(args.profile_name).strip() or None,
            rng_seed=int(args.rng_seed),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    logger.error("Unknown tuning-characterization mode %s", mode)
    return 2


def _run_restart_regression_analysis_cmd(args: argparse.Namespace) -> int:
    """Analyze ordered restart benchmark runs without depending on solver internals."""
    from larrak2.simulation_validation.restart_regression_suite import (
        analyze_restart_regression_runs,
        write_restart_regression_artifacts,
    )

    analysis = analyze_restart_regression_runs(
        run_dirs=list(getattr(args, "runs", []) or []),
        glob_pattern=str(getattr(args, "glob", "") or ""),
        latest=getattr(args, "latest", None),
        profile_name=str(getattr(args, "profile_name", "") or "").strip() or None,
        history_window=int(getattr(args, "history_window", 5)),
    )
    written = write_restart_regression_artifacts(
        analysis=analysis,
        outdir=args.outdir,
        suite=str(getattr(args, "suite", "all")),
    )
    print(json.dumps({"general": analysis["general"], "artifacts": written}, indent=2))
    return 0


def run_validation_preflight(
    regime_or_suite: str,
    *,
    config_path: str | Path,
    outdir: str | Path,
) -> int:
    """Run validation from a config path for reuse by other CLIs.

    Accepts canonical regime names with underscores, CLI names with hyphens,
    or the special value ``suite``.
    """
    command = str(regime_or_suite).strip()
    if not command:
        raise ValueError("regime_or_suite must be provided")

    args = argparse.Namespace(config=str(config_path), outdir=str(outdir))
    if command == "suite":
        return _run_suite_cmd(args)

    regime_name = command.replace("-", "_")
    return _run_single_regime_cmd(args, regime_name)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for larrak-validate-sim."""
    parser = argparse.ArgumentParser(
        prog="larrak-validate-sim",
        description="Canonical-regime simulation validation suite.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Validation regime")

    # Per-regime subcommands
    for regime_name, help_text in [
        ("chemistry", "Run chemistry-only regime validation"),
        ("spray", "Run spray/evaporation/premixing regime validation"),
        ("reacting-flow", "Run turbulent reacting-flow regime validation"),
        ("closed-cylinder", "Run closed-cylinder combustion/expansion validation"),
        ("full-handoff", "Run full engine phase-handoff validation"),
    ]:
        sub = subparsers.add_parser(regime_name, help=help_text)
        sub.add_argument("--config", required=True, help="Path to regime config JSON")
        sub.add_argument(
            "--outdir",
            default=f"outputs/validation/{regime_name.replace('-', '_')}",
            help="Output directory for artifacts",
        )

    # Suite subcommand
    suite_sub = subparsers.add_parser("suite", help="Run full validation suite")
    suite_sub.add_argument("--config", required=True, help="Path to suite config JSON")
    suite_sub.add_argument(
        "--outdir",
        default="outputs/validation/suite",
        help="Output directory for suite artifacts",
    )

    cache_sub = subparsers.add_parser(
        "chemistry-cache",
        help="Build or refresh the persistent offline chemistry cache",
    )
    cache_sub.add_argument(
        "--config",
        required=True,
        help="Path to a chemistry regime config or suite config containing chemistry",
    )
    cache_sub.add_argument(
        "--outdir",
        default="outputs/validation/chemistry_cache",
        help="Output directory for cache-build artifacts",
    )
    cache_sub.add_argument(
        "--output",
        default="",
        help="Optional override for the offline chemistry-results JSON path",
    )
    cache_sub.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore any existing offline cache and recompute it",
    )

    flame_compare_sub = subparsers.add_parser(
        "flame-speed-compare",
        help="Compare flame-speed tractability across mechanism candidates",
    )
    flame_compare_sub.add_argument(
        "--config",
        default="data/simulation_validation/flame_speed_mechanism_candidates.json",
        help="Path to the flame-speed mechanism comparison config JSON",
    )
    flame_compare_sub.add_argument(
        "--outdir",
        default="outputs/diagnostics/flame_speed_mechanism_compare",
        help="Output directory for comparison artifacts",
    )
    flame_compare_sub.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore any precomputed diagnostic artifacts and rerun all candidates live",
    )

    table_sub = subparsers.add_parser(
        "runtime-chemistry-table",
        help="Build or refresh an offline runtime chemistry table from a JSON config",
    )
    table_sub.add_argument("--config", required=True, help="Path to table config JSON")
    table_sub.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore any existing generated table files and rebuild them",
    )

    benchmark_sub = subparsers.add_parser(
        "engine-restart-benchmark",
        help="Replay the staged engine restart window under runtime chemistry variants",
    )
    benchmark_sub.add_argument("--run-dir", required=True, help="Base staged engine run directory")
    benchmark_sub.add_argument(
        "--tuned-params",
        required=True,
        help="Path to tuned-params JSON used for the original staged run",
    )
    benchmark_sub.add_argument(
        "--handoff-artifact",
        required=True,
        help="Path to the handoff artifact JSON that seeds the staged run",
    )
    benchmark_sub.add_argument(
        "--outdir",
        required=True,
        help="Output directory for the benchmark replay artifacts",
    )
    benchmark_sub.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="One or more benchmark profiles or runtime modes to replay",
    )
    benchmark_sub.add_argument(
        "--window-angle-deg",
        type=float,
        default=0.01,
        help="Crank-angle replay window from the checkpoint start",
    )
    benchmark_sub.add_argument(
        "--solver-name",
        default="larrakEngineFoam",
        help="OpenFOAM solver binary to run",
    )
    benchmark_sub.add_argument(
        "--docker-timeout-s",
        type=int,
        default=1800,
        help="Solver timeout for each replay stage",
    )
    benchmark_sub.add_argument(
        "--runtime-strategy-config",
        default="data/simulation_validation/engine_runtime_mechanism_strategy.json",
        help="Runtime chemistry strategy JSON; defaults to the canonical multitable ladder",
    )
    benchmark_sub.add_argument(
        "--package-label",
        default="",
        help="Optional checkpoint package label override from the runtime strategy",
    )
    benchmark_sub.add_argument(
        "--docker-image",
        default=None,
        help="Optional Docker image override for OpenFOAM execution",
    )
    benchmark_sub.add_argument(
        "--docker-bin",
        default=None,
        help="Optional Docker CLI override; otherwise use LARRAK_DOCKER_BIN, PATH, or known macOS Docker Desktop locations",
    )
    benchmark_sub.add_argument(
        "--refresh-runtime-tables",
        action="store_true",
        help="Refresh runtime tables referenced by the strategy before replaying",
    )
    benchmark_sub.add_argument(
        "--continue-across-stages",
        action="store_true",
        help="Continue across the remaining staged ignition bins instead of replaying only the first remaining stage",
    )
    benchmark_sub.add_argument(
        "--refresh-custom-solver",
        action="store_true",
        help="Force Docker rebuild of larrakEngineFoam even when the cached binary matches source_hash",
    )

    coverage_sub = subparsers.add_parser(
        "coverage-corpus-analysis",
        help="Measure transformed-space distances from miss targets to coverage corpus rows",
    )
    coverage_sub.add_argument(
        "--config",
        required=True,
        help="Path to runtime chemistry table JSON (ignition_entry config)",
    )
    coverage_sub.add_argument(
        "--extra-corpus",
        action="append",
        default=[],
        help="Additional runtimeChemistryCoverageCorpus.json path (repeatable); merged with config coverage_corpora",
    )
    coverage_sub.add_argument(
        "--authority-miss",
        default="",
        help="Optional runtimeChemistryAuthorityMiss.json to attach a single-row summary",
    )

    regression_sub = subparsers.add_parser(
        "restart-regression-analysis",
        help="Analyze ordered restart benchmark outputs for regression/improvement signals",
    )
    regression_sub.add_argument(
        "--runs",
        action="append",
        default=[],
        help="Explicit restart benchmark run directory to analyze; preserve the provided order",
    )
    regression_sub.add_argument(
        "--glob",
        default="",
        help="Glob pattern for restart benchmark run directories when --runs is not used",
    )
    regression_sub.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Optional cap on the latest N runs after glob expansion",
    )
    regression_sub.add_argument(
        "--suite",
        choices=["general", "scalars", "dense", "all"],
        default="all",
        help="Which analysis outputs to emit",
    )
    regression_sub.add_argument(
        "--profile-name",
        default="",
        help="Optional profile name when a run summary contains multiple profiles",
    )
    regression_sub.add_argument(
        "--history-window",
        type=int,
        default=5,
        help="Rolling history window for slope and stability calculations",
    )
    regression_sub.add_argument(
        "--outdir",
        default="outputs/diagnostics/restart_regression_analysis",
        help="Output directory for regression-analysis artifacts",
    )

    tuning_sub = subparsers.add_parser(
        "tuning-characterization",
        help="Ingest benchmark runs into experiment JSONL, propose knob vectors (GP-EI/NSGA2), or run a batch study",
    )
    tuning_sub.add_argument(
        "--mode",
        choices=["ingest", "propose", "run-batch"],
        dest="tc_mode",
        required=True,
    )
    tuning_sub.add_argument(
        "--runs",
        action="append",
        default=[],
        help="Restart benchmark outdir(s) containing engine_restart_benchmark_summary.json (ingest mode)",
    )
    tuning_sub.add_argument(
        "--glob",
        default="",
        help="Glob for benchmark outdirs when --runs is not used (ingest mode)",
    )
    tuning_sub.add_argument(
        "--latest",
        type=int,
        default=None,
        help="Keep only the latest N dirs after glob sort (ingest mode)",
    )
    tuning_sub.add_argument(
        "--experiments-jsonl",
        default="outputs/diagnostics/tuning_characterization/experiments.jsonl",
        help="Append-only experiment log path",
    )
    tuning_sub.add_argument(
        "--profile-name",
        default="",
        help="Profile name when summaries contain multiple profiles",
    )
    tuning_sub.add_argument(
        "--knob-schema",
        default="data/simulation_validation/tuning_knob_schema_chem323_ignition_entry_v1.json",
        help="Declarative knob schema JSON (propose and run-batch)",
    )
    tuning_sub.add_argument(
        "--n-proposals",
        type=int,
        default=4,
        dest="n_proposals",
        help="Number of knob vectors to propose (propose mode)",
    )
    tuning_sub.add_argument(
        "--strategy",
        default="gp_ei",
        choices=["random", "gp_ei", "nsga2_surrogate"],
        help="Search strategy for propose/run-batch",
    )
    tuning_sub.add_argument("--rng-seed", type=int, default=0, dest="rng_seed")
    tuning_sub.add_argument(
        "--base-table-config",
        default="data/simulation_validation/openfoam_runtime_chemistry_table_chem323_ignition_entry.json",
        help="Base runtime table JSON to patch (run-batch)",
    )
    tuning_sub.add_argument(
        "--strategy-config",
        default="",
        help="Optional strategy JSON path for manifest hashes (run-batch)",
    )
    tuning_sub.add_argument(
        "--study-outdir",
        default="outputs/diagnostics/tuning_characterization/study_runs",
        dest="study_outdir",
        help="Per-trial staging directory parent (run-batch)",
    )
    tuning_sub.add_argument("--n-trials", type=int, default=1, dest="n_trials")
    tuning_sub.add_argument("--refresh-table", action="store_true", dest="refresh_table")
    tuning_sub.add_argument("--run-benchmark", action="store_true", dest="run_benchmark")
    tuning_sub.add_argument("--benchmark-run-dir", default="", dest="benchmark_run_dir")
    tuning_sub.add_argument("--tuned-params", default="", dest="tuned_params")
    tuning_sub.add_argument("--handoff-artifact", default="", dest="handoff_artifact")
    tuning_sub.add_argument(
        "--runtime-strategy-config",
        default="",
        dest="runtime_strategy_config",
        help="Multitable strategy JSON for engine-restart-benchmark",
    )
    tuning_sub.add_argument(
        "--benchmark-profiles",
        nargs="*",
        default=[],
        dest="benchmark_profiles",
        help="Profiles for engine-restart-benchmark (default chem323_lookup_strict if empty)",
    )
    tuning_sub.add_argument("--window-angle-deg", type=float, default=0.01, dest="window_angle_deg")
    tuning_sub.add_argument("--docker-timeout-s", type=int, default=3600, dest="docker_timeout_s")
    tuning_sub.add_argument(
        "--refresh-custom-solver", action="store_true", dest="refresh_custom_solver"
    )
    tuning_sub.add_argument("--max-benchmarks", type=int, default=8, dest="max_benchmarks")
    tuning_sub.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Stage configs and manifests only; do not refresh table or run benchmark",
    )

    truth_sub = subparsers.add_parser(
        "combustion-truth",
        help="Run the gas-combustion truth workflow over the DOE core corridor",
    )
    truth_sub.add_argument(
        "--config",
        required=True,
        help="Path to the gas-combustion suite config JSON",
    )
    truth_sub.add_argument(
        "--profile",
        default="data/training/f2_nn_overnight_core_edge_v1.json",
        help="DOE/training profile defining the core corridor",
    )
    truth_sub.add_argument(
        "--outdir",
        default="outputs/combustion_truth/gas_combustion_v1",
        help="Output directory for truth records and per-point suite artifacts",
    )
    truth_sub.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap on the number of core operating points to run",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "suite":
        return _run_suite_cmd(args)
    if args.command == "chemistry-cache":
        return _run_chemistry_cache_cmd(args)
    if args.command == "flame-speed-compare":
        return _run_flame_speed_compare_cmd(args)
    if args.command == "runtime-chemistry-table":
        return _run_runtime_chemistry_table_cmd(args)
    if args.command == "engine-restart-benchmark":
        return _run_engine_restart_benchmark_cmd(args)
    if args.command == "coverage-corpus-analysis":
        return _run_coverage_corpus_analysis_cmd(args)
    if args.command == "restart-regression-analysis":
        return _run_restart_regression_analysis_cmd(args)
    if args.command == "tuning-characterization":
        return _run_tuning_characterization_cmd(args)
    if args.command == "combustion-truth":
        return _run_combustion_truth_cmd(args)

    # Map CLI names to internal regime names
    regime_map = {
        "chemistry": "chemistry",
        "spray": "spray",
        "reacting-flow": "reacting_flow",
        "closed-cylinder": "closed_cylinder",
        "full-handoff": "full_handoff",
    }
    regime_name = regime_map.get(args.command)
    if regime_name is None:
        logger.error("Unknown command: %s", args.command)
        return 1

    return _run_single_regime_cmd(args, regime_name)


if __name__ == "__main__":
    sys.exit(main())
