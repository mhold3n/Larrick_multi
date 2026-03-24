"""Compare flame-speed tractability across multiple mechanism candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .flame_speed_diagnostics import run_flame_speed_diagnostics


@dataclass
class FlameSpeedMechanismCandidate:
    """Mechanism candidate for flame-speed comparison."""

    candidate_id: str
    description: str
    mechanism_file: str
    mechanism_format: str = ""
    thermo_file: str = ""
    transport_file: str = ""
    generated_yaml_path: str = ""
    sanitizer_profile: str = ""
    fuel: str = "IC8H18"
    oxidizer: dict[str, float] = field(default_factory=lambda: {"O2": 0.2033, "N2": 0.7859})
    fuel_matched: bool = True
    benchmark_only: bool = False
    diagnostic_artifact_path: str = ""
    notes: str = ""


@dataclass
class FlameSpeedComparisonConfig:
    """Flame-speed comparison configuration."""

    comparison_id: str
    description: str
    case_set: str
    reference_candidate_id: str
    temperature_K: float
    pressure_bar: float
    equivalence_ratio: float
    candidates: list[FlameSpeedMechanismCandidate]


def load_flame_speed_comparison_config(path: str | Path) -> FlameSpeedComparisonConfig:
    """Load comparison config from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    shared = dict(payload.get("shared_conditions", {}) or {})
    oxidizer = dict(shared.get("oxidizer", {"O2": 0.2033, "N2": 0.7859}) or {})
    candidates = [
        FlameSpeedMechanismCandidate(
            candidate_id=str(item["candidate_id"]),
            description=str(item.get("description", "")),
            mechanism_file=str(item["mechanism_file"]),
            mechanism_format=str(item.get("mechanism_format", "")),
            thermo_file=str(item.get("thermo_file", "")),
            transport_file=str(item.get("transport_file", "")),
            generated_yaml_path=str(item.get("generated_yaml_path", "")),
            sanitizer_profile=str(item.get("sanitizer_profile", "")),
            fuel=str(item.get("fuel", "IC8H18")),
            oxidizer=dict(item.get("oxidizer", oxidizer) or oxidizer),
            fuel_matched=bool(item.get("fuel_matched", True)),
            benchmark_only=bool(item.get("benchmark_only", False)),
            diagnostic_artifact_path=str(item.get("diagnostic_artifact_path", "")),
            notes=str(item.get("notes", "")),
        )
        for item in payload.get("candidates", [])
    ]
    if not candidates:
        raise ValueError("Flame-speed comparison config must define at least one candidate")
    return FlameSpeedComparisonConfig(
        comparison_id=str(payload.get("comparison_id", "flame_speed_comparison_v1")),
        description=str(payload.get("description", "")),
        case_set=str(payload.get("case_set", "quick")),
        reference_candidate_id=str(
            payload.get("reference_candidate_id", candidates[0].candidate_id)
        ),
        temperature_K=float(shared.get("temperature_K", 353.0)),
        pressure_bar=float(shared.get("pressure_bar", 3.33)),
        equivalence_ratio=float(shared.get("equivalence_ratio", 1.0)),
        candidates=candidates,
    )


def _load_diagnostic_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected diagnostic JSON object at '{path}'")
    return payload


def _fastest_load_time_s(summary: dict[str, Any]) -> float | None:
    values = [
        float(item.get("load_time_s", 0.0))
        for item in summary.get("results", [])
        if float(item.get("load_time_s", 0.0)) > 0.0
    ]
    return min(values) if values else None


def _best_flame_result(summary: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        item
        for item in summary.get("results", [])
        if item.get("mode") == "free_flame"
        and bool(item.get("success"))
        and item.get("flame_speed_m_s") is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item.get("total_time_s", float("inf"))))


def _candidate_summary(
    *,
    candidate: FlameSpeedMechanismCandidate,
    diagnostic_summary: dict[str, Any],
    diagnostic_json_path: str,
    diagnostic_markdown_path: str,
    diagnostic_source: str,
    reference_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    load_time_s = _fastest_load_time_s(diagnostic_summary)
    best_flame = _best_flame_result(diagnostic_summary)
    flame_speed = None if best_flame is None else float(best_flame["flame_speed_m_s"])
    flame_total_s = None if best_flame is None else float(best_flame["total_time_s"])
    n_species = max(
        (int(item.get("n_species", 0)) for item in diagnostic_summary.get("results", [])), default=0
    )
    n_reactions = max(
        (int(item.get("n_reactions", 0)) for item in diagnostic_summary.get("results", [])),
        default=0,
    )

    comparison_to_reference = {
        "feasible": False,
        "load_time_ratio": None,
        "flame_speed_delta_m_s": None,
        "flame_speed_relative_error": None,
        "note": "",
    }
    if reference_snapshot is None:
        comparison_to_reference["note"] = "This is the reference candidate."
    else:
        ref_load_time = reference_snapshot.get("fastest_load_time_s")
        ref_flame_speed = reference_snapshot.get("flame_speed_m_s")
        if load_time_s is not None and ref_load_time not in (None, 0):
            comparison_to_reference["load_time_ratio"] = load_time_s / float(ref_load_time)

        if not candidate.fuel_matched:
            comparison_to_reference["note"] = (
                "Fuel is not matched to the reference mechanism, so only tractability timing "
                "comparisons are meaningful."
            )
        elif flame_speed is None or ref_flame_speed is None:
            comparison_to_reference["note"] = (
                "Direct flame-speed comparison is not feasible because at least one candidate "
                "did not complete a flame-speed solve."
            )
        else:
            delta = flame_speed - float(ref_flame_speed)
            denom = max(abs(float(ref_flame_speed)), 1.0e-12)
            comparison_to_reference.update(
                feasible=True,
                flame_speed_delta_m_s=delta,
                flame_speed_relative_error=delta / denom,
                note="Direct flame-speed comparison against the reference candidate is available.",
            )

    return {
        "candidate_id": candidate.candidate_id,
        "description": candidate.description,
        "mechanism_file": candidate.mechanism_file,
        "fuel": candidate.fuel,
        "fuel_matched": candidate.fuel_matched,
        "benchmark_only": candidate.benchmark_only,
        "notes": candidate.notes,
        "diagnostic_source": diagnostic_source,
        "diagnostic_json_path": diagnostic_json_path,
        "diagnostic_markdown_path": diagnostic_markdown_path,
        "diagnosis_classification": diagnostic_summary.get("diagnosis_classification", ""),
        "diagnosis_summary": diagnostic_summary.get("diagnosis_summary", ""),
        "n_species": n_species,
        "n_reactions": n_reactions,
        "fastest_load_time_s": load_time_s,
        "flame_speed_m_s": flame_speed,
        "flame_total_time_s": flame_total_s,
        "comparison_to_reference": comparison_to_reference,
    }


def write_flame_speed_comparison_artifacts(
    *,
    outdir: Path,
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    """Persist JSON and Markdown comparison artifacts."""
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "flame_speed_mechanism_comparison.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Flame-Speed Mechanism Comparison",
        "",
        f"**Comparison ID:** {summary['comparison_id']}",
        f"**Reference Candidate:** {summary['reference_candidate_id']}",
        "",
        "| Candidate | Classification | Fuel Matched | Fastest Load (s) | Flame Speed (m/s) | Comparison Note |",
        "|-----------|----------------|--------------|------------------|-------------------|-----------------|",
    ]
    for item in summary.get("candidates", []):
        load_value = (
            "" if item.get("fastest_load_time_s") is None else f"{item['fastest_load_time_s']:.3f}"
        )
        flame_value = (
            "" if item.get("flame_speed_m_s") is None else f"{item['flame_speed_m_s']:.6g}"
        )
        lines.append(
            f"| {item['candidate_id']} | {item['diagnosis_classification']} | "
            f"{item['fuel_matched']} | {load_value} | {flame_value} | "
            f"{item['comparison_to_reference']['note']} |"
        )
    md_path = outdir / "flame_speed_mechanism_comparison.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def compare_flame_speed_mechanisms(
    *,
    config_path: str | Path,
    outdir: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Run or load diagnostics for each candidate and compare them."""
    config = load_flame_speed_comparison_config(config_path)
    output_dir = Path(outdir)

    summaries_by_candidate: dict[str, dict[str, Any]] = {}
    artifact_paths_by_candidate: dict[str, tuple[str, str]] = {}
    source_by_candidate: dict[str, str] = {}
    for candidate in config.candidates:
        artifact_path = (
            Path(candidate.diagnostic_artifact_path) if candidate.diagnostic_artifact_path else None
        )
        if artifact_path is not None and artifact_path.exists() and not refresh:
            diagnostic_summary = _load_diagnostic_summary(artifact_path)
            markdown_path = artifact_path.with_suffix(".md")
            summaries_by_candidate[candidate.candidate_id] = diagnostic_summary
            artifact_paths_by_candidate[candidate.candidate_id] = (
                str(artifact_path),
                str(markdown_path),
            )
            source_by_candidate[candidate.candidate_id] = "precomputed"
            continue

        live_summary = run_flame_speed_diagnostics(
            mechanism_file=candidate.mechanism_file,
            mechanism_format=candidate.mechanism_format,
            thermo_file=candidate.thermo_file,
            transport_file=candidate.transport_file,
            generated_yaml_path=(
                candidate.generated_yaml_path
                or f"outputs/validation_runtime/mechanisms/{candidate.candidate_id}.yaml"
            ),
            sanitizer_profile=candidate.sanitizer_profile,
            case_set=config.case_set,
            temperature_K=config.temperature_K,
            pressure_bar=config.pressure_bar,
            equivalence_ratio=config.equivalence_ratio,
            fuel=candidate.fuel,
            oxidizer=candidate.oxidizer,
            outdir=output_dir / candidate.candidate_id,
        )
        diagnostic_json_path = str(live_summary["json_path"])
        diagnostic_markdown_path = str(live_summary["markdown_path"])
        live_payload = _load_diagnostic_summary(diagnostic_json_path)
        summaries_by_candidate[candidate.candidate_id] = live_payload
        artifact_paths_by_candidate[candidate.candidate_id] = (
            diagnostic_json_path,
            diagnostic_markdown_path,
        )
        source_by_candidate[candidate.candidate_id] = "live"

    if config.reference_candidate_id not in summaries_by_candidate:
        raise ValueError(
            f"Reference candidate '{config.reference_candidate_id}' is not defined in '{config_path}'"
        )

    reference_summary = _candidate_summary(
        candidate=next(
            item for item in config.candidates if item.candidate_id == config.reference_candidate_id
        ),
        diagnostic_summary=summaries_by_candidate[config.reference_candidate_id],
        diagnostic_json_path=artifact_paths_by_candidate[config.reference_candidate_id][0],
        diagnostic_markdown_path=artifact_paths_by_candidate[config.reference_candidate_id][1],
        diagnostic_source=source_by_candidate[config.reference_candidate_id],
        reference_snapshot=None,
    )

    candidate_summaries = [reference_summary]
    for candidate in config.candidates:
        if candidate.candidate_id == config.reference_candidate_id:
            continue
        candidate_summaries.append(
            _candidate_summary(
                candidate=candidate,
                diagnostic_summary=summaries_by_candidate[candidate.candidate_id],
                diagnostic_json_path=artifact_paths_by_candidate[candidate.candidate_id][0],
                diagnostic_markdown_path=artifact_paths_by_candidate[candidate.candidate_id][1],
                diagnostic_source=source_by_candidate[candidate.candidate_id],
                reference_snapshot=reference_summary,
            )
        )

    summary = {
        "comparison_id": config.comparison_id,
        "description": config.description,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "reference_candidate_id": config.reference_candidate_id,
        "case_set": config.case_set,
        "shared_conditions": {
            "temperature_K": config.temperature_K,
            "pressure_bar": config.pressure_bar,
            "equivalence_ratio": config.equivalence_ratio,
        },
        "candidates": candidate_summaries,
    }
    json_path, md_path = write_flame_speed_comparison_artifacts(
        outdir=output_dir,
        summary=summary,
    )
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(md_path)
    return summary
