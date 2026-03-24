"""Coverage for flame-speed mechanism comparison tooling."""

from __future__ import annotations

import json
from pathlib import Path

from larrak2.cli.validate_simulation import main as validate_simulation_main
from larrak2.simulation_validation.flame_speed_comparison import (
    compare_flame_speed_mechanisms,
    load_flame_speed_comparison_config,
)


def test_load_flame_speed_comparison_config_reads_candidates(tmp_path: Path) -> None:
    config_path = tmp_path / "comparison.json"
    config_path.write_text(
        json.dumps(
            {
                "comparison_id": "flame_cmp",
                "reference_candidate_id": "detailed",
                "shared_conditions": {
                    "temperature_K": 353.0,
                    "pressure_bar": 3.33,
                    "equivalence_ratio": 1.0,
                    "oxidizer": {"O2": 0.21, "N2": 0.79},
                },
                "candidates": [
                    {
                        "candidate_id": "detailed",
                        "description": "Detailed mechanism",
                        "mechanism_file": "mechanisms/detailed.yaml",
                    },
                    {
                        "candidate_id": "reduced",
                        "description": "Reduced mechanism",
                        "mechanism_file": "mechanisms/reduced.yaml",
                        "fuel": "c12h26",
                        "fuel_matched": False,
                        "benchmark_only": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_flame_speed_comparison_config(config_path)

    assert loaded.reference_candidate_id == "detailed"
    assert loaded.candidates[1].candidate_id == "reduced"
    assert loaded.candidates[1].fuel == "c12h26"
    assert loaded.candidates[1].fuel_matched is False


def test_compare_flame_speed_mechanisms_uses_precomputed_reference_and_live_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reference_json = tmp_path / "reference.json"
    reference_json.write_text(
        json.dumps(
            {
                "diagnosis_classification": "reduced_mechanism_recommended",
                "diagnosis_summary": "Reference timed out on flame speed.",
                "results": [
                    {
                        "case_id": "load_transport_none",
                        "mode": "load_only",
                        "success": True,
                        "timed_out": False,
                        "load_time_s": 378.0,
                        "total_time_s": 378.0,
                        "n_species": 1387,
                        "n_reactions": 9599,
                    },
                    {
                        "case_id": "free_flame_staged_mixture_averaged",
                        "mode": "free_flame",
                        "success": False,
                        "timed_out": True,
                        "total_time_s": 900.0,
                        "flame_speed_m_s": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    reference_json.with_suffix(".md").write_text("# ref\n", encoding="utf-8")

    config_path = tmp_path / "comparison.json"
    config_path.write_text(
        json.dumps(
            {
                "comparison_id": "flame_cmp",
                "reference_candidate_id": "detailed",
                "case_set": "quick",
                "shared_conditions": {
                    "temperature_K": 353.0,
                    "pressure_bar": 3.33,
                    "equivalence_ratio": 1.0,
                },
                "candidates": [
                    {
                        "candidate_id": "detailed",
                        "description": "Detailed mechanism",
                        "mechanism_file": "mechanisms/detailed.yaml",
                        "diagnostic_artifact_path": str(reference_json),
                    },
                    {
                        "candidate_id": "reduced",
                        "description": "Reduced mechanism",
                        "mechanism_file": "mechanisms/reduced.yaml",
                        "fuel": "c12h26",
                        "fuel_matched": False,
                        "benchmark_only": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def _fake_run_flame_speed_diagnostics(**kwargs):
        outdir = Path(kwargs["outdir"])
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "diagnosis_classification": "tractable",
            "diagnosis_summary": "Reduced candidate completed.",
            "results": [
                {
                    "case_id": "load_transport_none",
                    "mode": "load_only",
                    "success": True,
                    "timed_out": False,
                    "load_time_s": 1.5,
                    "total_time_s": 1.5,
                    "n_species": 100,
                    "n_reactions": 400,
                },
                {
                    "case_id": "free_flame_staged_mixture_averaged",
                    "mode": "free_flame",
                    "success": True,
                    "timed_out": False,
                    "total_time_s": 12.0,
                    "flame_speed_m_s": 0.41,
                    "n_species": 100,
                    "n_reactions": 400,
                },
            ],
        }
        json_path = outdir / "llnl_flame_speed_diagnostic.json"
        md_path = outdir / "llnl_flame_speed_diagnostic.md"
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        md_path.write_text("# reduced\n", encoding="utf-8")
        return {
            **payload,
            "json_path": str(json_path),
            "markdown_path": str(md_path),
        }

    monkeypatch.setattr(
        "larrak2.simulation_validation.flame_speed_comparison.run_flame_speed_diagnostics",
        _fake_run_flame_speed_diagnostics,
    )

    summary = compare_flame_speed_mechanisms(
        config_path=config_path,
        outdir=tmp_path / "out",
    )

    assert summary["reference_candidate_id"] == "detailed"
    assert summary["candidates"][0]["diagnostic_source"] == "precomputed"
    assert summary["candidates"][1]["diagnostic_source"] == "live"
    assert summary["candidates"][1]["diagnosis_classification"] == "tractable"
    assert summary["candidates"][1]["comparison_to_reference"]["load_time_ratio"] == 1.5 / 378.0
    assert summary["candidates"][1]["comparison_to_reference"]["feasible"] is False
    assert "Fuel is not matched" in summary["candidates"][1]["comparison_to_reference"]["note"]


def test_flame_speed_compare_cli_runs(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "comparison.json"
    config_path.write_text(
        json.dumps(
            {"candidates": [{"candidate_id": "x", "description": "", "mechanism_file": "x.yaml"}]}
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "larrak2.simulation_validation.flame_speed_comparison.compare_flame_speed_mechanisms",
        lambda **kwargs: {"comparison_id": "cmp", "candidates": [], **kwargs},
    )

    code = validate_simulation_main(
        [
            "flame-speed-compare",
            "--config",
            str(config_path),
            "--outdir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
