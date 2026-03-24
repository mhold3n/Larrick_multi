"""End-to-end coverage for the gas-combustion validation/truth path."""

from __future__ import annotations

import json
from pathlib import Path

from larrak2.cli.validate_simulation import run_validation_preflight
from larrak2.simulation_validation.combustion_truth import run_combustion_truth_workflow

SUITE_CONFIG = "data/simulation_validation/gas_combustion_suite_config.json"
PROFILE_CONFIG = "data/training/f2_nn_overnight_core_edge_v1.json"


def _force_missing_cantera(monkeypatch) -> None:
    monkeypatch.setattr(
        "larrak2.simulation_validation.solver_adapters.importlib.util.find_spec",
        lambda name: None if name == "cantera" else object(),
    )


def _suite_config_with_offline_cache(tmp_path: Path) -> Path:
    suite = json.loads(Path(SUITE_CONFIG).read_text(encoding="utf-8"))
    fixture_payload = json.loads(
        Path("data/simulation_validation/gas_combustion_chemistry_fixture_results.json").read_text(
            encoding="utf-8"
        )
    )
    cache_path = tmp_path / "gas_combustion_llnl_detailed.json"
    cache_path.write_text(json.dumps(fixture_payload, indent=2), encoding="utf-8")

    adapter = suite["regimes"]["chemistry"]["case_spec"]["solver_config"]["simulation_adapter"]
    adapter["offline_results_path"] = str(cache_path)

    suite_path = tmp_path / "gas_combustion_suite_config.json"
    suite_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    return suite_path


def test_gas_combustion_suite_passes_with_five_regimes(monkeypatch, tmp_path: Path) -> None:
    _force_missing_cantera(monkeypatch)
    suite_path = _suite_config_with_offline_cache(tmp_path)

    outdir = tmp_path / "gas_suite"
    code = run_validation_preflight("suite", config_path=suite_path, outdir=outdir)
    assert code == 0

    manifest = json.loads((outdir / "suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite_id"] == "gas_combustion_v1"
    assert manifest["overall_passed"] is True
    assert [entry["regime"] for entry in manifest["scoreboard"]] == [
        "chemistry",
        "spray",
        "reacting_flow",
        "closed_cylinder",
        "full_handoff",
    ]
    assert all(entry["status"] == "passed" for entry in manifest["scoreboard"])


def test_combustion_truth_workflow_writes_core_records(monkeypatch, tmp_path: Path) -> None:
    _force_missing_cantera(monkeypatch)
    suite_path = _suite_config_with_offline_cache(tmp_path)

    outdir = tmp_path / "combustion_truth"
    summary = run_combustion_truth_workflow(
        suite_config_path=suite_path,
        profile_path=PROFILE_CONFIG,
        outdir=outdir,
        max_points=2,
    )

    assert summary["n_points"] == 2
    assert summary["n_passed"] == 2

    records = [
        json.loads(line)
        for line in (outdir / "combustion_truth_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert all(record["truth_valid"] is True for record in records)
    assert all("closed_cylinder_outputs" in record for record in records)
    assert all(record["regime_statuses"]["closed_cylinder"] == "passed" for record in records)
