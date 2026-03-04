"""Tests for archive metadata and version stamping."""

import json
import tempfile
from pathlib import Path

from larrak2.cli.run_pareto import main as pareto_main
from larrak2.core.constraints import get_constraint_names
from larrak2.core.encoding import ENCODING_VERSION


def test_summary_contains_versions():
    """Run a tiny Pareto job and verify metadata is stamped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exit_code = pareto_main(
            [
                "--pop",
                "4",
                "--gen",
                "1",
                "--rpm",
                "2000",
                "--torque",
                "100",
                "--fidelity",
                "0",
                "--allow-nonproduction-paths",
                "--algorithm",
                "nsga2",
                "--seed",
                "11",
                "--outdir",
                tmpdir,
            ]
        )
        assert exit_code == 0

        summary_path = Path(tmpdir) / "summary.json"
        assert summary_path.exists(), "summary.json missing"

        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["encoding_version"] == ENCODING_VERSION
        assert "model_versions" in summary and "thermo_v1" in summary["model_versions"]
        assert summary["n_obj"] >= 3
        assert summary["n_obj"] == len(summary.get("objective_names", []))
        expected_constraints = get_constraint_names(int(summary["fidelity"]))
        assert summary["n_constr"] == len(expected_constraints)
        assert summary["constraint_names"] == expected_constraints
        assert "constraint_scales" in summary


def test_nsga3_ref_dir_cap_and_effective_metadata_are_reported():
    """NSGA-III should respect ref-dir cap and emit effective runtime metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exit_code = pareto_main(
            [
                "--pop",
                "8",
                "--gen",
                "1",
                "--algorithm",
                "nsga3",
                "--partitions",
                "4",
                "--nsga3-max-ref-dirs",
                "40",
                "--rpm",
                "2200",
                "--torque",
                "110",
                "--fidelity",
                "0",
                "--allow-nonproduction-paths",
                "--seed",
                "17",
                "--outdir",
                tmpdir,
            ]
        )
        assert exit_code == 0

        summary_path = Path(tmpdir) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        assert summary["algorithm"] == "nsga3"
        assert int(summary["requested_pop"]) == 8
        assert int(summary["requested_partitions"]) == 4
        assert int(summary["n_ref_dirs"]) <= 40
        assert int(summary["effective_partitions"]) <= 4
        assert int(summary["effective_pop"]) >= int(summary["requested_pop"])
        assert int(summary["effective_pop"]) >= int(summary["n_ref_dirs"])

        gate = summary.get("production_gate", {})
        for key in (
            "production_profile",
            "production_gate_pass",
            "production_gate_failures",
            "fallback_paths_used",
            "nonproduction_overrides",
            "n_eval_errors",
            "algorithm_used",
            "fidelity",
            "constraint_phase",
        ):
            assert key in gate
