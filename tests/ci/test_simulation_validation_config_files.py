"""Fixture tests for JSON validation configs shipped in data/simulation_validation."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_DIR = Path("data/simulation_validation")
REGIME_CONFIGS = (
    "chemistry_config.json",
    "chemistry_flame_speed_reduced_config.json",
    "chemistry_flame_speed_reduced_chem679_config.json",
    "chemistry_smoke_config.json",
    "spray_config.json",
    "reacting_flow_config.json",
    "closed_cylinder_config.json",
    "full_handoff_config.json",
)
MULTITABLE_STAGE_CONFIGS = (
    "openfoam_runtime_chemistry_table_chem323_ignition_entry.json",
    "openfoam_runtime_chemistry_table_chem323_ignition_ramp.json",
    "openfoam_runtime_chemistry_table_chem323_ignition_branch.json",
    "openfoam_runtime_chemistry_table_chem323_ignition_hot_core.json",
    "openfoam_runtime_chemistry_table_chem323_ignition_tail.json",
)


def _load_config(filename: str) -> dict:
    return json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))


class TestSimulationValidationConfigFiles:
    def test_all_metric_specs_define_comparison_mode(self):
        for filename in REGIME_CONFIGS:
            config = _load_config(filename)
            metrics = config["dataset"].get("metrics", [])
            assert metrics, f"{filename} should define at least one metric"
            for metric in metrics:
                assert "comparison_mode" in metric, (
                    f"{filename} metric '{metric.get('metric_id', '<missing>')}' "
                    "must define comparison_mode explicitly"
                )

    def test_closed_cylinder_measured_config_stays_explicit(self):
        config = _load_config("closed_cylinder_config.json")
        dataset = config["dataset"]

        assert dataset["source_type"] == "measured"
        assert "doi" in dataset["provenance"]
        assert all("comparison_mode" in metric for metric in dataset["metrics"])

    def test_chemistry_smoke_config_has_all_metric_values(self):
        config = _load_config("chemistry_smoke_config.json")
        simulation_data = config["simulation_data"]

        for metric in config["dataset"]["metrics"]:
            metric_id = metric["metric_id"]
            assert metric_id in simulation_data
            assert f"{metric_id}_measured" in simulation_data

    def test_gas_combustion_suite_config_declares_five_regimes(self):
        config = _load_config("gas_combustion_suite_config.json")
        assert config["suite_id"] == "gas_combustion_v1"
        assert config["regime_order"] == [
            "chemistry",
            "spray",
            "reacting_flow",
            "closed_cylinder",
            "full_handoff",
        ]
        assert config["prerequisites"]["full_handoff"] == [
            "spray",
            "reacting_flow",
            "closed_cylinder",
        ]

    def test_gas_combustion_suite_metrics_define_comparison_mode(self):
        config = _load_config("gas_combustion_suite_config.json")
        for regime_name, regime_cfg in config["regimes"].items():
            metrics = regime_cfg["dataset"].get("metrics", [])
            assert metrics, f"{regime_name} should define at least one metric"
            for metric in metrics:
                assert "comparison_mode" in metric, (
                    f"{regime_name} metric '{metric.get('metric_id', '<missing>')}' "
                    "must define comparison_mode explicitly"
                )

    def test_gas_combustion_suite_chemistry_declares_offline_cache(self):
        config = _load_config("gas_combustion_suite_config.json")
        adapter = config["regimes"]["chemistry"]["case_spec"]["solver_config"]["simulation_adapter"]

        assert adapter["offline_results_path"] == (
            "outputs/validation_runtime/chemistry_cache/gas_combustion_llnl_detailed.json"
        )
        assert adapter["offline_results_only"] is True
        assert "fixture_results_path" not in adapter

    def test_chemistry_regime_config_declares_offline_cache_adapter(self):
        config = _load_config("chemistry_config.json")
        adapter = config["case_spec"]["solver_config"]["simulation_adapter"]

        assert adapter["backend"] == "auto"
        assert adapter["offline_results_path"] == (
            "outputs/validation_runtime/chemistry_cache/gas_combustion_llnl_detailed.json"
        )
        assert "fixture_results_path" not in adapter

    def test_reduced_flame_speed_config_targets_chem323(self):
        config = _load_config("chemistry_flame_speed_reduced_config.json")
        adapter = config["case_spec"]["solver_config"]["simulation_adapter"]
        metric_cfg = adapter["metrics"]["laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane"]

        assert adapter["backend"] == "native_cantera"
        assert adapter["mechanism_file"].endswith("Chem323.inp.txt")
        assert metric_cfg["mechanism_file"].endswith("Chem323.inp.txt")
        assert metric_cfg["grid_points"] == 5
        assert metric_cfg["refine_grid"] is False
        assert metric_cfg["staged_energy"] == [False, True]

    def test_reduced_flame_speed_config_targets_chem679(self):
        config = _load_config("chemistry_flame_speed_reduced_chem679_config.json")
        adapter = config["case_spec"]["solver_config"]["simulation_adapter"]
        metric_cfg = adapter["metrics"]["laminar_flame_speed_phi1_Tu353K_P3p33bar_iso_octane"]

        assert adapter["backend"] == "native_cantera"
        assert adapter["mechanism_file"].endswith("Chem679.inp.txt")
        assert metric_cfg["mechanism_file"].endswith("Chem679.inp.txt")
        assert metric_cfg["grid_points"] == 5
        assert metric_cfg["refine_grid"] is False
        assert metric_cfg["staged_energy"] == [False, True]

    def test_reduced_mechanism_comparison_config_declares_benchmark_case_set(self):
        config = _load_config("flame_speed_reduced_mechanism_candidates.json")

        assert config["case_set"] == "benchmark"
        assert config["reference_candidate_id"] == "chem679_reduced_gasoline_surrogate"
        assert [item["candidate_id"] for item in config["candidates"]] == [
            "chem679_reduced_gasoline_surrogate",
            "chem323_reduced_gasoline_surrogate",
        ]

    def test_multitable_strategy_surface_is_restored_and_mirrored(self):
        canonical = _load_config("engine_runtime_mechanism_strategy.json")
        explicit = _load_config("engine_runtime_mechanism_strategy_multitable_v1.json")

        assert canonical["strategy_id"] == "engine_runtime_mechanism_multitable_v1"
        assert explicit["strategy_id"] == canonical["strategy_id"]
        assert explicit["runtime_package"] == canonical["runtime_package"]
        assert explicit["checkpoint_packages"] == canonical["checkpoint_packages"]
        assert set(canonical["runtime_package"]["stage_runtime_tables"]) == {
            "ignition_entry",
            "ignition_ramp",
            "ignition_branch",
            "ignition_hot_core",
            "ignition_tail",
        }

    def test_multitable_stage_local_configs_exist(self):
        for filename in MULTITABLE_STAGE_CONFIGS:
            payload = _load_config(filename)
            table_cfg = dict(payload.get("runtime_chemistry_table", {}) or {})
            assert table_cfg["package_dir"] == "mechanisms/openfoam/v2512/chem323_reduced"
            assert table_cfg["table_id"].startswith("chem323_engine_ignition_")

    def test_ignition_entry_config_uses_current_species_and_qdot_frontiers(self):
        payload = _load_config("openfoam_runtime_chemistry_table_chem323_ignition_entry.json")
        table_cfg = dict(payload["runtime_chemistry_table"])

        assert table_cfg["seed_species_miss_artifacts"] == [
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v66h_chem323_ignition_entry_seeded_h2o2_sanity/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json",
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v66i_chem323_ignition_entry_seeded_c5h82ooh45_diag_sanity/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json",
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v66k_chem323_ignition_entry_frontier_rebalanced_sanity/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json",
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v66p_chem323_ignition_entry_qdot_quad_sanity/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json",
        ]
        assert not any("v65" in str(path) for path in table_cfg["seed_species_miss_artifacts"])
        assert table_cfg["seed_qdot_miss_artifacts"] == [
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v66c_chem323_ignition_entry_seedable_qdot_capture/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json",
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v66j_chem323_ignition_entry_seeded_cy3c5h8o_diag_sanity/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json",
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v66l_chem323_ignition_entry_seeded_ch2oh_sanity/chem323_lookup_strict/runtimeChemistryAuthorityMiss.json",
        ]
        assert table_cfg["coverage_corpora"] == [
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v62_chem323_multitable_handoff_gate/chem323_lookup_strict/runtimeChemistryCoverageCorpus.json",
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v63_chem323_multitable_entry_refreshed/chem323_lookup_strict/runtimeChemistryCoverageCorpus.json",
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v64_chem323_multitable_entry_qdot_refreshed/chem323_lookup_strict/runtimeChemistryCoverageCorpus.json",
            "outputs/diagnostics/engine_restart_benchmark_live_parallel_v66s_chem323_ignition_entry_corpus_capture/chem323_lookup_strict/runtimeChemistryCoverageCorpus.json",
        ]
        assert table_cfg["current_window_qdot_target_limit"] == 3
        assert table_cfg["current_window_qdot_support_per_target"] == 12
        assert table_cfg["current_window_diag_target_limit"] == 4
        assert "state_axes" not in table_cfg
        assert not any("v65" in str(path) for path in table_cfg["coverage_corpora"])
        assert not any("v66a" in str(path) for path in table_cfg["coverage_corpora"])
        assert not any("v66c" in str(path) for path in table_cfg["coverage_corpora"])
