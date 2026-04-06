# Multitable refresh and validation order (v66)

Source of truth: `engine_runtime_mechanism_strategy_multitable_v1.json` → `runtime_package.stage_runtime_tables`.

Refresh and validate **in this order** until the entry-stage replacement gate passes:

1. **`chem323_engine_ignition_entry_v1`** — `openfoam_runtime_chemistry_table_chem323_ignition_entry.json`
2. **`ignition_ramp`** — `openfoam_runtime_chemistry_table_chem323_ignition_ramp.json`
3. **`ignition_branch`** — `openfoam_runtime_chemistry_table_chem323_ignition_branch.json`
4. **`ignition_hot_core`** — `openfoam_runtime_chemistry_table_chem323_ignition_hot_core.json`
5. **`ignition_tail`** — `openfoam_runtime_chemistry_table_chem323_ignition_tail.json`

**Benchmark scope:** The default v66 recipe (`v66_engine_restart_recipe.json`) and `engine-restart-benchmark` replay **only the first remaining ignition stage** (entry-first). Add `--continue-across-stages` only **after** the entry authority surface passes its gate, so ramp/branch tables are not tuned while entry is still broken.
