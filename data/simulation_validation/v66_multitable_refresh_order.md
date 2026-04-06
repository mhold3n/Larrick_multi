# Multitable refresh and validation order (v66)

Source of truth: `engine_runtime_mechanism_strategy_multitable_v1.json` → `runtime_package.stage_runtime_tables`.

Refresh and validate **in this order** until the entry-stage replacement gate passes:

1. **`chem323_engine_ignition_entry_v1`** — `openfoam_runtime_chemistry_table_chem323_ignition_entry.json`
2. **`ignition_ramp`** — `openfoam_runtime_chemistry_table_chem323_ignition_ramp.json`
3. **`ignition_branch`** — `openfoam_runtime_chemistry_table_chem323_ignition_branch.json`
4. **`ignition_hot_core`** — `openfoam_runtime_chemistry_table_chem323_ignition_hot_core.json`
5. **`ignition_tail`** — `openfoam_runtime_chemistry_table_chem323_ignition_tail.json`

**Benchmark scope:** The default v66 recipe (`v66_engine_restart_recipe.json`) and `engine-restart-benchmark` replay **only the first remaining ignition stage** (entry-first). Add `--continue-across-stages` only **after** the entry authority surface passes its gate, so ramp/branch tables are not tuned while entry is still broken.

**Handoff before multi-stage benchmarks:** When you eventually enable `--continue-across-stages`, each downstream stage uses its own `stage_runtime_tables` config; keep **the same** handoff bundle, corpus philosophy (solver-emitted coverage corpora), and regression harness (`restart-regression-analysis` ordering) so stage-to-stage transitions are not fighting mismatched table generations. Refresh downstream tables **one stage at a time** after entry’s `chem323_runtime_replacement_gate_passed` is true.
