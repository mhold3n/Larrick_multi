# Larrick Suite Mapping

This map defines how the vendored `GUI/` code maps to first-party Larrick
suite modules. The GUI is now owned inside `larrick_multi` and is no longer
treated as an external CamPro dependency.

## Port to repository mapping

| GUI port contract | Target module | Current adapter | Planned real adapter |
|---|---|---|---|
| `OptimizationPort` | `scripts/larrick_gui_bridge.py` + `larrak-optimization` | `StubOptimizationAdapter` / `LarrickOptimizationAdapter` | `LarrickOptimizationAdapter` (`larrick-real`) |
| `AnalysisPort` | `scripts/larrick_gui_bridge.py` + `larrak-analysis` | `LarrickAnalysisAdapter` | `LarrickAnalysisAdapter` (real analysis payload) |
| `EnginePort` | `scripts/larrick_gui_bridge.py` + `larrak-engines` | `LarrickEngineAdapter` | `LarrickEngineAdapter` (real engine payload) |
| `OrchestrationPort` | `scripts/larrick_gui_bridge.py` + `larrak-orchestration` | `LarrickOrchestrationAdapter` | `LarrickOrchestrationAdapter` (`larrick-real`) |
| `SimulationPort` | `scripts/larrick_gui_bridge.py` + `larrak-simulation` | `LarrickSimulationAdapter` | `LarrickSimulationAdapter` (real simulation payload) |
| `CorePort` | `larrak-core` | Placeholder interface only | `LarrickCoreAdapter` |

## Existing compatibility path

- `LegacyPythonOptimizationAdapter` is retained only for transition checks.
- Primary integration path is the first-party Larrick bridge script.

## Replacement checklist

1. Keep bridge request/response JSON contract stable for GUI callers.
2. Implement/extend corresponding `*Adapter` classes in `desktop/pipeline`.
3. Use stub-first behavior for deterministic dashboard startup.
4. Add contract tests for stub and `larrick-real` mode switches.
5. Promote real handlers by mode flag when acceptance tests pass.

## First-party ownership notes

- `GUI/` is a vendored and modified first-party subtree in this monorepo.
- Keep this file synchronized with actual adapter ownership and mode behavior.
