# Larrick GUI Scope

This document defines the first-party GUI contract inside `larrick_multi`.

## Preserved GUI surfaces

- Unified Optimization host tile
- Parameter input forms and tab navigation
- Optimization status controls (start, cancel, reset)
- Result tabs: Motion Law, Gear Profiles, Efficiency, FEA Analysis, Advanced placeholder
- Accessibility panel and Debug panel

## Runtime policy

- Default runtime mode is `stub`.
- Legacy CamPro Python pipeline is compatibility-only and requires explicit mode:
  - JVM property: `-Dcampro.backend.mode=legacy-campro`
- Larrick bridge modes:
  - `-Dcampro.backend.mode=larrick-stub`
  - `-Dcampro.backend.mode=larrick-real`
- Native Rust/C++ paths are not required for default GUI startup.

## Entry point policy

- Canonical desktop entrypoint: `com.campro.v5.DesktopMainKt` (Larrick-branded runtime)
- `:desktop:run` and Compose desktop application both use that same entrypoint.
- Shadow JAR main class remains aligned to `DesktopMainKt`.

## Notes for agents

- Intended behavior: GUI can run without external compute engines.
- Observed behavior: deterministic stub payloads are written under
  `output/larrick_bridge/` for inspection and regression checks.
