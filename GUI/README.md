# Larrick Multi GUI

This folder contains the vendored and modified desktop GUI for `larrick_multi`.
It is first-party code in this monorepo and is no longer treated as an external
CamPro dependency.

## Scope

- `desktop/`: Kotlin Compose desktop UI and adapter layer.
- `data-litvin/`: Kotlin data models used by the desktop UI.
- `docs/`: GUI integration and mode-contract notes.
- `gradle*`, `settings.gradle.kts`, `build.gradle.kts`: desktop build runtime.

## Build and run

- Build: `./gradlew :desktop:build`
- Run: `./gradlew :desktop:run`

## Backend modes

Set via JVM property `campro.backend.mode`:

- `stub` (default): deterministic local fixture behavior.
- `larrick-stub`: calls `scripts/larrick_gui_bridge.py` in deterministic bridge mode.
- `larrick-real`: bridge mode with opt-in real orchestration execution paths.
- `legacy-campro`: compatibility mode using `GUI/scripts/kotlin_bridge_cli.py`.

Optional bridge overrides:

- `-Dlarrick.gui.pythonExe=/path/to/python3`
- `-Dlarrick.gui.bridgeScript=/path/to/larrick_gui_bridge.py`

Prefer setting `LARRICK_MULTI_ROOT` so relative data/script resolution is stable.
