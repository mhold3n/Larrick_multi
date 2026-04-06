from __future__ import annotations

import json
from pathlib import Path

from larrak2.simulation_validation.engine_runtime_mechanism import resolve_engine_runtime_package


def _write_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "package_manifest.json").write_text(
        json.dumps({"package_id": "chem323_reduced_v2512", "package_hash": "runtime-hash"}),
        encoding="utf-8",
    )


def test_resolve_engine_runtime_package_prefers_repo_relative_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    package_dir = tmp_path / "mechanisms" / "openfoam" / "v2512" / "chem323_reduced"
    _write_package(package_dir)
    strategy_dir = tmp_path / "data" / "simulation_validation"
    strategy_dir.mkdir(parents=True)
    strategy_path = strategy_dir / "strategy.json"
    strategy_path.write_text(
        json.dumps(
            {
                "runtime_package": {
                    "label": "chem323_runtime",
                    "package_dir": "mechanisms/openfoam/v2512/chem323_reduced",
                }
            }
        ),
        encoding="utf-8",
    )

    resolved_dir, manifest = resolve_engine_runtime_package(config_path=str(strategy_path))

    assert resolved_dir.resolve() == package_dir.resolve()
    assert manifest["package_id"] == "chem323_reduced_v2512"
