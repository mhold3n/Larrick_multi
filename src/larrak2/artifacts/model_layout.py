"""Logical model-artifact layout policy (strict outputs-only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from larrak2.core.artifact_paths import (
    DEFAULT_CALCULIX_NN_ARTIFACT,
    DEFAULT_HIFI_SURROGATE_DIR,
    DEFAULT_INITIALIZATION_SURROGATE_DIR,
    DEFAULT_MACHINING_NN_ARTIFACT,
    DEFAULT_OPENFOAM_NN_ARTIFACT,
    DEFAULT_SURROGATE_V1_DIR,
)


@dataclass(frozen=True)
class ModelArtifactSpec:
    """One canonical model family destination."""

    key: str
    canonical: Path
    kind: str  # file | directory
    purpose: str


def planned_model_layout() -> list[ModelArtifactSpec]:
    """Return canonical model families under outputs/artifacts/."""
    return [
        ModelArtifactSpec(
            key="openfoam_nn",
            canonical=DEFAULT_OPENFOAM_NN_ARTIFACT,
            kind="file",
            purpose="OpenFOAM NN breathing surrogate",
        ),
        ModelArtifactSpec(
            key="calculix_nn",
            canonical=DEFAULT_CALCULIX_NN_ARTIFACT,
            kind="file",
            purpose="CalculiX NN stress surrogate",
        ),
        ModelArtifactSpec(
            key="v1_gbr",
            canonical=DEFAULT_SURROGATE_V1_DIR,
            kind="directory",
            purpose="v1 GBR surrogate ensemble artifacts",
        ),
        ModelArtifactSpec(
            key="machining_nn",
            canonical=DEFAULT_MACHINING_NN_ARTIFACT,
            kind="file",
            purpose="Machining NN surrogate artifact",
        ),
        ModelArtifactSpec(
            key="hifi",
            canonical=DEFAULT_HIFI_SURROGATE_DIR,
            kind="directory",
            purpose="HiFi ensemble surrogate artifacts",
        ),
        ModelArtifactSpec(
            key="initialization_voxel",
            canonical=DEFAULT_INITIALIZATION_SURROGATE_DIR,
            kind="directory",
            purpose="Initialization geometric surrogate artifacts",
        ),
    ]


def ensure_model_layout(*, repo_root: str | Path = ".") -> list[Path]:
    """Create canonical artifact directories and return their canonical paths."""
    root = Path(repo_root)
    resolved: list[Path] = []
    for spec in planned_model_layout():
        target = root / spec.canonical
        if spec.kind == "file":
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
        resolved.append(target)
    return resolved
