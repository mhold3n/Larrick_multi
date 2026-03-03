"""Canonical artifact paths used by CLI workflows and runtime loaders."""

from __future__ import annotations

from pathlib import Path

OUTPUTS_ROOT = Path("outputs")
ARTIFACTS_ROOT = OUTPUTS_ROOT / "artifacts"
SURROGATES_ROOT = ARTIFACTS_ROOT / "surrogates"

DEFAULT_OPENFOAM_NN_DIR = SURROGATES_ROOT / "openfoam_nn"
DEFAULT_OPENFOAM_NN_ARTIFACT = DEFAULT_OPENFOAM_NN_DIR / "openfoam_breathing.pt"

DEFAULT_CALCULIX_NN_DIR = SURROGATES_ROOT / "calculix_nn"
DEFAULT_CALCULIX_NN_ARTIFACT = DEFAULT_CALCULIX_NN_DIR / "calculix_stress.pt"

DEFAULT_GEAR_LOSS_NN_DIR = SURROGATES_ROOT / "gear_loss_nn"
DEFAULT_MACHINING_NN_DIR = SURROGATES_ROOT / "machining_nn"
DEFAULT_MACHINING_NN_ARTIFACT = DEFAULT_MACHINING_NN_DIR / "machining_surrogate.pth"
DEFAULT_HIFI_SURROGATE_DIR = SURROGATES_ROOT / "hifi"
DEFAULT_SURROGATE_V1_DIR = SURROGATES_ROOT / "v1_gbr"
DEFAULT_INITIALIZATION_SURROGATE_DIR = SURROGATES_ROOT / "initialization_voxel"
DEFAULT_STACK_SURROGATE_DIR = SURROGATES_ROOT / "stack_f1"
DEFAULT_STACK_SURROGATE_ARTIFACT = DEFAULT_STACK_SURROGATE_DIR / "stack_f1_surrogate.npz"

DEPRECATED_MODELS_ROOT = Path("models")
DEPRECATED_SRC_RUNTIME_ROOT = Path("src")


def prefer_existing_path(primary: Path, *fallbacks: Path) -> Path:
    """Return first existing path among primary + fallbacks, else primary."""
    candidates = [Path(primary), *(Path(p) for p in fallbacks)]
    for path in candidates:
        if path.exists():
            return path
    return Path(primary)


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except Exception:
        return False


def _assert_not_deprecated_models_path(path: str | Path, *, purpose: str) -> Path:
    """Hard guard against using deprecated ``models/`` paths."""
    target = Path(path)

    legacy_root = (Path.cwd() / DEPRECATED_MODELS_ROOT).resolve(strict=False)
    src_root = (Path.cwd() / DEPRECATED_SRC_RUNTIME_ROOT).resolve(strict=False)
    target_abs = (Path.cwd() / target).resolve(strict=False)
    if _is_within(target_abs, legacy_root):
        raise ValueError(
            f"Refusing to use deprecated models path for {purpose}: '{target}'. "
            "Use outputs/artifacts/... instead."
        )
    if _is_within(target_abs, src_root):
        raise ValueError(
            f"Refusing to use runtime artifact path under src/ for {purpose}: '{target}'. "
            "Use outputs/artifacts/... instead."
        )
    return target


def assert_not_legacy_models_path(path: str | Path, *, purpose: str) -> Path:
    """Reject deprecated ``models/`` paths for reads/writes."""
    return _assert_not_deprecated_models_path(path, purpose=purpose)


def assert_not_legacy_models_write(path: str | Path, *, purpose: str) -> Path:
    """Backward-compatible alias for write guards."""
    return _assert_not_deprecated_models_path(path, purpose=purpose)
