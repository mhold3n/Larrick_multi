"""Canonical artifact paths used by CLI workflows and runtime loaders."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np

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
# Legacy f1 thermo path retained for compatibility. New canonical path is
# outputs/artifacts/surrogates/thermo_symbolic_f1/thermo_symbolic_f1.npz.
DEFAULT_THERMO_SYMBOLIC_DIR = SURROGATES_ROOT / "thermo_symbolic"
DEFAULT_THERMO_SYMBOLIC_ARTIFACT = DEFAULT_THERMO_SYMBOLIC_DIR / "thermo_symbolic_f1.npz"

DEPRECATED_MODELS_ROOT = Path("models")
DEPRECATED_SRC_RUNTIME_ROOT = Path("src")


def prefer_existing_path(primary: Path, *fallbacks: Path) -> Path:
    """Return first existing path among primary + fallbacks, else primary."""
    candidates = [Path(primary), *(Path(p) for p in fallbacks)]
    for path in candidates:
        if path.exists():
            return path
    return Path(primary)


def _normalize_fidelity(fidelity: int) -> int:
    f = int(fidelity)
    if f < 0:
        raise ValueError(f"fidelity must be >= 0, got {f}")
    return f


def stack_artifact_dir_for_fidelity(fidelity: int) -> Path:
    """Canonical stack-surrogate directory by fidelity."""
    f = _normalize_fidelity(fidelity)
    return SURROGATES_ROOT / f"stack_f{f}"


def stack_artifact_path_for_fidelity(fidelity: int) -> Path:
    """Canonical stack-surrogate artifact path by fidelity."""
    f = _normalize_fidelity(fidelity)
    return stack_artifact_dir_for_fidelity(f) / f"stack_f{f}_surrogate.npz"


def thermo_symbolic_dir_for_fidelity(fidelity: int) -> Path:
    """Canonical thermo-symbolic directory by fidelity."""
    f = _normalize_fidelity(fidelity)
    return SURROGATES_ROOT / f"thermo_symbolic_f{f}"


def thermo_symbolic_artifact_path_for_fidelity(fidelity: int) -> Path:
    """Canonical thermo-symbolic artifact path by fidelity."""
    f = _normalize_fidelity(fidelity)
    return thermo_symbolic_dir_for_fidelity(f) / f"thermo_symbolic_f{f}.npz"


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


def _stack_missing_message(*, fidelity: int, path: Path) -> str:
    return (
        "Missing required stack surrogate artifact for CasADi refinement: "
        f"'{path}' (fidelity={int(fidelity)}). "
        "Remediation: python -m larrak2.cli.run train-stack-surrogate "
        f"--fidelity {int(fidelity)}"
    )


def _thermo_missing_message(*, fidelity: int, path: Path) -> str:
    return (
        "Missing required thermo symbolic artifact for CasADi refinement: "
        f"'{path}' (fidelity={int(fidelity)}). "
        "Remediation: python -m larrak2.cli.run train-thermo-symbolic "
        f"--fidelity {int(fidelity)}"
    )


def _stack_fidelity_mismatch_message(*, expected: int, detected: int, path: Path) -> str:
    return (
        "Stack surrogate artifact fidelity mismatch: "
        f"path='{path}', requested_fidelity={int(expected)}, detected_fidelity={int(detected)}. "
        "Use a matching artifact or retrain with: "
        "python -m larrak2.cli.run train-stack-surrogate "
        f"--fidelity {int(expected)}"
    )


def _thermo_fidelity_mismatch_message(*, expected: int, detected: int, path: Path) -> str:
    return (
        "Thermo symbolic artifact fidelity mismatch: "
        f"path='{path}', requested_fidelity={int(expected)}, detected_fidelity={int(detected)}. "
        "Use a matching artifact or retrain with: "
        "python -m larrak2.cli.run train-thermo-symbolic "
        f"--fidelity {int(expected)}"
    )


def _fidelity_from_path_hint(path: Path, *, kind: str) -> int | None:
    text = path.as_posix()
    pattern = r"stack_f(\d+)" if kind == "stack" else r"thermo_symbolic_f(\d+)"
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def _fidelity_from_artifact_meta(path: Path) -> int | None:
    try:
        with np.load(path, allow_pickle=True) as data:
            if "__meta_json__" not in data:
                return None
            raw = data["__meta_json__"].item()
            meta = json.loads(str(raw))
            if not isinstance(meta, dict) or "fidelity" not in meta:
                return None
            return int(meta["fidelity"])
    except Exception:
        return None


def _assert_stack_fidelity_match(*, path: Path, fidelity: int) -> None:
    hinted = _fidelity_from_path_hint(path, kind="stack")
    meta = _fidelity_from_artifact_meta(path) if path.exists() else None
    for detected in (hinted, meta):
        if detected is not None and int(detected) != int(fidelity):
            raise ValueError(
                _stack_fidelity_mismatch_message(
                    expected=int(fidelity),
                    detected=int(detected),
                    path=path,
                )
            )


def _assert_thermo_fidelity_match(*, path: Path, fidelity: int) -> None:
    hinted = _fidelity_from_path_hint(path, kind="thermo")
    meta = _fidelity_from_artifact_meta(path) if path.exists() else None
    for detected in (hinted, meta):
        if detected is not None and int(detected) != int(fidelity):
            raise ValueError(
                _thermo_fidelity_mismatch_message(
                    expected=int(fidelity),
                    detected=int(detected),
                    path=path,
                )
            )


def resolve_stack_artifact_path(
    *,
    fidelity: int,
    explicit_path: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve stack artifact path for runtime/training consumers."""
    f = _normalize_fidelity(fidelity)
    if explicit_path is not None and str(explicit_path).strip():
        path = assert_not_legacy_models_path(explicit_path, purpose="stack surrogate model")
    else:
        path = stack_artifact_path_for_fidelity(f)
    if must_exist and not Path(path).exists():
        raise FileNotFoundError(_stack_missing_message(fidelity=f, path=Path(path)))
    if Path(path).exists():
        _assert_stack_fidelity_match(path=Path(path), fidelity=f)
    return Path(path)


def resolve_thermo_symbolic_artifact_path(
    *,
    fidelity: int,
    explicit_path: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve thermo-symbolic artifact path for runtime/training consumers."""
    f = _normalize_fidelity(fidelity)
    if explicit_path is not None and str(explicit_path).strip():
        path = assert_not_legacy_models_path(explicit_path, purpose="thermo symbolic artifact")
        if must_exist and not Path(path).exists():
            raise FileNotFoundError(_thermo_missing_message(fidelity=f, path=Path(path)))
        if Path(path).exists():
            _assert_thermo_fidelity_match(path=Path(path), fidelity=f)
        return Path(path)

    canonical = thermo_symbolic_artifact_path_for_fidelity(f)
    if must_exist and canonical.exists():
        _assert_thermo_fidelity_match(path=canonical, fidelity=f)
        return canonical
    if must_exist and int(f) == 1 and DEFAULT_THERMO_SYMBOLIC_ARTIFACT.exists():
        _assert_thermo_fidelity_match(path=DEFAULT_THERMO_SYMBOLIC_ARTIFACT, fidelity=f)
        warnings.warn(
            (
                "Using deprecated legacy thermo symbolic artifact path "
                f"'{DEFAULT_THERMO_SYMBOLIC_ARTIFACT}'. "
                "Move to canonical path "
                f"'{canonical}'."
            ),
            UserWarning,
            stacklevel=2,
        )
        return DEFAULT_THERMO_SYMBOLIC_ARTIFACT
    if must_exist and not canonical.exists():
        raise FileNotFoundError(_thermo_missing_message(fidelity=f, path=canonical))
    if canonical.exists():
        _assert_thermo_fidelity_match(path=canonical, fidelity=f)
    return canonical
