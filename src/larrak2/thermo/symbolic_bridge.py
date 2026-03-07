"""Thermo symbolic surrogate bridge for CasADi slice NLP overlays."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..core.artifact_paths import DEFAULT_THERMO_SYMBOLIC_ARTIFACT
from ..core.encoding import N_TOTAL
from ..core.types import EvalContext
from ..surrogate.quality_contract import load_quality_report, thermo_symbolic_quality_fail_reasons
from ..surrogate.stack.runtime import parse_feature_index
from .symbolic_artifact import ThermoSymbolicArtifact, load_thermo_symbolic_artifact

LOGGER = logging.getLogger(__name__)


def _import_casadi():
    try:
        import casadi as ca
    except Exception as exc:  # pragma: no cover - handled by caller
        raise ImportError(
            "CasADi import failed in active runtime "
            f"({sys.executable}) for thermo symbolic bridge: "
            f"{type(exc).__name__}: {exc}. "
            "Install optional dependency in this interpreter: pip install -e '.[casadi]'"
        ) from exc
    return ca


def assemble_thermo_symbolic_feature_vector(
    *,
    artifact: ThermoSymbolicArtifact,
    x_full_sym,
    rpm: float,
    torque: float,
):
    """Build symbolic feature vector from full design variable + operating point."""
    ca = _import_casadi()
    feats = []
    for name in artifact.feature_names:
        if name == "rpm":
            feats.append(ca.DM([float(rpm)])[0])
            continue
        if name == "torque":
            feats.append(ca.DM([float(torque)])[0])
            continue
        idx = parse_feature_index(name)
        if idx is None:
            raise ValueError(f"Unsupported thermo symbolic feature '{name}'")
        feats.append(x_full_sym[int(idx)])
    return ca.vertcat(*feats)


def symbolic_thermo_forward(artifact: ThermoSymbolicArtifact, x_features_sym):
    """Evaluate thermo symbolic affine model and return output vector."""
    ca = _import_casadi()
    x_mean = ca.DM(np.asarray(artifact.x_mean, dtype=np.float64))
    x_std = ca.DM(np.where(np.abs(artifact.x_std) > 0.0, artifact.x_std, 1.0))
    y_mean = ca.DM(np.asarray(artifact.y_mean, dtype=np.float64))
    y_std = ca.DM(np.where(np.abs(artifact.y_std) > 0.0, artifact.y_std, 1.0))
    W = ca.DM(np.asarray(artifact.weight, dtype=np.float64))
    b = ca.DM(np.asarray(artifact.bias, dtype=np.float64))

    h = (x_features_sym - x_mean) / x_std
    y_n = W @ h + b
    return y_n * y_std + y_mean


def numeric_thermo_forward(
    artifact: ThermoSymbolicArtifact,
    x_full: np.ndarray,
    *,
    rpm: float,
    torque: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate thermo symbolic model numerically for one sample."""
    x = np.asarray(x_full, dtype=np.float64).reshape(-1)
    feats = np.zeros(len(artifact.feature_names), dtype=np.float64)
    for i, name in enumerate(artifact.feature_names):
        if name == "rpm":
            feats[i] = float(rpm)
            continue
        if name == "torque":
            feats[i] = float(torque)
            continue
        idx = parse_feature_index(name)
        if idx is None or idx < 0 or idx >= x.size:
            raise ValueError(f"Invalid thermo symbolic feature '{name}' for vector size {x.size}")
        feats[i] = float(x[idx])

    x_scale = np.where(np.abs(artifact.x_std) > 0.0, artifact.x_std, 1.0)
    y_scale = np.where(np.abs(artifact.y_std) > 0.0, artifact.y_std, 1.0)
    y = (
        artifact.weight @ ((feats - artifact.x_mean) / x_scale) + artifact.bias
    ) * y_scale + artifact.y_mean
    n_obj = len(artifact.objective_names)
    return np.asarray(y[:n_obj], dtype=np.float64), np.asarray(y[n_obj:], dtype=np.float64)


def _raise_or_warn(mode: str, message: str) -> None:
    if mode == "strict":
        raise RuntimeError(message)
    if mode == "warn":
        LOGGER.warning(message)


def _strict_remediation_message(*, artifact_path: str, reason: str) -> str:
    train_cmd = (
        "python -m larrak2.cli.run train-thermo-symbolic "
        "--fidelity <fidelity> --rpm <rpm> --torque <torque>"
    )
    override_hint = (
        f'python -m larrak2.cli.run <entrypoint> --thermo-symbolic-artifact-path "{artifact_path}"'
    )
    return (
        f"{reason} | thermo_symbolic_path='{artifact_path}'. "
        f"Remediation: run `{train_cmd}` then retry, or override with `{override_hint}`."
    )


def _find_duplicates(names: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    dups: set[str] = set()
    for name in names:
        if name in seen:
            dups.add(name)
        seen.add(name)
    return sorted(dups)


def _preflight_overlay_contract_violations(
    *,
    artifact: ThermoSymbolicArtifact,
    stack_objective_names: tuple[str, ...],
    stack_constraint_names: tuple[str, ...],
) -> list[str]:
    errs: list[str] = []

    stack_obj_dups = _find_duplicates(tuple(str(v) for v in stack_objective_names))
    stack_con_dups = _find_duplicates(tuple(str(v) for v in stack_constraint_names))
    art_obj_dups = _find_duplicates(tuple(str(v) for v in artifact.objective_names))
    art_con_dups = _find_duplicates(tuple(str(v) for v in artifact.constraint_names))
    if stack_obj_dups:
        errs.append(f"stack objective names contain duplicates: {stack_obj_dups}")
    if stack_con_dups:
        errs.append(f"stack constraint names contain duplicates: {stack_con_dups}")
    if art_obj_dups:
        errs.append(f"artifact objective names contain duplicates: {art_obj_dups}")
    if art_con_dups:
        errs.append(f"artifact constraint names contain duplicates: {art_con_dups}")

    stack_obj_set = {str(v) for v in stack_objective_names}
    stack_con_set = {str(v) for v in stack_constraint_names}
    art_obj_set = {str(v) for v in artifact.objective_names}
    art_con_set = {str(v) for v in artifact.constraint_names}

    stack_overlap = sorted(stack_obj_set.intersection(stack_con_set))
    artifact_overlap = sorted(art_obj_set.intersection(art_con_set))
    if stack_overlap:
        errs.append(
            f"ambiguous stack output naming (objective/constraint overlap): {stack_overlap}"
        )
    if artifact_overlap:
        errs.append(
            "ambiguous thermo symbolic artifact naming "
            f"(objective/constraint overlap): {artifact_overlap}"
        )

    missing_obj = [name for name in artifact.objective_names if str(name) not in stack_obj_set]
    missing_con = [name for name in artifact.constraint_names if str(name) not in stack_con_set]
    if missing_obj:
        errs.append(f"artifact objectives are not subset-compatible with stack: {missing_obj}")
    if missing_con:
        errs.append(f"artifact constraints are not subset-compatible with stack: {missing_con}")

    bad_features: list[str] = []
    for name in artifact.feature_names:
        if name in {"rpm", "torque"}:
            continue
        idx = parse_feature_index(name)
        if idx is None or idx < 0 or idx >= int(N_TOTAL):
            bad_features.append(str(name))
    if bad_features:
        errs.append(
            "artifact feature names must resolve to x_###, rpm, or torque "
            f"within [0, {int(N_TOTAL) - 1}]: {bad_features}"
        )

    return errs


def apply_thermo_symbolic_overlay(
    *,
    ctx: EvalContext,
    x_full_sym,
    stack_objective_names: tuple[str, ...],
    stack_constraint_names: tuple[str, ...],
    F_hat,
    G_hat,
) -> tuple[Any, Any, dict[str, Any]]:
    """Overlay stack objective/constraint expressions with thermo symbolic terms."""
    mode = str(getattr(ctx, "thermo_symbolic_mode", "strict") or "strict").lower()
    diag: dict[str, Any] = {
        "thermo_symbolic_mode": mode,
        "thermo_symbolic_used": False,
        "thermo_symbolic_version": "",
        "thermo_symbolic_path": "",
        "thermo_symbolic_overlay_objectives": [],
        "thermo_symbolic_overlay_constraints": [],
    }
    if mode == "off":
        return F_hat, G_hat, diag

    artifact_path = str(getattr(ctx, "thermo_symbolic_artifact_path", "") or "").strip() or str(
        DEFAULT_THERMO_SYMBOLIC_ARTIFACT
    )
    diag["thermo_symbolic_path"] = artifact_path
    try:
        artifact = load_thermo_symbolic_artifact(
            artifact_path,
            validation_mode=mode,
        )
    except Exception as exc:  # pragma: no cover - exercised by strict/warn branches
        msg = _strict_remediation_message(
            artifact_path=artifact_path,
            reason=f"Failed to load thermo symbolic artifact: {exc}",
        )
        _raise_or_warn(mode, msg)
        diag["thermo_symbolic_error"] = msg
        return F_hat, G_hat, diag

    if mode == "warn":
        qpath = Path(artifact_path).parent / "quality_report.json"
        if qpath.exists():
            try:
                reasons = thermo_symbolic_quality_fail_reasons(load_quality_report(qpath))
            except Exception:
                reasons = []
            if reasons:
                diag["thermo_symbolic_error"] = (
                    "thermo symbolic quality degraded in warn mode: " + "; ".join(reasons)
                )

    if int(artifact.fidelity) != int(ctx.fidelity):
        msg = _strict_remediation_message(
            artifact_path=artifact_path,
            reason=(
                "Thermo symbolic artifact fidelity mismatch: "
                f"artifact={artifact.fidelity}, context={ctx.fidelity}"
            ),
        )
        _raise_or_warn(mode, msg)
        diag["thermo_symbolic_error"] = msg
        return F_hat, G_hat, diag

    violations = _preflight_overlay_contract_violations(
        artifact=artifact,
        stack_objective_names=stack_objective_names,
        stack_constraint_names=stack_constraint_names,
    )
    if violations:
        msg = _strict_remediation_message(
            artifact_path=artifact_path,
            reason="Thermo symbolic overlay contract violation: " + "; ".join(violations),
        )
        _raise_or_warn(mode, msg)
        diag["thermo_symbolic_error"] = msg
        return F_hat, G_hat, diag

    feats_sym = assemble_thermo_symbolic_feature_vector(
        artifact=artifact,
        x_full_sym=x_full_sym,
        rpm=float(ctx.rpm),
        torque=float(ctx.torque),
    )
    y = symbolic_thermo_forward(artifact, feats_sym)
    n_obj = len(artifact.objective_names)
    obj_expr = y[:n_obj]
    con_expr = y[n_obj:]

    obj_idx_map = {name: i for i, name in enumerate(stack_objective_names)}
    con_idx_map = {name: i for i, name in enumerate(stack_constraint_names)}
    obj_overlay: dict[int, Any] = {}
    con_overlay: dict[int, Any] = {}

    for i, name in enumerate(artifact.objective_names):
        if name in obj_idx_map:
            obj_overlay[int(obj_idx_map[name])] = obj_expr[i]
    for i, name in enumerate(artifact.constraint_names):
        if name in con_idx_map:
            con_overlay[int(con_idx_map[name])] = con_expr[i]

    ca = _import_casadi()
    F_terms = [F_hat[i] for i in range(len(stack_objective_names))]
    G_terms = [G_hat[i] for i in range(len(stack_constraint_names))]
    for idx, expr in obj_overlay.items():
        F_terms[int(idx)] = expr
    for idx, expr in con_overlay.items():
        G_terms[int(idx)] = expr

    F_new = ca.vertcat(*F_terms) if F_terms else ca.MX([])
    G_new = ca.vertcat(*G_terms) if G_terms else ca.MX([])
    diag.update(
        {
            "thermo_symbolic_used": True,
            "thermo_symbolic_version": str(artifact.version_hash),
            "thermo_symbolic_path": str(Path(artifact_path)),
            "thermo_symbolic_overlay_objectives": [
                str(stack_objective_names[idx]) for idx in sorted(obj_overlay.keys())
            ],
            "thermo_symbolic_overlay_constraints": [
                str(stack_constraint_names[idx]) for idx in sorted(con_overlay.keys())
            ],
        }
    )
    return F_new, G_new, diag
