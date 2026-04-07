"""Active-variable set selection for high-dimensional slice refinement."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from larrak_runtime.core.encoding import N_TOTAL, bounds, group_indices
from larrak_runtime.core.evaluator import evaluate_candidate
from larrak_runtime.core.types import EvalContext


@dataclass
class SliceSelection:
    """Active/frozen variable split and sensitivity diagnostics."""

    active_indices: list[int]
    frozen_indices: list[int]
    scores: list[float]


def _normalize_weights(weights: np.ndarray, n_obj: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_obj, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != n_obj:
        raise ValueError(f"weights length {w.size} does not match n_obj {n_obj}")
    return w


def _scalarized_value(
    F: np.ndarray,
    G: np.ndarray,
    *,
    mode: str,
    weights: np.ndarray | None,
    eps_constraints: np.ndarray | None,
    violation_penalty: float,
) -> float:
    F = np.asarray(F, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    if mode == "weighted_sum":
        w = _normalize_weights(weights, F.size)
        obj = float(np.dot(w, F))
    elif mode == "eps_constraint":
        obj = float(F[0])
        if F.size > 1:
            eps = np.asarray(
                eps_constraints if eps_constraints is not None else F, dtype=np.float64
            )
            if eps.size != F.size:
                raise ValueError(f"eps_constraints length {eps.size} does not match n_obj {F.size}")
            obj += violation_penalty * float(np.maximum(F[1:] - eps[1:], 0.0).sum())
    else:
        raise ValueError(f"Unknown mode: {mode}")

    g_violation = np.maximum(G, 0.0)
    obj += violation_penalty * float(np.dot(g_violation, g_violation))
    return obj


def sensitivity_scores(
    x0: np.ndarray,
    ctx: EvalContext,
    *,
    mode: str = "weighted_sum",
    weights: np.ndarray | None = None,
    eps_constraints: np.ndarray | None = None,
    step_frac: float = 1e-3,
    violation_penalty: float = 10.0,
    freeze_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute finite-difference sensitivity scores for each decision variable."""
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.size != N_TOTAL:
        raise ValueError(f"Expected x0 length {N_TOTAL}, got {x0.size}")

    mask = (
        np.zeros(N_TOTAL, dtype=bool)
        if freeze_mask is None
        else np.asarray(freeze_mask, dtype=bool)
    )
    if mask.size != N_TOTAL:
        raise ValueError(f"freeze_mask length {mask.size} does not match N_TOTAL={N_TOTAL}")

    base = evaluate_candidate(x0, ctx)
    base_scalar = _scalarized_value(
        base.F,
        base.G,
        mode=mode,
        weights=weights,
        eps_constraints=eps_constraints,
        violation_penalty=violation_penalty,
    )
    _ = base_scalar

    xl, xu = bounds()
    scores = np.full(N_TOTAL, -np.inf, dtype=np.float64)

    for idx in range(N_TOTAL):
        if mask[idx]:
            continue

        delta = max(abs(x0[idx]) * step_frac, (xu[idx] - xl[idx]) * step_frac, 1e-8)

        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[idx] = min(x0[idx] + delta, xu[idx])
        x_minus[idx] = max(x0[idx] - delta, xl[idx])

        plus = evaluate_candidate(x_plus, ctx)
        minus = evaluate_candidate(x_minus, ctx)

        s_plus = _scalarized_value(
            plus.F,
            plus.G,
            mode=mode,
            weights=weights,
            eps_constraints=eps_constraints,
            violation_penalty=violation_penalty,
        )
        s_minus = _scalarized_value(
            minus.F,
            minus.G,
            mode=mode,
            weights=weights,
            eps_constraints=eps_constraints,
            violation_penalty=violation_penalty,
        )

        deriv = (s_plus - s_minus) / max((x_plus[idx] - x_minus[idx]), 1e-12)
        scores[idx] = abs(float(deriv))

    return scores


def _rank_indices(indices: list[int], scores: np.ndarray) -> list[int]:
    return sorted(indices, key=lambda i: (-float(scores[i]), int(i)))


def select_active_set(
    x0: np.ndarray,
    ctx: EvalContext,
    *,
    active_k: int | None = None,
    min_per_group: int = 1,
    method: str = "sensitivity",
    mode: str = "weighted_sum",
    weights: np.ndarray | None = None,
    eps_constraints: np.ndarray | None = None,
    freeze_mask: np.ndarray | None = None,
) -> SliceSelection:
    """Select active decision variables for local refinement."""
    if method != "sensitivity":
        raise ValueError(f"Unsupported slice selection method: {method}")

    x0 = np.asarray(x0, dtype=np.float64)
    if x0.size != N_TOTAL:
        raise ValueError(f"Expected x0 length {N_TOTAL}, got {x0.size}")

    mask = (
        np.zeros(N_TOTAL, dtype=bool)
        if freeze_mask is None
        else np.asarray(freeze_mask, dtype=bool)
    )
    if mask.size != N_TOTAL:
        raise ValueError(f"freeze_mask length {mask.size} does not match N_TOTAL={N_TOTAL}")

    scores = sensitivity_scores(
        x0,
        ctx,
        mode=mode,
        weights=weights,
        eps_constraints=eps_constraints,
        freeze_mask=mask,
    )

    available = [i for i in range(N_TOTAL) if not mask[i]]
    if not available:
        return SliceSelection(
            active_indices=[], frozen_indices=list(range(N_TOTAL)), scores=scores.tolist()
        )

    default_k = max(6, int(ceil(0.25 * N_TOTAL)))
    k = default_k if active_k is None else int(active_k)
    k = max(1, min(k, len(available)))

    groups = group_indices()
    selected: list[int] = []

    # Enforce per-group minimum active members where possible.
    for _, idxs in sorted(groups.items()):
        candidates = [i for i in idxs if not mask[i]]
        if not candidates:
            continue
        ranked = _rank_indices(candidates, scores)
        for i in ranked[: max(0, min_per_group)]:
            if i not in selected:
                selected.append(i)

    # Ensure k can satisfy group floor.
    if len(selected) > k:
        k = len(selected)

    ranked_all = _rank_indices(available, scores)
    for i in ranked_all:
        if len(selected) >= k:
            break
        if i not in selected:
            selected.append(i)

    selected = sorted(selected)
    frozen = [i for i in range(N_TOTAL) if i not in selected]

    return SliceSelection(
        active_indices=selected,
        frozen_indices=frozen,
        scores=scores.tolist(),
    )
