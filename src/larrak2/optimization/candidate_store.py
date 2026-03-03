"""Pareto candidate loading, filtering, ranking, and export utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.core.archive_io import load_archive
from larrak2.core.encoding import N_TOTAL


@dataclass(frozen=True)
class CandidateEntry:
    """One Pareto candidate with precomputed feasibility diagnostics."""

    index: int
    x_full: np.ndarray
    F: np.ndarray
    G: np.ndarray
    feasible: bool
    max_violation: float


def _fit_weights(raw: np.ndarray | None, n_obj: int) -> np.ndarray:
    if raw is None:
        return np.ones(n_obj, dtype=np.float64)

    weights = np.asarray(raw, dtype=np.float64).reshape(-1)
    if weights.size == n_obj:
        return weights
    if weights.size < n_obj:
        out = np.zeros(n_obj, dtype=np.float64)
        out[: weights.size] = weights
        return out
    return weights[:n_obj]


class CandidateStore:
    """Repository-backed candidate pool loaded from Pareto artifacts."""

    def __init__(
        self,
        *,
        source_dir: Path,
        X: np.ndarray,
        F: np.ndarray,
        G: np.ndarray,
        summary: dict[str, Any],
    ) -> None:
        self.source_dir = Path(source_dir)
        self.X = np.asarray(X, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.float64)
        self.G = np.asarray(G, dtype=np.float64)
        self.summary = dict(summary)

        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={self.X.shape}")
        if self.F.ndim != 2:
            raise ValueError(f"F must be 2D, got shape={self.F.shape}")
        if self.G.ndim != 2:
            raise ValueError(f"G must be 2D, got shape={self.G.shape}")
        if self.X.shape[0] != self.F.shape[0] or self.X.shape[0] != self.G.shape[0]:
            raise ValueError(
                f"Row mismatch: X={self.X.shape[0]}, F={self.F.shape[0]}, G={self.G.shape[0]}"
            )
        if self.X.shape[1] != N_TOTAL:
            raise ValueError(f"Expected decision width {N_TOTAL}, got {self.X.shape[1]}")

    @classmethod
    def from_archive_dir(cls, source_dir: str | Path) -> CandidateStore:
        root = Path(source_dir)
        X, F, G, summary = load_archive(root)
        return cls(source_dir=root, X=X, F=F, G=G, summary=summary)

    @property
    def n_candidates(self) -> int:
        return int(self.X.shape[0])

    def feasible_mask(self, *, tolerance: float = 0.0) -> np.ndarray:
        tol = float(tolerance)
        if self.n_candidates == 0:
            return np.zeros(0, dtype=bool)
        return np.all(self.G <= tol, axis=1)

    def feasible_indices(self, *, tolerance: float = 0.0) -> list[int]:
        mask = self.feasible_mask(tolerance=tolerance)
        return [int(i) for i in np.where(mask)[0]]

    def entry(self, idx: int) -> CandidateEntry:
        if idx < 0 or idx >= self.n_candidates:
            raise IndexError(f"Candidate index out of range: {idx}")
        G_i = self.G[idx]
        max_violation = float(max(0.0, np.max(G_i))) if G_i.size > 0 else 0.0
        return CandidateEntry(
            index=int(idx),
            x_full=self.X[idx].copy(),
            F=self.F[idx].copy(),
            G=G_i.copy(),
            feasible=bool(np.all(G_i <= 0.0)),
            max_violation=max_violation,
        )

    def rank_indices(
        self,
        *,
        objective_weights: np.ndarray | None = None,
        feasible_only: bool = True,
        violation_penalty: float = 1000.0,
        tolerance: float = 0.0,
    ) -> list[int]:
        """Return candidate indices sorted best-to-worst by score."""
        if self.n_candidates == 0:
            return []

        n_obj = int(self.F.shape[1])
        weights = _fit_weights(objective_weights, n_obj)
        feasible = self.feasible_mask(tolerance=tolerance)

        rows: list[tuple[float, int]] = []
        for i in range(self.n_candidates):
            if feasible_only and not bool(feasible[i]):
                continue
            max_v = float(max(0.0, np.max(self.G[i]))) if self.G.shape[1] > 0 else 0.0
            score = float(np.dot(weights, self.F[i])) + float(violation_penalty) * max_v * max_v
            rows.append((score, int(i)))

        if not rows and feasible_only:
            return self.rank_indices(
                objective_weights=weights,
                feasible_only=False,
                violation_penalty=violation_penalty,
                tolerance=tolerance,
            )

        rows.sort(key=lambda item: (float(item[0]), int(item[1])))
        return [int(i) for _, i in rows]

    def export_x_full_star(self, idx: int, out_path: str | Path) -> Path:
        """Export one selected candidate artifact."""
        entry = self.entry(idx)
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_dir": str(self.source_dir),
            "candidate_index": int(entry.index),
            "x_full": entry.x_full.tolist(),
            "F": entry.F.tolist(),
            "G": entry.G.tolist(),
            "feasible": bool(entry.feasible),
            "max_violation": float(entry.max_violation),
            "n_var": int(entry.x_full.size),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target
