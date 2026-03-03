"""Budget manager for sparse truth evaluations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Budget selection strategy labels."""

    BEST_PREDICTED = "best"
    HIGH_UNCERTAINTY = "uncertain"
    DISAGREEMENT = "disagreement"
    RANDOM = "random"


@dataclass
class BudgetAllocation:
    """Fractional split across selection strategies."""

    best_fraction: float = 0.4
    uncertain_fraction: float = 0.3
    disagreement_fraction: float = 0.2
    random_fraction: float = 0.1

    def __post_init__(self) -> None:
        total = (
            float(self.best_fraction)
            + float(self.uncertain_fraction)
            + float(self.disagreement_fraction)
            + float(self.random_fraction)
        )
        if total <= 0:
            self.best_fraction = 1.0
            self.uncertain_fraction = 0.0
            self.disagreement_fraction = 0.0
            self.random_fraction = 0.0
            return
        if not np.isclose(total, 1.0):
            LOGGER.warning("Budget fractions sum to %.6f; normalizing", total)
            self.best_fraction /= total
            self.uncertain_fraction /= total
            self.disagreement_fraction /= total
            self.random_fraction /= total


@dataclass
class BudgetState:
    """Tracks active budget usage."""

    total: int
    used: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, int(self.total) - int(self.used))

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def consume(self, count: int, reason: str = "") -> None:
        used_now = max(0, min(int(count), self.remaining))
        self.used += used_now
        self.history.append(
            {
                "count": int(used_now),
                "reason": str(reason),
                "remaining": int(self.remaining),
            }
        )


class BudgetManager:
    """Selects candidates for expensive truth evaluation."""

    def __init__(
        self,
        total_sim_calls: int,
        allocation: BudgetAllocation | None = None,
        reserve_for_validation: float = 0.1,
        seed: int = 42,
    ) -> None:
        if int(total_sim_calls) < 0:
            raise ValueError(f"total_sim_calls must be >= 0, got {total_sim_calls}")

        self.allocation = allocation or BudgetAllocation()
        self.reserve_fraction = float(np.clip(reserve_for_validation, 0.0, 0.9))

        reserve = int(round(int(total_sim_calls) * self.reserve_fraction))
        active_budget = max(0, int(total_sim_calls) - reserve)

        self.state = BudgetState(total=active_budget)
        self.validation_budget = int(reserve)
        self._rng = np.random.default_rng(int(seed))

        LOGGER.info(
            "Budget initialized: active=%d validation=%d total=%d",
            self.state.total,
            self.validation_budget,
            int(total_sim_calls),
        )

    def remaining(self) -> int:
        return int(self.state.remaining)

    def exhausted(self) -> bool:
        return bool(self.state.exhausted)

    def _normalize_vector(self, values: np.ndarray | list[float] | None, n: int) -> np.ndarray:
        if values is None:
            return np.zeros(n, dtype=np.float64)
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == n:
            return arr
        if arr.size == 0:
            return np.zeros(n, dtype=np.float64)
        if arr.size > n:
            return arr[:n]
        out = np.zeros(n, dtype=np.float64)
        out[: arr.size] = arr
        if arr.size > 0:
            out[arr.size :] = arr[-1]
        return out

    def _allocation_counts(self, batch_size: int) -> tuple[int, int, int, int]:
        n_best = int(np.floor(batch_size * self.allocation.best_fraction))
        n_unc = int(np.floor(batch_size * self.allocation.uncertain_fraction))
        n_dis = int(np.floor(batch_size * self.allocation.disagreement_fraction))
        n_rand = int(np.floor(batch_size * self.allocation.random_fraction))

        counts = [n_best, n_unc, n_dis, n_rand]
        while sum(counts) < batch_size:
            next_idx = int(
                np.argmax(
                    np.array(
                        [
                            self.allocation.best_fraction,
                            self.allocation.uncertain_fraction,
                            self.allocation.disagreement_fraction,
                            self.allocation.random_fraction,
                        ],
                        dtype=np.float64,
                    )
                )
            )
            counts[next_idx] += 1
        while sum(counts) > batch_size:
            for i in range(4):
                if counts[i] > 0 and sum(counts) > batch_size:
                    counts[i] -= 1

        if batch_size > 0 and counts[0] == 0:
            counts[0] = 1
        if batch_size > 1 and counts[1] == 0:
            counts[1] = 1
        while sum(counts) > batch_size:
            for i in (3, 2, 1, 0):
                if counts[i] > 0 and sum(counts) > batch_size:
                    counts[i] -= 1
                    break
        return int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3])

    def select(
        self,
        candidates: list[Any],
        predictions: np.ndarray,
        uncertainty: np.ndarray,
        cem_feasibility: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> list[int]:
        """Select indices for truth evaluation and consume active budget."""
        n = int(len(candidates))
        if n <= 0:
            return []

        preds = self._normalize_vector(predictions, n)
        unc = self._normalize_vector(uncertainty, n)
        feas = self._normalize_vector(cem_feasibility, n) if cem_feasibility is not None else None

        if batch_size is None:
            batch_size = min(self.state.remaining, max(1, n // 10))
        batch_size = max(0, min(int(batch_size), self.state.remaining, n))
        if batch_size <= 0:
            return []

        n_best, n_uncertain, n_disagree, n_random = self._allocation_counts(batch_size)
        selected: set[int] = set()

        def _take_by_scores(scores: np.ndarray, count: int) -> None:
            if count <= 0:
                return
            available = [i for i in range(n) if i not in selected]
            if not available:
                return
            ordered = sorted(available, key=lambda i: (-float(scores[i]), int(i)))
            for idx in ordered[:count]:
                selected.add(int(idx))

        _take_by_scores(preds, n_best)
        _take_by_scores(unc, n_uncertain)

        if feas is not None and n_disagree > 0:
            disagreement = preds * (1.0 - np.clip(feas, 0.0, 1.0))
            _take_by_scores(disagreement, n_disagree)

        if n_random > 0:
            available = [i for i in range(n) if i not in selected]
            if available:
                picks = self._rng.choice(
                    np.asarray(available, dtype=np.int64),
                    size=min(n_random, len(available)),
                    replace=False,
                )
                for idx in np.asarray(picks, dtype=np.int64).tolist():
                    selected.add(int(idx))

        if len(selected) < batch_size:
            remaining = [i for i in range(n) if i not in selected]
            ordered = sorted(remaining, key=lambda i: (-float(preds[i]), int(i)))
            for idx in ordered:
                if len(selected) >= batch_size:
                    break
                selected.add(int(idx))

        result = sorted(selected)[:batch_size]
        self.state.consume(len(result), reason="select_batch")
        return result

    def select_for_validation(
        self,
        candidates: list[Any],
        predictions: np.ndarray,
    ) -> list[int]:
        """Select top candidates from reserved validation budget."""
        n = int(len(candidates))
        if n <= 0 or self.validation_budget <= 0:
            return []
        preds = self._normalize_vector(predictions, n)
        n_select = min(n, int(self.validation_budget))
        top = sorted(range(n), key=lambda i: (-float(preds[i]), int(i)))[:n_select]
        self.validation_budget -= int(len(top))
        return [int(i) for i in top]

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_active": int(self.state.total),
            "used_active": int(self.state.used),
            "remaining_active": int(self.state.remaining),
            "validation_remaining": int(self.validation_budget),
            "history": list(self.state.history),
        }


__all__ = [
    "BudgetAllocation",
    "BudgetManager",
    "BudgetState",
    "SelectionStrategy",
]
