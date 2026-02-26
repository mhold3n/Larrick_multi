"""CEM adapter for orchestration feasibility and generation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from larrak2.core.encoding import N_GEAR, N_THERMO, N_TOTAL, bounds, decode_candidate
from larrak2.realworld.surrogates import (
    RealWorldSurrogateParams,
    evaluate_realworld_surrogates,
)

LOGGER = logging.getLogger(__name__)


class CEMAdapter:
    """Generates high-dimensional candidates and scores CEM feasibility."""

    def __init__(self) -> None:
        self._xl, self._xu = bounds()

    def generate_batch(
        self,
        params: dict[str, Any],
        n: int,
        *,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        n = max(0, int(n))
        if n == 0:
            return []

        seed_x = np.asarray(params.get("x_seed", params.get("x0", [])), dtype=np.float64).reshape(-1)
        has_seed = seed_x.size == N_TOTAL
        spread = float(params.get("seed_spread", 0.10))
        spread = max(1e-6, min(1.0, spread))
        width = (self._xu - self._xl) * spread

        candidates: list[dict[str, Any]] = []
        for i in range(n):
            if has_seed:
                if i == 0:
                    x = np.clip(seed_x, self._xl, self._xu)
                else:
                    x = np.clip(seed_x + rng.normal(0.0, width, size=N_TOTAL), self._xl, self._xu)
            else:
                x = rng.uniform(self._xl, self._xu, size=N_TOTAL)
            candidates.append({"x": np.asarray(x, dtype=np.float64)})
        return candidates

    def check_feasibility(self, candidate: dict[str, Any]) -> tuple[bool, float]:
        x = np.asarray(candidate.get("x", []), dtype=np.float64).reshape(-1)
        if x.size != N_TOTAL:
            return False, 0.0

        try:
            decoded = decode_candidate(x)
            rw = decoded.realworld
            rw_params = RealWorldSurrogateParams(
                surface_finish_level=float(rw.surface_finish_level),
                lube_mode_level=float(rw.lube_mode_level),
                material_quality_level=float(rw.material_quality_level)
                if rw.material_quality_level is not None
                else 0.5,
                coating_level=float(rw.coating_level),
                hunting_level=float(rw.hunting_level),
                oil_flow_level=float(rw.oil_flow_level),
                oil_supply_temp_level=float(rw.oil_supply_temp_level),
                evacuation_level=float(rw.evacuation_level),
                material_state=rw.material_state,
            )
            cem = evaluate_realworld_surrogates(rw_params)
            feasible = bool(
                cem.lambda_min >= 1.0
                and cem.scuff_margin_C > 0.0
                and cem.micropitting_safety >= 1.0
                and cem.material_temp_margin_C > 0.0
            )
            score = (
                min(cem.lambda_min, 3.0) / 3.0 * 0.35
                + min(max(cem.scuff_margin_C, 0.0) / 200.0, 1.0) * 0.30
                + min(cem.micropitting_safety, 5.0) / 5.0 * 0.25
                + (0.10 if cem.material_temp_margin_C > 0.0 else 0.0)
            )
            candidate["cem"] = {
                "lambda_min": float(cem.lambda_min),
                "scuff_margin_C": float(cem.scuff_margin_C),
                "micropitting_safety": float(cem.micropitting_safety),
                "material_temp_margin_C": float(cem.material_temp_margin_C),
                "feasible": feasible,
            }
            return feasible, float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            LOGGER.debug("CEM feasibility check failed: %s", exc)
            return False, 0.0

    def repair(self, candidate: dict[str, Any]) -> dict[str, Any]:
        repaired = dict(candidate)
        x = np.asarray(repaired.get("x", []), dtype=np.float64).reshape(-1)
        if x.size != N_TOTAL:
            x = np.zeros(N_TOTAL, dtype=np.float64)
        x = np.clip(x, self._xl, self._xu)

        # Conservative repair in real-world slice to improve CEM margin.
        rw0 = N_THERMO + N_GEAR
        x[rw0 + 0] = max(x[rw0 + 0], 0.50)  # finish
        x[rw0 + 1] = max(x[rw0 + 1], 0.60)  # lube mode
        x[rw0 + 2] = max(x[rw0 + 2], 0.50)  # material quality
        x[rw0 + 3] = max(x[rw0 + 3], 0.40)  # coating
        x[rw0 + 5] = max(x[rw0 + 5], 0.60)  # oil flow
        x[rw0 + 7] = max(x[rw0 + 7], 0.60)  # evacuation
        x = np.clip(x, self._xl, self._xu)

        repaired["x"] = x
        return repaired

    def update_distribution(self, history: list[dict[str, Any]]) -> None:  # noqa: ARG002
        # Current CEM path is rule-based and stateless for generation.
        return None


__all__ = ["CEMAdapter"]

