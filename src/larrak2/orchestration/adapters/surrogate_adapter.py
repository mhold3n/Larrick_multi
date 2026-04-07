"""HiFi surrogate adapter for orchestration predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from larrak_runtime.core.artifact_paths import (
    DEFAULT_HIFI_SURROGATE_DIR,
    assert_not_legacy_models_path,
)
from larrak_runtime.core.encoding import N_TOTAL, decode_candidate
from larrak_runtime.surrogate.quality_contract import validate_artifact_quality
from larrak_simulation.training.hifi_schema import NormalizationParams

from larrak2.surrogate.hifi.models import (
    FlowCoefficientSurrogate,
    StructuralSurrogate,
    ThermalSurrogate,
)

LOGGER = logging.getLogger(__name__)


class HifiSurrogateAdapter:
    """Predicts candidate merit with wave-1 HiFi surrogate assets when available."""

    def __init__(
        self,
        model_dir: str | Path = str(DEFAULT_HIFI_SURROGATE_DIR),
        *,
        default_rpm: float = 3000.0,
        default_torque: float = 200.0,
        allow_heuristic_fallback: bool = False,
        validation_mode: str = "strict",
        required_quality_artifacts: list[str | Path] | None = None,
    ) -> None:
        requested = Path(model_dir)
        self.model_dir = Path(
            assert_not_legacy_models_path(
                requested,
                purpose="HiFi surrogate model directory",
            )
        )
        self.default_rpm = float(default_rpm)
        self.default_torque = float(default_torque)
        self.allow_heuristic_fallback = bool(allow_heuristic_fallback)
        self.validation_mode = str(validation_mode)
        if self.validation_mode not in {"strict", "warn", "off"}:
            raise ValueError(
                "validation_mode must be one of {'strict', 'warn', 'off'}, "
                f"got {self.validation_mode!r}"
            )
        required = [
            "thermal_surrogate.pt",
            "structural_surrogate.pt",
            "flow_surrogate.pt",
            "normalization.json",
        ]
        if required_quality_artifacts:
            required.extend(str(v) for v in required_quality_artifacts)
        # Stable de-dup preserving order.
        self.required_quality_artifacts: list[str] = []
        seen: set[str] = set()
        for item in required:
            token = str(item).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            self.required_quality_artifacts.append(token)

        self.norm = NormalizationParams()
        self.thermal_model: ThermalSurrogate | None = None
        self.structural_model: StructuralSurrogate | None = None
        self.flow_model: FlowCoefficientSurrogate | None = None
        self._training_data: list[tuple[dict[str, Any], float]] = []
        self._load_assets()
        if not self.using_hifi_models and not self.allow_heuristic_fallback:
            raise RuntimeError(
                "HiFi surrogate assets are incomplete/unvalidated and heuristic fallback is disabled. "
                "Train artifacts under outputs/artifacts/surrogates/hifi and include quality_report.json."
            )

    def _load_assets(self) -> None:
        validate_artifact_quality(
            self.model_dir,
            surrogate_kind="hifi",
            validation_mode=self.validation_mode,
            required_artifacts=list(self.required_quality_artifacts),
        )
        norm_path = self.model_dir / "normalization.json"
        if norm_path.exists():
            try:
                self.norm = NormalizationParams.load(norm_path)
            except Exception as exc:
                LOGGER.warning("Failed loading HiFi normalization params: %s", exc)

        try:
            t_path = self.model_dir / "thermal_surrogate.pt"
            s_path = self.model_dir / "structural_surrogate.pt"
            f_path = self.model_dir / "flow_surrogate.pt"
            if t_path.exists():
                self.thermal_model = ThermalSurrogate.load(str(t_path))
                self.thermal_model.eval()
            if s_path.exists():
                self.structural_model = StructuralSurrogate.load(str(s_path))
                self.structural_model.eval()
            if f_path.exists():
                self.flow_model = FlowCoefficientSurrogate.load(str(f_path))
                self.flow_model.eval()
        except Exception as exc:
            LOGGER.warning("Failed loading one or more HiFi models: %s", exc)
            self.thermal_model = None
            self.structural_model = None
            self.flow_model = None

    @property
    def using_hifi_models(self) -> bool:
        return (
            self.thermal_model is not None
            and self.structural_model is not None
            and self.flow_model is not None
        )

    def _extract_features(self, candidate: dict[str, Any]) -> np.ndarray:
        x = np.asarray(candidate.get("x", []), dtype=np.float64).reshape(-1)
        if x.size != N_TOTAL:
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        decoded = decode_candidate(x)
        bore = float(candidate.get("bore_mm", 85.0))
        stroke = float(candidate.get("stroke_mm", 90.0))
        cr = float(candidate.get("compression_ratio", 12.0))
        rpm = float(candidate.get("rpm", self.default_rpm))
        load = float(candidate.get("load_fraction", self.default_torque / 400.0))
        load = float(np.clip(load, 0.0, 1.2))

        # Tie load slightly to lambda and valve timing controls.
        lam = float(decoded.thermo.lambda_af)
        load_adj = load * (1.0 - 0.15 * abs(lam - 1.0))
        load_adj = float(np.clip(load_adj, 0.1, 1.0))

        class _Record:
            def __init__(
                self, bore: float, stroke: float, cr: float, rpm: float, load: float
            ) -> None:
                self.bore = bore
                self.stroke = stroke
                self.cr = cr
                self.rpm = rpm
                self.load = load

        rec = _Record(bore=bore, stroke=stroke, cr=cr, rpm=rpm, load=load_adj)
        return self.norm.normalize_inputs(rec)  # type: ignore[arg-type]

    def _heuristic_predict(self, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        merits: list[float] = []
        uncertainty: list[float] = []
        for c in candidates:
            x = np.asarray(c.get("x", []), dtype=np.float64).reshape(-1)
            if x.size != N_TOTAL:
                merits.append(-1.0)
                uncertainty.append(1.0)
                continue
            dec = decode_candidate(x)
            lam = float(dec.thermo.lambda_af)
            gear_norm = float(np.linalg.norm(dec.gear.pitch_coeffs))
            rw_mean = float(
                np.mean(
                    [
                        dec.realworld.surface_finish_level,
                        dec.realworld.lube_mode_level,
                        dec.realworld.material_quality_level
                        if dec.realworld.material_quality_level is not None
                        else 0.5,
                        dec.realworld.coating_level,
                        dec.realworld.oil_flow_level,
                    ]
                )
            )
            score = (1.0 - abs(lam - 1.0)) + 0.35 * rw_mean - 0.08 * gear_norm
            unc = 0.08 + 0.22 * abs(lam - 1.0)
            merits.append(float(score))
            uncertainty.append(float(np.clip(unc, 0.01, 2.0)))
        return np.asarray(merits, dtype=np.float64), np.asarray(uncertainty, dtype=np.float64)

    def predict(self, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if not candidates:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        if not self.using_hifi_models:
            if self.allow_heuristic_fallback:
                return self._heuristic_predict(candidates)
            raise RuntimeError(
                "HiFi surrogate models are unavailable in strict production mode. "
                "Set explicit non-production override allow_heuristic_fallback=True if needed."
            )

        features = np.asarray([self._extract_features(c) for c in candidates], dtype=np.float32)
        thermal_mean, thermal_std = self.thermal_model.predict(features)  # type: ignore[union-attr]
        structural_mean, structural_std = self.structural_model.predict(features)  # type: ignore[union-attr]
        flow_mean, flow_std = self.flow_model.predict(features)  # type: ignore[union-attr]

        t_m = np.asarray(thermal_mean, dtype=np.float64).reshape(-1)
        t_s = np.asarray(thermal_std, dtype=np.float64).reshape(-1)
        s_m = np.asarray(structural_mean, dtype=np.float64).reshape(-1)
        s_s = np.asarray(structural_std, dtype=np.float64).reshape(-1)
        f_m = np.asarray(flow_mean, dtype=np.float64).reshape(-1)
        f_s = np.asarray(flow_std, dtype=np.float64).reshape(-1)

        # Scalar merit used by budget selector: higher is better.
        merit = f_m - 0.002 * t_m - 0.001 * s_m
        uncertainty = f_s + 0.002 * t_s + 0.001 * s_s
        uncertainty = np.clip(uncertainty, 1e-6, None)
        return merit.astype(np.float64), uncertainty.astype(np.float64)

    def update(self, data: list[tuple[dict[str, Any], float]]) -> None:
        self._training_data.extend(data)

    def get_training_data(self) -> list[tuple[dict[str, Any], float]]:
        return list(self._training_data)


__all__ = ["HifiSurrogateAdapter"]
