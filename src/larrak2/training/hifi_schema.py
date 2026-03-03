"""Training data schema for HiFi surrogate models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainingRecord:
    """One HiFi training sample."""

    case_id: int
    run_id: str = ""

    bore: float = 0.0
    stroke: float = 0.0
    cr: float = 0.0
    rpm: float = 0.0
    load: float = 0.0

    T_crown_max: float | None = None
    T_liner_max: float | None = None
    htc_mean: float | None = None

    von_mises_max: float | None = None
    displacement_max: float | None = None
    safety_factor: float | None = None

    cd_effective: float | None = None
    swirl_ratio: float | None = None
    tumble_ratio: float | None = None

    p_max: float | None = None
    imep: float | None = None
    heat_release_rate_max: float | None = None

    solver_success: bool = True
    computation_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRecord:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class NormalizationParams:
    """Min/max normalization settings."""

    bore_range: tuple[float, float] = (70.0, 100.0)
    stroke_range: tuple[float, float] = (75.0, 110.0)
    cr_range: tuple[float, float] = (10.0, 16.0)
    rpm_range: tuple[float, float] = (1000.0, 7000.0)
    load_range: tuple[float, float] = (0.2, 1.0)

    T_crown_range: tuple[float, float] = (350.0, 700.0)
    T_liner_range: tuple[float, float] = (300.0, 550.0)
    von_mises_range: tuple[float, float] = (0.0, 400.0)
    cd_range: tuple[float, float] = (0.3, 0.8)
    p_max_range: tuple[float, float] = (40.0, 150.0)

    @staticmethod
    def _norm(val: float, lo: float, hi: float) -> float:
        return (val - lo) / (hi - lo) if hi > lo else 0.5

    def normalize_inputs(self, record: TrainingRecord) -> np.ndarray:
        return np.array(
            [
                self._norm(record.bore, *self.bore_range),
                self._norm(record.stroke, *self.stroke_range),
                self._norm(record.cr, *self.cr_range),
                self._norm(record.rpm, *self.rpm_range),
                self._norm(record.load, *self.load_range),
            ],
            dtype=np.float32,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NormalizationParams:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class TrainingDataset:
    """Container for HiFi training records."""

    def __init__(
        self,
        records: list[TrainingRecord],
        norm_params: NormalizationParams | None = None,
    ) -> None:
        self.records = records
        self.norm_params = norm_params or NormalizationParams()

    @classmethod
    def from_parquet(cls, path: str | Path) -> TrainingDataset:
        import pandas as pd

        df = pd.read_parquet(path)
        records: list[TrainingRecord] = []
        for _, row in df.iterrows():
            records.append(
                TrainingRecord(
                    case_id=int(row.get("case_id", 0)),
                    bore=float(row.get("param_bore_mm", 85.0)),
                    stroke=float(row.get("param_stroke_mm", 90.0)),
                    cr=float(row.get("param_compression_ratio", 12.0)),
                    rpm=float(row.get("param_rpm", 3000.0)),
                    load=float(row.get("param_load_fraction", 1.0)),
                    T_crown_max=row.get("output_T_crown_max"),
                    von_mises_max=row.get("output_von_mises_max"),
                    cd_effective=row.get("output_cd_effective"),
                    p_max=row.get("output_p_max"),
                    solver_success=bool(row.get("success", True)),
                )
            )
        return cls(records)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingDataset:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records: list[TrainingRecord] = []
        for item in data:
            params = item.get("params", {})
            outputs = item.get("outputs", {})
            records.append(
                TrainingRecord(
                    case_id=int(item.get("case_id", 0)),
                    bore=float(params.get("bore_mm", 85.0)),
                    stroke=float(params.get("stroke_mm", 90.0)),
                    cr=float(params.get("compression_ratio", 12.0)),
                    rpm=float(params.get("rpm", 3000.0)),
                    load=float(params.get("load_fraction", 1.0)),
                    T_crown_max=outputs.get("T_crown_max"),
                    von_mises_max=outputs.get("von_mises_max"),
                    cd_effective=outputs.get("cd_effective"),
                    p_max=outputs.get("p_max"),
                    solver_success=bool(item.get("success", True)),
                )
            )
        return cls(records)

    def split(
        self,
        train_frac: float = 0.8,
        seed: int = 42,
    ) -> tuple[TrainingDataset, TrainingDataset]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.records))
        split = int(len(idx) * train_frac)
        train = [self.records[i] for i in idx[:split]]
        val = [self.records[i] for i in idx[split:]]
        return TrainingDataset(train, self.norm_params), TrainingDataset(val, self.norm_params)

    def _build_xy(self, attr: str, lo_hi: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = lo_hi
        X: list[np.ndarray] = []
        y: list[float] = []
        for rec in self.records:
            value = getattr(rec, attr)
            if value is None or not rec.solver_success:
                continue
            X.append(self.norm_params.normalize_inputs(rec))
            y.append((float(value) - lo) / (hi - lo))

        if not X:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1)

    def get_thermal_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self._build_xy("T_crown_max", self.norm_params.T_crown_range)

    def get_structural_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self._build_xy("von_mises_max", self.norm_params.von_mises_range)

    def get_flow_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self._build_xy("cd_effective", self.norm_params.cd_range)
