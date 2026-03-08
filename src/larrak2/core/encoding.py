"""Candidate encoding and decoding.

This module defines the parameter structure and provides pack/unpack
functions for the flat decision vector x.

Layout (ENCODING_VERSION = "0.6"):
    x[0:10]  - ThermoParams (10 floats)
    x[10:19] - GearParams (9 floats: base_radius + 7 pitch_coeffs + face_width)
    x[19:27] - RealWorldParams (8 floats: surface_finish, lube, material, coating,
    hunting, oil_flow, oil_supply_temp, evacuation)
    Total: 27 decision variables

Legacy layout (ENCODING_VERSION = "0.4"):
    x[0:5]   - ThermoParams (5 floats, no valve timing)
    x[5:14]  - GearParams
    x[14:22] - RealWorldParams
    Total: 22 decision variables

Previous layout (ENCODING_VERSION = "0.5"):
    x[0:9]   - ThermoParams (9 floats, no spark timing)
    x[9:18]  - GearParams
    x[18:26] - RealWorldParams
    Total: 26 decision variables
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..thermo.chemistry_profile import (
    SPARK_TIMING_VAR_NAME,
    legacy_spark_timing_default,
    spark_timing_bounds,
)
from ..thermo.timing_profile import THERMO_TIMING_VAR_NAMES, legacy_timing_defaults, thermo_timing_bounds

LEGACY_ENCODING_VERSION = "0.4"
PRECHEM_ENCODING_VERSION = "0.5"
ENCODING_VERSION = "0.6"

# Decision variable count
LEGACY_N_THERMO = 5
PRECHEM_N_THERMO = 9
N_THERMO = 10
N_GEAR = 9  # base_radius(1) + pitch_coeffs(7) + face_width(1)
N_REALWORLD = (
    8  # surface_finish, lube_mode, material, coating, hunting, oil_flow, oil_temp, evacuation
)
LEGACY_N_TOTAL = LEGACY_N_THERMO + N_GEAR + N_REALWORLD  # 22
PRECHEM_N_TOTAL = PRECHEM_N_THERMO + N_GEAR + N_REALWORLD  # 26
N_TOTAL = N_THERMO + N_GEAR + N_REALWORLD  # 27

THERMO_VAR_NAMES = (
    "compression_duration",
    "expansion_duration",
    "heat_release_center",
    "heat_release_width",
    "lambda_af",
    *THERMO_TIMING_VAR_NAMES,
    SPARK_TIMING_VAR_NAME,
)

GEAR_VAR_NAMES = (
    "base_radius",
    "pitch_coeff_0",
    "pitch_coeff_1",
    "pitch_coeff_2",
    "pitch_coeff_3",
    "pitch_coeff_4",
    "pitch_coeff_5",
    "pitch_coeff_6",
    "face_width_mm",
)

REALWORLD_VAR_NAMES = (
    "surface_finish_level",
    "lube_mode_level",
    "material_quality_level",
    "coating_level",
    "hunting_level",
    "oil_flow_level",
    "oil_supply_temp_level",
    "evacuation_level",
)


@dataclass(frozen=True)
class VariableMetadata:
    """Metadata for one encoded decision variable."""

    index: int
    name: str
    group: str
    lower: float
    upper: float


@dataclass
class ThermoParams:
    """Thermodynamic control parameters.

    Attributes:
        compression_duration: Compression phase duration (degrees), [30, 90].
        expansion_duration: Expansion phase duration (degrees), [60, 120].
        heat_release_center: Center of heat release (degrees), [0, 30].
        heat_release_width: Heat release duration (degrees), [10, 60].
        lambda_af: Air-fuel equivalence ratio λ [-], where λ < 1 is rich and λ > 1 is lean.
    """

    compression_duration: float
    expansion_duration: float
    heat_release_center: float
    heat_release_width: float
    lambda_af: float
    intake_open_offset_from_bdc: float
    intake_duration_deg: float
    exhaust_open_offset_from_expansion_tdc: float
    exhaust_duration_deg: float
    spark_timing_deg_from_compression_tdc: float = -8.0
    timing_profile_id: str = "thermo_valve_timing_profile_v1"
    timing_source: str = "encoded_candidate"
    timing_legacy_injected: bool = False

    def to_array(self) -> np.ndarray:
        """Convert to flat array."""
        return np.array(
            [
                self.compression_duration,
                self.expansion_duration,
                self.heat_release_center,
                self.heat_release_width,
                self.lambda_af,
                self.intake_open_offset_from_bdc,
                self.intake_duration_deg,
                self.exhaust_open_offset_from_expansion_tdc,
                self.exhaust_duration_deg,
                self.spark_timing_deg_from_compression_tdc,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> ThermoParams:
        """Create from flat array."""
        flat = np.asarray(arr, dtype=np.float64).reshape(-1)
        if flat.size == LEGACY_N_THERMO:
            defaults = legacy_timing_defaults()
            spark_default = np.array([legacy_spark_timing_default()], dtype=np.float64)
            flat = np.concatenate([flat, defaults, spark_default], dtype=np.float64)
            return cls(
                compression_duration=float(flat[0]),
                expansion_duration=float(flat[1]),
                heat_release_center=float(flat[2]),
                heat_release_width=float(flat[3]),
                lambda_af=float(flat[4]),
                intake_open_offset_from_bdc=float(flat[5]),
                intake_duration_deg=float(flat[6]),
                exhaust_open_offset_from_expansion_tdc=float(flat[7]),
                exhaust_duration_deg=float(flat[8]),
                spark_timing_deg_from_compression_tdc=float(flat[9]),
                timing_source="legacy_default_profile",
                timing_legacy_injected=True,
            )
        if flat.size == PRECHEM_N_THERMO:
            spark_default = float(legacy_spark_timing_default())
            return cls(
                compression_duration=float(flat[0]),
                expansion_duration=float(flat[1]),
                heat_release_center=float(flat[2]),
                heat_release_width=float(flat[3]),
                lambda_af=float(flat[4]),
                intake_open_offset_from_bdc=float(flat[5]),
                intake_duration_deg=float(flat[6]),
                exhaust_open_offset_from_expansion_tdc=float(flat[7]),
                exhaust_duration_deg=float(flat[8]),
                spark_timing_deg_from_compression_tdc=spark_default,
                timing_source="chemistry_default_profile",
                timing_legacy_injected=True,
            )
        if flat.size != N_THERMO:
            raise ValueError(
                "ThermoParams expects "
                f"{LEGACY_N_THERMO}, {PRECHEM_N_THERMO}, or {N_THERMO} values, got {flat.size}"
            )
        return cls(
            compression_duration=float(flat[0]),
            expansion_duration=float(flat[1]),
            heat_release_center=float(flat[2]),
            heat_release_width=float(flat[3]),
            lambda_af=float(flat[4]),
            intake_open_offset_from_bdc=float(flat[5]),
            intake_duration_deg=float(flat[6]),
            exhaust_open_offset_from_expansion_tdc=float(flat[7]),
            exhaust_duration_deg=float(flat[8]),
            spark_timing_deg_from_compression_tdc=float(flat[9]),
        )


@dataclass
class GearParams:
    """Gear synthesis control parameters.

    Attributes:
        base_radius: Base circle radius (mm), [20, 60].
        pitch_coeffs: Fourier coefficients for pitch curve (7 floats).
        face_width_mm: Gear face width (mm), [4, 14]. Upper bound = gear body z-height.
    """

    base_radius: float
    pitch_coeffs: np.ndarray  # 7 coefficients
    face_width_mm: float  # Constrained by gear body z-height

    def to_array(self) -> np.ndarray:
        """Convert to flat array."""
        return np.concatenate([[self.base_radius], self.pitch_coeffs, [self.face_width_mm]])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> GearParams:
        """Create from flat array."""
        return cls(
            base_radius=float(arr[0]),
            pitch_coeffs=np.array(arr[1:8], dtype=np.float64),
            face_width_mm=float(arr[8]),
        )


@dataclass
class RealWorldParams:
    """Real-world engineering decision parameters.

    Continuous 0–1 levels that map to CEM enum tiers during evaluation.

    Attributes:
        surface_finish_level: 0→as-ground, 0.5→fine-ground, 1→superfinished.
        lube_mode_level: 0→dry, 0.33→splash, 0.67→jet, 1→phase-gated jet.
        material_quality_level: 0→9310, 0.5→CBS-50NiL, 1→Ferrium C64.
        coating_level: 0→none, 0.5→ta-C, 1→W-DLC/CrN duplex.
        hunting_level: 0→1-set, 0.5→4-set, 1→8-set pseudo-hunting.
        oil_flow_level: 0→0.5 L/min, 1→10 L/min.
        oil_supply_temp_level: 0→40°C, 1→120°C.
        evacuation_level: 0→passive drain, 1→active scavenge.
        material_state: Optional 4D properties [case_HRC, core_KIC, temp_C, clean].
    """

    surface_finish_level: float
    lube_mode_level: float
    material_quality_level: float | None = None
    coating_level: float = 0.0
    hunting_level: float = 0.0
    oil_flow_level: float = 0.5
    oil_supply_temp_level: float = 0.5
    evacuation_level: float = 0.5
    material_state: np.ndarray | None = None

    def to_array(self) -> np.ndarray:
        """Convert to flat array.

        If material_state is present, prepends the -999.0 version sentinel
        and packs the 4D state vector (length 12). Otherwise falls back to
        legacy 1D format (length 8).
        """
        if self.material_state is not None:
            return np.array(
                [
                    -999.0,  # Version Sentinel
                    *self.material_state,  # 4 floats
                    self.surface_finish_level,
                    self.lube_mode_level,
                    self.coating_level,
                    self.hunting_level,
                    self.oil_flow_level,
                    self.oil_supply_temp_level,
                    self.evacuation_level,
                ],
                dtype=np.float64,
            )

        # Legacy fallback
        mat_lvl = self.material_quality_level if self.material_quality_level is not None else 0.5
        return np.array(
            [
                self.surface_finish_level,
                self.lube_mode_level,
                mat_lvl,
                self.coating_level,
                self.hunting_level,
                self.oil_flow_level,
                self.oil_supply_temp_level,
                self.evacuation_level,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> RealWorldParams:
        """Create from flat array, handling encoding versions."""
        # Version 2.0 Check (-999.0 sentinel)
        if len(arr) >= 12 and np.isclose(arr[0], -999.0):
            return cls(
                material_state=np.array(arr[1:5], dtype=np.float64),
                surface_finish_level=float(arr[5]),
                lube_mode_level=float(arr[6]),
                coating_level=float(arr[7]),
                hunting_level=float(arr[8]),
                oil_flow_level=float(arr[9]),
                oil_supply_temp_level=float(arr[10]),
                evacuation_level=float(arr[11]),
            )

        # Version 1.0 Legacy Check
        return cls(
            surface_finish_level=float(arr[0]),
            lube_mode_level=float(arr[1]),
            material_quality_level=float(arr[2]),
            coating_level=float(arr[3]),
            hunting_level=float(arr[4]),
            oil_flow_level=float(arr[5]),
            oil_supply_temp_level=float(arr[6]),
            evacuation_level=float(arr[7]),
        )


@dataclass
class Candidate:
    """Complete candidate solution.

    Combines thermo, gear, and real-world parameters into a single
    structured object decoded from the flat decision vector.
    """

    thermo: ThermoParams
    gear: GearParams
    realworld: RealWorldParams


def legacy_index_to_current(idx: int) -> int:
    """Translate a legacy 22D index into the current 27D layout."""
    i = int(idx)
    if i < 0 or i >= LEGACY_N_TOTAL:
        raise IndexError(f"Legacy index out of range: {idx}")
    if i < LEGACY_N_THERMO:
        return i
    return i + (N_THERMO - LEGACY_N_THERMO)


def prechem_index_to_current(idx: int) -> int:
    """Translate a 26D pre-chemistry index into the current 27D layout."""
    i = int(idx)
    if i < 0 or i >= PRECHEM_N_TOTAL:
        raise IndexError(f"Pre-chemistry index out of range: {idx}")
    if i < PRECHEM_N_THERMO:
        return i
    return i + (N_THERMO - PRECHEM_N_THERMO)


def resolve_index_for_encoding(idx: int, encoding_version: str | None) -> int:
    """Resolve an artifact/archive index into the current canonical layout."""
    version = str(encoding_version or "").strip() or LEGACY_ENCODING_VERSION
    if version == ENCODING_VERSION:
        return int(idx)
    if version == PRECHEM_ENCODING_VERSION:
        return prechem_index_to_current(idx)
    if version == LEGACY_ENCODING_VERSION or version == "0.0":
        return legacy_index_to_current(idx)
    raise ValueError(
        f"Unsupported encoding_version '{encoding_version}' for feature index resolution"
    )


def upgrade_legacy_candidate_vector(x_legacy: np.ndarray) -> np.ndarray:
    """Inject default thermo chemistry controls into a legacy 22D decision vector."""
    arr = np.asarray(x_legacy, dtype=np.float64).reshape(-1)
    if arr.size != LEGACY_N_TOTAL:
        raise ValueError(f"Expected legacy vector length {LEGACY_N_TOTAL}, got {arr.size}")
    thermo = ThermoParams.from_array(arr[:LEGACY_N_THERMO])
    gear = GearParams.from_array(arr[LEGACY_N_THERMO : LEGACY_N_THERMO + N_GEAR])
    realworld = RealWorldParams.from_array(arr[LEGACY_N_THERMO + N_GEAR : LEGACY_N_TOTAL])
    return encode_candidate(Candidate(thermo=thermo, gear=gear, realworld=realworld))


def upgrade_legacy_candidate_matrix(X_legacy: np.ndarray) -> np.ndarray:
    """Inject default thermo chemistry controls into each row of a legacy 22D candidate matrix."""
    arr = np.asarray(X_legacy, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D legacy candidate matrix, got shape={arr.shape}")
    if arr.shape[1] != LEGACY_N_TOTAL:
        raise ValueError(
            f"Expected legacy candidate width {LEGACY_N_TOTAL}, got {arr.shape[1]}"
        )
    return np.vstack([upgrade_legacy_candidate_vector(row) for row in arr])


def upgrade_prechem_candidate_vector(x_prechem: np.ndarray) -> np.ndarray:
    """Inject default spark timing into a 26D pre-chemistry decision vector."""
    arr = np.asarray(x_prechem, dtype=np.float64).reshape(-1)
    if arr.size != PRECHEM_N_TOTAL:
        raise ValueError(f"Expected pre-chemistry vector length {PRECHEM_N_TOTAL}, got {arr.size}")
    thermo = ThermoParams.from_array(arr[:PRECHEM_N_THERMO])
    gear = GearParams.from_array(arr[PRECHEM_N_THERMO : PRECHEM_N_THERMO + N_GEAR])
    realworld = RealWorldParams.from_array(arr[PRECHEM_N_THERMO + N_GEAR : PRECHEM_N_TOTAL])
    return encode_candidate(Candidate(thermo=thermo, gear=gear, realworld=realworld))


def upgrade_prechem_candidate_matrix(X_prechem: np.ndarray) -> np.ndarray:
    """Inject default spark timing into each row of a 26D pre-chemistry candidate matrix."""
    arr = np.asarray(X_prechem, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D pre-chemistry candidate matrix, got shape={arr.shape}")
    if arr.shape[1] != PRECHEM_N_TOTAL:
        raise ValueError(
            f"Expected pre-chemistry candidate width {PRECHEM_N_TOTAL}, got {arr.shape[1]}"
        )
    return np.vstack([upgrade_prechem_candidate_vector(row) for row in arr])


def decode_candidate(x: np.ndarray) -> Candidate:
    """Decode flat array to structured Candidate.

    Args:
        x: Flat decision vector of length 22 (legacy), 26 (pre-chemistry), or 27 (current).

    Returns:
        Candidate with ThermoParams, GearParams, and RealWorldParams.
    """
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == LEGACY_N_TOTAL:
        thermo = ThermoParams.from_array(arr[:LEGACY_N_THERMO])
        gear_start = LEGACY_N_THERMO
        gear = GearParams.from_array(arr[gear_start : gear_start + N_GEAR])
        realworld = RealWorldParams.from_array(arr[gear_start + N_GEAR : LEGACY_N_TOTAL])
        return Candidate(thermo=thermo, gear=gear, realworld=realworld)
    if arr.size == PRECHEM_N_TOTAL:
        thermo = ThermoParams.from_array(arr[:PRECHEM_N_THERMO])
        gear_start = PRECHEM_N_THERMO
        gear = GearParams.from_array(arr[gear_start : gear_start + N_GEAR])
        realworld = RealWorldParams.from_array(arr[gear_start + N_GEAR : PRECHEM_N_TOTAL])
        return Candidate(thermo=thermo, gear=gear, realworld=realworld)
    if arr.size != N_TOTAL:
        raise ValueError(
            f"Expected {LEGACY_N_TOTAL} (legacy), {PRECHEM_N_TOTAL} (pre-chemistry), "
            f"or {N_TOTAL} (current) variables, got {arr.size}"
        )

    thermo = ThermoParams.from_array(arr[:N_THERMO])
    gear = GearParams.from_array(arr[N_THERMO : N_THERMO + N_GEAR])
    realworld = RealWorldParams.from_array(arr[N_THERMO + N_GEAR : N_THERMO + N_GEAR + N_REALWORLD])

    return Candidate(thermo=thermo, gear=gear, realworld=realworld)


def encode_candidate(candidate: Candidate) -> np.ndarray:
    """Encode structured Candidate to flat array.

    Args:
        candidate: Structured candidate solution.

    Returns:
        Flat decision vector of length N_TOTAL (27).
    """
    return np.concatenate(
        [
            candidate.thermo.to_array(),
            candidate.gear.to_array(),
            candidate.realworld.to_array(),
        ],
        dtype=np.float64,
    )


def bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds for decision variables.

    Returns:
        (xl, xu) tuple of bound arrays, each of length N_TOTAL (27).
    """
    # ThermoParams bounds
    # compression, expansion, hr_center, hr_width, lambda_af, valve timing, spark timing
    timing_lb, timing_ub = thermo_timing_bounds()
    spark_lb, spark_ub = spark_timing_bounds()
    thermo_lb = np.concatenate(
        [np.array([30.0, 60.0, 0.0, 10.0, 0.6]), timing_lb, np.array([spark_lb])],
        dtype=np.float64,
    )
    thermo_ub = np.concatenate(
        [np.array([90.0, 120.0, 30.0, 60.0, 1.6]), timing_ub, np.array([spark_ub])],
        dtype=np.float64,
    )

    # GearParams bounds: base_radius + 7 pitch coefficients + face_width
    gear_lb = np.array([20.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 4.0])
    gear_ub = np.array([60.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 14.0])

    # RealWorldParams bounds: 8 continuous levels [0, 1]
    # [finish, lube_mode, material, coating, hunting, oil_flow, oil_temp, evacuation]
    rw_lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rw_ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    xl = np.concatenate([thermo_lb, gear_lb, rw_lb])
    xu = np.concatenate([thermo_ub, gear_ub, rw_ub])

    return xl, xu


def variable_manifest() -> list[VariableMetadata]:
    """Return ordered metadata for all encoded decision variables."""
    xl, xu = bounds()
    names = THERMO_VAR_NAMES + GEAR_VAR_NAMES + REALWORLD_VAR_NAMES
    groups = (
        ["thermo"] * len(THERMO_VAR_NAMES)
        + ["gear"] * len(GEAR_VAR_NAMES)
        + ["realworld"] * len(REALWORLD_VAR_NAMES)
    )

    return [
        VariableMetadata(
            index=i,
            name=str(names[i]),
            group=str(groups[i]),
            lower=float(xl[i]),
            upper=float(xu[i]),
        )
        for i in range(N_TOTAL)
    ]


def group_indices() -> dict[str, list[int]]:
    """Return variable indices grouped by subsystem."""
    out: dict[str, list[int]] = {}
    for info in variable_manifest():
        out.setdefault(info.group, []).append(info.index)
    return out


def random_candidate(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random candidate within bounds.

    Args:
        rng: Random number generator (uses default if None).

    Returns:
        Random decision vector within bounds (length 27).
    """
    if rng is None:
        rng = np.random.default_rng()

    xl, xu = bounds()
    return rng.uniform(xl, xu)


def mid_bounds_candidate() -> np.ndarray:
    """Return candidate at midpoint of bounds (likely feasible for toy physics)."""
    xl, xu = bounds()
    return (xl + xu) / 2
