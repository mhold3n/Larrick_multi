"""Dataset registry — architecture for future experimental data ingestion.

Provides a registration and lookup system so that CEM modules can be
backed by real experimental/textbook datasets without code changes.

Datasets are described by ``DatasetDescriptor`` and managed by the
singleton ``DatasetRegistry``.  Placeholder datasets are auto-registered
on import with sensible defaults from the research documents.

Data format convention:
    - CSV or JSON files in ``data/cem/`` directory
    - Loaded on first access, cached in memory
    - Schema hash validates that the loader matches the expected columns
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Resolve data/cem/ relative to the package, not the CWD.
# Layout: src/larrak2/cem/registry.py → parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_CEM_ROOT = _REPO_ROOT / "data" / "cem"

# ---------------------------------------------------------------------------
# Dataset descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetDescriptor:
    """Metadata for a CEM dataset.

    Attributes:
        name: Unique identifier (e.g. "material_fatigue_life").
        domain: CEM domain this dataset belongs to
            (tribology / material / surface / lubrication / post_processing).
        version: Semantic version string.
        source_ref: Citation or source reference.
        path: Relative path to data file (from project root), or None
            if the dataset is entirely in-memory (placeholder).
        columns: Expected column names (for schema validation).
        schema_hash: Auto-computed hash of column names + version.
    """

    name: str
    domain: str
    version: str = "0.1.0-placeholder"
    source_ref: str = ""
    path: str | None = None
    columns: tuple[str, ...] = ()
    schema_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        # Compute schema hash
        data = json.dumps(
            {
                "name": self.name,
                "domain": self.domain,
                "version": self.version,
                "columns": list(self.columns),
            },
            sort_keys=True,
        )
        h = hashlib.sha256(data.encode()).hexdigest()
        object.__setattr__(self, "schema_hash", h[:16])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class DatasetRegistry:
    """Central registry for CEM datasets.

    Manages registration, lookup, and lazy loading of datasets.
    """

    def __init__(self) -> None:
        self._descriptors: dict[str, DatasetDescriptor] = {}
        self._cache: dict[str, Any] = {}

    def register(self, descriptor: DatasetDescriptor) -> None:
        """Register a dataset descriptor."""
        if descriptor.name in self._descriptors:
            existing = self._descriptors[descriptor.name]
            if existing.schema_hash != descriptor.schema_hash:
                logger.warning(
                    "Overwriting dataset '%s' (hash %s → %s)",
                    descriptor.name,
                    existing.schema_hash,
                    descriptor.schema_hash,
                )
        self._descriptors[descriptor.name] = descriptor
        # Invalidate cache on re-registration
        self._cache.pop(descriptor.name, None)

    def get(self, name: str) -> DatasetDescriptor:
        """Look up a dataset descriptor by name.

        Raises:
            KeyError: If dataset is not registered.
        """
        return self._descriptors[name]

    def list_available(self) -> list[str]:
        """Return names of all registered datasets."""
        return sorted(self._descriptors.keys())

    def list_by_domain(self, domain: str) -> list[DatasetDescriptor]:
        """Return all descriptors for a given domain."""
        return [d for d in self._descriptors.values() if d.domain == domain]

    def load_table(self, name: str) -> dict[str, list]:
        """Load a dataset table.

        If a file path is specified, loads from disk (CSV/JSON).
        Otherwise returns an empty placeholder structure.

        Returns:
            Dictionary mapping column names to lists of values.
        """
        if name in self._cache:
            return self._cache[name]

        desc = self.get(name)

        if desc.path is not None:
            p = Path(desc.path)
            if p.exists():
                table = self._load_file(p, desc)
            else:
                logger.warning("Dataset file not found: %s — returning empty placeholder", p)
                table = {col: [] for col in desc.columns}
        else:
            # Auto-locate from data/cem/{name}.csv or .json (package-relative)
            auto_found = False
            for ext in (".csv", ".json"):
                candidate_path = _DATA_CEM_ROOT / f"{desc.name}{ext}"
                if candidate_path.exists():
                    logger.info("Auto-located dataset: %s", candidate_path)
                    table = self._load_file(candidate_path, desc)
                    auto_found = True
                    break
            if not auto_found:
                table = {col: [] for col in desc.columns}

        self._cache[name] = table
        return table

    def load_required_table(
        self,
        name: str,
        *,
        validation_mode: str = "strict",
        key_columns: tuple[str, ...] = (),
    ) -> tuple[dict[str, list], list[str]]:
        """Load table with strict/warn/off validation semantics.

        Returns:
            (table, messages)
            messages contains degradation notes in warn/off modes.
        """
        mode = str(validation_mode).strip().lower()
        if mode not in {"strict", "warn", "off"}:
            raise ValueError(
                f"validation_mode must be one of ['strict', 'warn', 'off'], got {validation_mode!r}"
            )

        table = self.load_table(name)
        desc = self.get(name)
        messages: list[str] = []

        if not table:
            msg = f"Dataset '{name}' returned empty table object."
            if mode == "strict":
                raise ValueError(msg)
            messages.append(msg)
            logger.warning(msg)
            return table, messages

        # Determine row count from descriptor columns where possible.
        candidate_cols = [c for c in desc.columns if c in table]
        if not candidate_cols:
            candidate_cols = list(table.keys())
        n_rows = max((len(table.get(col, [])) for col in candidate_cols), default=0)

        if n_rows <= 0:
            msg = f"Dataset '{name}' has no rows."
            if mode == "strict":
                raise ValueError(msg)
            messages.append(msg)
            logger.warning(msg)
            return table, messages

        if key_columns:
            for key in key_columns:
                if key not in table:
                    msg = f"Dataset '{name}' is missing required key column '{key}'."
                    if mode == "strict":
                        raise ValueError(msg)
                    messages.append(msg)
                    logger.warning(msg)
                    continue
                vals = table.get(key, [])
                bad_rows = [
                    int(i)
                    for i, v in enumerate(vals)
                    if str(v).strip() == "" or str(v).strip().lower() in {"nan", "none"}
                ]
                if bad_rows:
                    msg = (
                        f"Dataset '{name}' has empty required keys in column '{key}' "
                        f"at rows {bad_rows[:5]}"
                    )
                    if mode == "strict":
                        raise ValueError(msg)
                    messages.append(msg)
                    logger.warning(msg)

        return table, messages

    @staticmethod
    def _load_file(path: Path, desc: DatasetDescriptor) -> dict[str, list]:
        """Load a CSV or JSON file into column-dict format.

        Validates that all columns declared in ``desc.columns`` are present
        in the loaded header.  Extra columns are allowed.

        Raises:
            ValueError: If required columns are missing from the file.
        """
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                # List of records → column dict
                if data:
                    cols = list(data[0].keys())
                    table = {c: [row.get(c) for row in data] for c in cols}
                else:
                    table = {col: [] for col in desc.columns}
            else:
                table = data  # Already column dict
        elif path.suffix == ".csv":
            import csv as _csv

            text = path.read_text().strip()
            if not text:
                return {col: [] for col in desc.columns}
            reader = _csv.reader(text.splitlines())
            header = [h.strip() for h in next(reader)]
            table = {h: [] for h in header}
            for row_num, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    logger.warning(
                        "Dataset '%s' row %d has %d values (expected %d) — skipping",
                        desc.name,
                        row_num,
                        len(row),
                        len(header),
                    )
                    continue
                for h, v in zip(header, row):
                    table[h].append(v.strip())
        else:
            logger.warning("Unsupported file format: %s", path.suffix)
            return {col: [] for col in desc.columns}

        # --- Schema validation ---
        if desc.columns:
            required = set(desc.columns)
            present = set(table.keys())
            missing = required - present
            if missing:
                raise ValueError(
                    f"Dataset '{desc.name}' loaded from {path} is missing "
                    f"required columns: {sorted(missing)}. "
                    f"Present columns: {sorted(present)}"
                )

        return table


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_REGISTRY: DatasetRegistry | None = None


def get_registry() -> DatasetRegistry:
    """Return the singleton DatasetRegistry, initializing placeholder datasets."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = DatasetRegistry()
        _register_placeholders(_REGISTRY)
    return _REGISTRY


# ---------------------------------------------------------------------------
# Placeholder dataset registration
# ---------------------------------------------------------------------------


def _register_placeholders(reg: DatasetRegistry) -> None:
    """Register placeholder dataset descriptors.

    These define the expected schema for each domain.  Actual data files
    are out of scope for this sprint — the architecture accepts them
    once sourced.
    """
    reg.register(
        DatasetDescriptor(
            name="material_properties",
            domain="material",
            source_ref="NASA Glenn + Carpenter + Pyrowear datasheets",
            columns=(
                "alloy",
                "max_service_temp_C",
                "case_hardness_HRC",
                "core_hardness_HRC",
                "fatigue_life_multiplier",
                "cost_tier",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="tribology_ehl_coefficients",
            domain="tribology",
            source_ref="ISO/TS 6336-22 + ISO 14635-1/2 mapped calibration",
            columns=(
                "oil_type",
                "finish_tier",
                "temp_C_min",
                "temp_C_max",
                "ehl_constant",
                "viscosity_speed_exp",
                "pressure_exp",
                "temp_ref_C",
                "temp_exp",
                "unit_system",
                "provenance",
                "version",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="scuffing_critical_temperatures",
            domain="tribology",
            source_ref="ISO/TS 6336-20/21 + ISO 14635-1/2 FZG test procedures",
            columns=(
                "oil_type",
                "additive_package",
                "method",
                "T_crit_C",
                "load_stage",
                "test_method",
                "unit_temp",
                "provenance",
                "version",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="micropitting_lambda_perm",
            domain="tribology",
            source_ref="ISO/TS 6336-22 micropitting permissible film thickness",
            columns=(
                "oil_type",
                "finish_tier",
                "lambda_perm",
                "unit_lambda",
                "provenance",
                "version",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="fzg_step_load_map",
            domain="tribology",
            source_ref="ISO 14635-1/2 FZG procedure mapping to scuff calibration",
            columns=(
                "test_standard",
                "test_method",
                "load_stage",
                "T_crit_C",
                "oil_type",
                "additive_package",
                "unit_temp",
                "provenance",
                "version",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="surface_finish_endurance",
            domain="surface",
            source_ref="REM/AGMA FZG micropitting + NASA scuffing TOF",
            columns=(
                "finish_method",
                "Ra_um",
                "Rz_um",
                "composite_roughness_factor",
                "micropitting_life_multiplier",
                "scuffing_TOF_multiplier",
                "cost_multiplier",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="lubrication_cooling_curves",
            domain="lubrication",
            source_ref="NASA oil-jet studies + API 677",
            columns=(
                "delivery_mode",
                "flow_rate_L_min",
                "pitch_vel_m_s",
                "tooth_temp_reduction_C",
                "churning_loss_fraction",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="coating_rcf_performance",
            domain="post_processing",
            source_ref="Oerlikon Balzers + Platit + Scientific Reports",
            columns=(
                "coating_type",
                "substrate",
                "hertz_stress_MPa",
                "cycles_to_failure",
                "friction_coeff",
                "temperature_C",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="heat_treat_hardness_curves",
            domain="post_processing",
            source_ref="Pyrowear/CBS-50 NiL/Ferrium datasheets",
            columns=(
                "alloy",
                "treatment",
                "temper_temp_C",
                "case_hardness_HRC",
                "core_hardness_HRC",
            ),
        )
    )

    # --- Phase 7: Screening & Galerkin datasets ---

    reg.register(
        DatasetDescriptor(
            name="route_metadata",
            domain="material",
            source_ref="Phase 8 Aggregated Literature/Datasheets",
            columns=(
                "route_id",
                "process_family",
                "case_hardness_hrc_nom",
                "core_toughness_KIC_MPa_m05",
                "max_service_temp_C",
                "cleanliness_grade_proxy",
                "quality_grade",
                "provenance",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="limit_stress_numbers",
            domain="material",
            source_ref="ISO 6336-5 / AGMA 2101 allowable stress numbers",
            columns=(
                "route_id",
                "material_group",
                "sigma_Hlim_MPa",
                "hardness_hrc_min",
                "hardness_hrc_max",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="material_route_cloud",
            domain="material",
            source_ref="ASM Alloy Center / Total Materia / SpringerMaterials",
            columns=(
                "route_id",
                "youngs_modulus_GPa",
                "poissons_ratio",
                "rho_kg_m3",
                "k_W_mK",
                "cp_J_kgK",
                "toughness_KIC_MPa_m05",
            ),
        )
    )

    reg.register(
        DatasetDescriptor(
            name="temperature_curves",
            domain="material",
            source_ref="JAHM MPDB / ASM Handbook temperature-dependent curves",
            columns=(
                "route_id",
                "property",
                "temp_c",
                "value",
            ),
        )
    )
