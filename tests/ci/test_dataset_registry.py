"""Tests for Phase 7 dataset registry, property curves, and life damage integration."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from larrak2.cem.registry import DatasetRegistry, DatasetDescriptor, get_registry
from larrak2.cem.property_curves import (
    get_property_at_temp,
    invalidate_cache as invalidate_curves_cache,
)
from larrak2.realworld.life_damage import (
    get_sigma_ref_for_route,
    invalidate_limit_stress_cache,
    _SIGMA_REF_MPA,
)
from larrak2.cem.material_db import MaterialClass


# ---------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------


class TestSchemaValidation:
    """Registry _load_file should raise on missing required columns."""

    def test_missing_column_raises(self):
        reg = DatasetRegistry()
        desc = DatasetDescriptor(
            name="test_bad",
            domain="test",
            columns=("col_a", "col_b", "col_c"),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col_a,col_b\n1,2\n")
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                reg._load_file(Path(tmp), desc)
        finally:
            os.unlink(tmp)

    def test_extra_columns_allowed(self):
        reg = DatasetRegistry()
        desc = DatasetDescriptor(
            name="test_ok",
            domain="test",
            columns=("col_a",),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col_a,col_b\n1,2\n")
            tmp = f.name
        try:
            table = reg._load_file(Path(tmp), desc)
            assert "col_a" in table
            assert "col_b" in table  # extra allowed
        finally:
            os.unlink(tmp)

    def test_malformed_row_skipped(self):
        """Rows with wrong column count should be skipped, not crash."""
        reg = DatasetRegistry()
        desc = DatasetDescriptor(
            name="test_rows",
            domain="test",
            columns=("a", "b"),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,2\n3\n4,5\n")
            tmp = f.name
        try:
            table = reg._load_file(Path(tmp), desc)
            assert len(table["a"]) == 2  # row "3" skipped
            assert table["a"] == ["1", "4"]
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------


class TestDatasetLoading:
    """New datasets load from data/cem/ with correct shapes."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        """Force fresh registry for each test."""
        import larrak2.cem.registry as _reg

        _reg._REGISTRY = None
        yield
        _reg._REGISTRY = None

    def test_limit_stress_numbers_loads(self):
        table = get_registry().load_table("limit_stress_numbers")
        assert len(table["route_id"]) >= 5
        assert "AISI_9310" in table["route_id"]

    def test_material_route_cloud_loads(self):
        table = get_registry().load_table("material_route_cloud")
        assert len(table["route_id"]) >= 1
        assert "youngs_modulus_GPa" in table

    def test_temperature_curves_loads(self):
        table = get_registry().load_table("temperature_curves")
        assert len(table["route_id"]) >= 4  # at least 4 props


# ---------------------------------------------------------------
# Property curves (interpolation, extrapolation, diffusivity)
# ---------------------------------------------------------------


class TestPropertyCurves:
    """get_property_at_temp interpolation and edge cases."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        import larrak2.cem.registry as _reg

        _reg._REGISTRY = None
        invalidate_curves_cache()
        yield
        invalidate_curves_cache()

    def test_interpolation_at_boundary(self):
        val = get_property_at_temp("AISI_9310", "youngs_modulus_GPa", 25.0)
        assert val == pytest.approx(205.0, abs=0.1)

    def test_interpolation_midpoint(self):
        val = get_property_at_temp("AISI_9310", "youngs_modulus_GPa", 87.5)
        assert val == pytest.approx(202.5, abs=0.1)

    def test_extrapolation_raises(self):
        with pytest.raises(ValueError, match="outside the data range"):
            get_property_at_temp("AISI_9310", "youngs_modulus_GPa", -50.0)

    def test_missing_route_raises(self):
        with pytest.raises(ValueError, match="No temperature curve found"):
            get_property_at_temp("NONEXISTENT", "youngs_modulus_GPa", 25.0)

    def test_missing_property_raises(self):
        with pytest.raises(ValueError, match="No temperature curve found"):
            get_property_at_temp("AISI_9310", "nonexistent_prop", 25.0)

    def test_diffusivity_derivation(self):
        alpha = get_property_at_temp("AISI_9310", "diffusivity_m2_s", 25.0)
        expected = 44.5 / (7850 * 475)
        assert alpha == pytest.approx(expected, rel=1e-6)

    def test_diffusivity_missing_constituent_raises(self):
        """If a constituent property is missing, diffusivity should fail."""
        with pytest.raises(ValueError):
            get_property_at_temp("NONEXISTENT", "diffusivity_m2_s", 25.0)


# ---------------------------------------------------------------
# Sigma ref calibration
# ---------------------------------------------------------------


class TestSigmaRefCalibration:
    """get_sigma_ref_for_route calibration-preserving scaling."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        import larrak2.cem.registry as _reg

        _reg._REGISTRY = None
        invalidate_limit_stress_cache()
        yield
        invalidate_limit_stress_cache()

    def test_baseline_identity(self):
        """AISI_9310 should return exactly _SIGMA_REF_MPA."""
        ref = get_sigma_ref_for_route("AISI_9310")
        assert ref == pytest.approx(_SIGMA_REF_MPA)

    def test_higher_alloy_scales_up(self):
        """M50NiL has sigma_Hlim=2000, baseline=1500, so ref = 2000."""
        ref = get_sigma_ref_for_route("M50NiL")
        expected = _SIGMA_REF_MPA * (2000.0 / 1500.0)
        assert ref == pytest.approx(expected, rel=1e-6)

    def test_all_material_classes_present(self):
        """Every MaterialClass.value must have a row in limit_stress_numbers."""
        for mc in MaterialClass:
            ref = get_sigma_ref_for_route(mc.value)
            assert ref > 0


# ---------------------------------------------------------------
# Integration: data path exists
# ---------------------------------------------------------------


class TestDataPath:
    """Verify dataset root resolution is correct."""

    def test_data_cem_root_exists(self):
        from larrak2.cem.registry import _DATA_CEM_ROOT

        assert _DATA_CEM_ROOT.exists(), f"_DATA_CEM_ROOT does not exist: {_DATA_CEM_ROOT}"

    def test_limit_stress_csv_exists(self):
        from larrak2.cem.registry import _DATA_CEM_ROOT

        assert (_DATA_CEM_ROOT / "limit_stress_numbers.csv").exists()

    def test_temperature_curves_csv_exists(self):
        from larrak2.cem.registry import _DATA_CEM_ROOT

        assert (_DATA_CEM_ROOT / "temperature_curves.csv").exists()
