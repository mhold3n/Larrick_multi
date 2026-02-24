"""Tests for real-world surrogate system (optimization-loop side)."""

import numpy as np

from larrak2.realworld.constraints import (
    REALWORLD_CONSTRAINT_NAMES,
    compute_realworld_constraints,
)
from larrak2.realworld.surrogates import (
    DEFAULT_REALWORLD_PARAMS,
    RealWorldSurrogateParams,
    evaluate_realworld_surrogates,
)


class TestSurrogateParams:
    """Test RealWorldSurrogateParams creation and defaults."""

    def test_defaults_valid(self):
        """Default params should be within [0, 1]."""
        p = DEFAULT_REALWORLD_PARAMS
        assert 0.0 <= p.surface_finish_level <= 1.0
        assert 0.0 <= p.lube_mode_level <= 1.0
        if p.material_quality_level is not None:
            assert 0.0 <= p.material_quality_level <= 1.0
        assert 0.0 <= p.coating_level <= 1.0

    def test_custom_params(self):
        """Custom params should be accepted."""
        p = RealWorldSurrogateParams(
            surface_finish_level=0.9,
            lube_mode_level=0.1,
            material_quality_level=0.5,
            coating_level=0.3,
        )
        assert p.surface_finish_level == 0.9
        assert p.lube_mode_level == 0.1


class TestSurrogateEvaluation:
    """Test evaluate_realworld_surrogates output validity."""

    def test_default_params_finite(self):
        """Default params should produce finite results."""
        result = evaluate_realworld_surrogates(DEFAULT_REALWORLD_PARAMS)
        assert np.isfinite(result.lambda_min)
        assert np.isfinite(result.scuff_margin_C)
        assert np.isfinite(result.micropitting_safety)
        assert np.isfinite(result.material_temp_margin_C)
        assert np.isfinite(result.total_cost_index)

    def test_extreme_low_finite(self):
        """All-zero levels should produce finite (possibly bad) results."""
        p = RealWorldSurrogateParams(
            surface_finish_level=0.0,
            lube_mode_level=0.0,
            material_quality_level=0.0,
            coating_level=0.0,
        )
        result = evaluate_realworld_surrogates(p)
        assert np.isfinite(result.lambda_min)
        assert np.isfinite(result.scuff_margin_C)

    def test_extreme_high_finite(self):
        """All-one levels should produce finite (good) results."""
        p = RealWorldSurrogateParams(
            surface_finish_level=1.0,
            lube_mode_level=1.0,
            material_quality_level=1.0,
            coating_level=1.0,
        )
        result = evaluate_realworld_surrogates(p)
        assert np.isfinite(result.lambda_min)
        assert np.isfinite(result.scuff_margin_C)
        # High quality should have positive temp margin
        assert result.material_temp_margin_C > 0

    def test_better_finish_higher_lambda(self):
        """Monotonicity: better surface finish should increase λ."""
        p_rough = RealWorldSurrogateParams(surface_finish_level=0.0)
        p_smooth = RealWorldSurrogateParams(surface_finish_level=1.0)
        r_rough = evaluate_realworld_surrogates(p_rough)
        r_smooth = evaluate_realworld_surrogates(p_smooth)
        assert r_smooth.lambda_min >= r_rough.lambda_min, (
            f"λ should increase with better finish: "
            f"rough={r_rough.lambda_min:.3f}, smooth={r_smooth.lambda_min:.3f}"
        )

    def test_feature_importance_present(self):
        """Feature importance ranking should have entries."""
        result = evaluate_realworld_surrogates(DEFAULT_REALWORLD_PARAMS)
        assert len(result.feature_importance) > 0
        # Should be sorted descending by importance
        importances = [v for _, v in result.feature_importance]
        assert importances == sorted(importances, reverse=True)

    def test_lube_regime_valid(self):
        """Lubrication regime should be one of the known values."""
        result = evaluate_realworld_surrogates(DEFAULT_REALWORLD_PARAMS)
        assert result.lube_regime in {"boundary", "mixed", "full_ehl"}


class TestConstraints:
    """Test constraint computation from surrogate results."""

    def test_constraint_names(self):
        """Constraint names should be the expected list."""
        assert len(REALWORLD_CONSTRAINT_NAMES) == 7
        assert "rw_lambda_min" in REALWORLD_CONSTRAINT_NAMES
        assert "rw_life_damage_10k" in REALWORLD_CONSTRAINT_NAMES

    def test_constraints_finite(self):
        """Default params should produce finite constraints."""
        result = evaluate_realworld_surrogates(DEFAULT_REALWORLD_PARAMS)
        G, names = compute_realworld_constraints(result)
        assert len(G) == 7
        assert len(names) == 7
        assert all(np.isfinite(g) for g in G)

    def test_constraint_sign_convention(self):
        """G <= 0 should be feasible for high-quality params."""
        p = RealWorldSurrogateParams(
            surface_finish_level=1.0,
            lube_mode_level=1.0,
            material_quality_level=1.0,
            coating_level=0.0,
        )
        result = evaluate_realworld_surrogates(p, operating_temp_C=100.0)
        G, names = compute_realworld_constraints(result, operating_temp_C=100.0)

        # Material temp margin should be satisfied (positive margin → G < 0)
        idx = names.index("rw_material_temp")
        assert G[idx] <= 0, f"Material temp constraint should be feasible: G={G[idx]}"

    def test_cost_constraint_zero_below_threshold(self):
        """Cost constraint should be zero when below threshold."""
        p = RealWorldSurrogateParams(
            surface_finish_level=0.0,
            lube_mode_level=0.0,
            material_quality_level=0.0,
            coating_level=0.0,
        )
        result = evaluate_realworld_surrogates(p)
        G, names = compute_realworld_constraints(result)
        idx = names.index("rw_cost_index")
        # Cheapest options should be below threshold
        assert G[idx] <= 0, f"Cheapest options should not violate cost: G={G[idx]}"
