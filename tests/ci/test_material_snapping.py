import numpy as np
import pytest

from larrak2.cem.material_snapping import (
    NORM_BOUNDS,
    get_soft_selected_routes,
    invalidate_snapping_cache,
)
from larrak2.core.encoding import RealWorldParams


@pytest.fixture(autouse=True)
def setup_teardown_cache():
    # Ensure a fresh cache state for tests
    invalidate_snapping_cache()
    yield
    invalidate_snapping_cache()


def test_soft_selected_routes_exact_match():
    # Pyrowear_53: [61.0, 60.0, 200.0, 0.5]
    state = np.array([
        (61.0 - 50.0) / 20.0,  # case_hardness (50-70)
        (60.0 - 30.0) / 130.0, # KIC (30-160)
        (200.0 - 100.0) / 400.0, # Temp (100-500)
        (0.5 - 0.0) / 1.0,     # Cleanliness (0-1)
    ])
    
    dist, routes = get_soft_selected_routes(state, gear_bulk_temp_C=150.0, k=3, tau=0.1)
    
    assert np.isclose(dist, 0.0, atol=1e-5)
    assert routes[0][0] == "Pyrowear_53"
    assert routes[0][1] > 0.9  # Should dominate the softmax weighting heavily


def test_temperature_filtering():
    # Same vector as Pyrowear, but request 300C gear temp
    state = np.array([0.55, 0.23, 0.25, 0.5])
    
    dist, routes = get_soft_selected_routes(state, gear_bulk_temp_C=300.0, k=3, tau=0.1)
    
    # Pyrowear_53 max temp is 200C. It should be filtered out.
    # The nearest valid routes should be the 315C or 400C alloys (CBS-50NiL, M50NiL, Ferrium C64)
    route_ids = [r[0] for r in routes]
    assert "Pyrowear_53" not in route_ids
    assert "AISI_9310" not in route_ids  # Max 150C


def test_no_valid_routes():
    # Request 600C gear temp
    state = np.array([0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="No routes available"):
        get_soft_selected_routes(state, gear_bulk_temp_C=600.0)


def test_encoding_backwards_compatibility():
    # Legacy Array (Length 8)
    arr_legacy = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    params = RealWorldParams.from_array(arr_legacy)
    assert params.material_quality_level == 0.3
    assert params.material_state is None
    
    # It should seamlessly re-encode to the legacy length
    arr_legacy_out = params.to_array()
    assert len(arr_legacy_out) == 8
    np.testing.assert_allclose(arr_legacy, arr_legacy_out)
    
    # Phase 8 Array (Length 13 with -999.0 sentinel)
    state_vector = np.array([0.1, 0.2, 0.3, 0.4])
    params_new = RealWorldParams(
        surface_finish_level=0.5,
        lube_mode_level=0.6,
        material_quality_level=None,
        coating_level=0.7,
        hunting_level=0.8,
        oil_flow_level=0.9,
        oil_supply_temp_level=1.0,
        evacuation_level=0.1,
        material_state=state_vector,
    )
    
    arr_new = params_new.to_array()
    assert len(arr_new) == 12
    assert arr_new[0] == -999.0
    np.testing.assert_allclose(arr_new[1:5], state_vector)
    
    # Parse back
    params_reconstructed = RealWorldParams.from_array(arr_new)
    assert params_reconstructed.material_state is not None
    np.testing.assert_allclose(params_reconstructed.material_state, state_vector)
    assert params_reconstructed.surface_finish_level == 0.5
