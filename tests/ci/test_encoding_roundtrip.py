"""Test encoding roundtrip consistency."""

import numpy as np
import pytest

from larrak2.core.encoding import (
    Candidate,
    ThermoParams,
    GearParams,
    decode_candidate,
    encode_candidate,
    bounds,
    random_candidate,
    mid_bounds_candidate,
    N_TOTAL,
)


def test_roundtrip_consistency():
    """Test that decode(encode(candidate)) equals original candidate."""
    thermo = ThermoParams(
        compression_duration=60.0,
        expansion_duration=90.0,
        heat_release_center=15.0,
        heat_release_width=30.0,
        lambda_af=1.0,
    )
    gear = GearParams(
        base_radius=40.0,
        pitch_coeffs=np.array([0.1, -0.05, 0.02, 0.0, 0.0, 0.0, 0.0]),
    )
    candidate = Candidate(thermo=thermo, gear=gear)

    # Encode
    x = encode_candidate(candidate)
    assert len(x) == N_TOTAL, f"Expected {N_TOTAL} variables, got {len(x)}"

    # Decode
    decoded = decode_candidate(x)

    # Check thermo params
    assert decoded.thermo.compression_duration == thermo.compression_duration
    assert decoded.thermo.expansion_duration == thermo.expansion_duration
    assert decoded.thermo.heat_release_center == thermo.heat_release_center
    assert decoded.thermo.heat_release_width == thermo.heat_release_width
    assert decoded.thermo.lambda_af == thermo.lambda_af

    # Check gear params
    assert decoded.gear.base_radius == gear.base_radius
    np.testing.assert_array_almost_equal(decoded.gear.pitch_coeffs, gear.pitch_coeffs)


def test_encode_decode_roundtrip_array():
    """Test that encode(decode(x)) recovers original x."""
    rng = np.random.default_rng(42)

    for _ in range(10):
        x_original = random_candidate(rng)
        candidate = decode_candidate(x_original)
        x_recovered = encode_candidate(candidate)

        np.testing.assert_array_almost_equal(
            x_original, x_recovered,
            decimal=10,
            err_msg="encode(decode(x)) should recover original x"
        )


def test_bounds_shape():
    """Test that bounds have correct shape."""
    xl, xu = bounds()

    assert xl.shape == (N_TOTAL,)
    assert xu.shape == (N_TOTAL,)
    assert np.all(xl < xu), "Lower bounds should be less than upper bounds"


def test_mid_bounds_within_bounds():
    """Test that mid_bounds_candidate is within bounds."""
    xl, xu = bounds()
    x_mid = mid_bounds_candidate()

    assert np.all(x_mid >= xl), "mid_bounds below lower bounds"
    assert np.all(x_mid <= xu), "mid_bounds above upper bounds"


def test_random_candidate_within_bounds():
    """Test that random_candidate produces values within bounds."""
    xl, xu = bounds()
    rng = np.random.default_rng(42)

    for _ in range(100):
        x = random_candidate(rng)
        assert np.all(x >= xl), "random_candidate below lower bounds"
        assert np.all(x <= xu), "random_candidate above upper bounds"


def test_decode_wrong_length():
    """Test that decode raises on wrong length."""
    with pytest.raises(ValueError, match="Expected.*variables"):
        decode_candidate(np.array([1.0, 2.0, 3.0]))


def test_thermo_params_to_from_array():
    """Test ThermoParams array conversion."""
    original = ThermoParams(
        compression_duration=45.0,
        expansion_duration=85.0,
        heat_release_center=10.0,
        heat_release_width=25.0,
        lambda_af=1.0,
    )

    arr = original.to_array()
    assert len(arr) == 5

    recovered = ThermoParams.from_array(arr)
    assert recovered.compression_duration == original.compression_duration
    assert recovered.expansion_duration == original.expansion_duration


def test_gear_params_to_from_array():
    """Test GearParams array conversion."""
    original = GearParams(
        base_radius=35.0,
        pitch_coeffs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    )

    arr = original.to_array()
    assert len(arr) == 8  # 1 + 7 coeffs

    recovered = GearParams.from_array(arr)
    assert recovered.base_radius == original.base_radius
    np.testing.assert_array_almost_equal(recovered.pitch_coeffs, original.pitch_coeffs)
