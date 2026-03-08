"""Test encoding roundtrip consistency."""

import numpy as np
import pytest

from larrak2.core.encoding import (
    LEGACY_N_TOTAL,
    N_GEAR,
    N_REALWORLD,
    N_THERMO,
    N_TOTAL,
    Candidate,
    GearParams,
    RealWorldParams,
    ThermoParams,
    bounds,
    decode_candidate,
    encode_candidate,
    legacy_index_to_current,
    mid_bounds_candidate,
    random_candidate,
    upgrade_legacy_candidate_vector,
)


def test_roundtrip_consistency():
    """Test that decode(encode(candidate)) equals original candidate."""
    thermo = ThermoParams(
        compression_duration=60.0,
        expansion_duration=90.0,
        heat_release_center=15.0,
        heat_release_width=30.0,
        lambda_af=1.0,
        intake_open_offset_from_bdc=-4.0,
        intake_duration_deg=80.0,
        exhaust_open_offset_from_expansion_tdc=-4.0,
        exhaust_duration_deg=90.0,
    )
    gear = GearParams(
        base_radius=40.0,
        pitch_coeffs=np.array([0.1, -0.05, 0.02, 0.0, 0.0, 0.0, 0.0]),
        face_width_mm=10.0,
    )
    realworld = RealWorldParams(
        surface_finish_level=0.7,
        lube_mode_level=0.5,
        material_quality_level=0.3,
        coating_level=0.1,
    )
    candidate = Candidate(thermo=thermo, gear=gear, realworld=realworld)

    # Encode
    x = encode_candidate(candidate)
    assert len(x) == N_TOTAL, f"Expected {N_TOTAL} variables, got {len(x)}"
    assert N_TOTAL == (N_THERMO + N_GEAR + N_REALWORLD), (
        f"N_TOTAL should equal N_THERMO+N_GEAR+N_REALWORLD, got {N_TOTAL}"
    )

    # Decode
    decoded = decode_candidate(x)

    # Check thermo params
    assert decoded.thermo.compression_duration == thermo.compression_duration
    assert decoded.thermo.expansion_duration == thermo.expansion_duration
    assert decoded.thermo.heat_release_center == thermo.heat_release_center
    assert decoded.thermo.heat_release_width == thermo.heat_release_width
    assert decoded.thermo.lambda_af == thermo.lambda_af
    assert decoded.thermo.intake_open_offset_from_bdc == thermo.intake_open_offset_from_bdc
    assert decoded.thermo.intake_duration_deg == thermo.intake_duration_deg
    assert (
        decoded.thermo.exhaust_open_offset_from_expansion_tdc
        == thermo.exhaust_open_offset_from_expansion_tdc
    )
    assert decoded.thermo.exhaust_duration_deg == thermo.exhaust_duration_deg

    # Check gear params
    assert decoded.gear.base_radius == gear.base_radius
    np.testing.assert_array_almost_equal(decoded.gear.pitch_coeffs, gear.pitch_coeffs)
    assert decoded.gear.face_width_mm == gear.face_width_mm

    # Check realworld params
    assert decoded.realworld.surface_finish_level == realworld.surface_finish_level
    assert decoded.realworld.lube_mode_level == realworld.lube_mode_level
    assert decoded.realworld.material_quality_level == realworld.material_quality_level
    assert decoded.realworld.coating_level == realworld.coating_level


def test_encode_decode_roundtrip_array():
    """Test that encode(decode(x)) recovers original x."""
    rng = np.random.default_rng(42)

    for _ in range(10):
        x_original = random_candidate(rng)
        candidate = decode_candidate(x_original)
        x_recovered = encode_candidate(candidate)

        np.testing.assert_array_almost_equal(
            x_original,
            x_recovered,
            decimal=10,
            err_msg="encode(decode(x)) should recover original x",
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
        intake_open_offset_from_bdc=-6.0,
        intake_duration_deg=72.0,
        exhaust_open_offset_from_expansion_tdc=-5.0,
        exhaust_duration_deg=88.0,
    )

    arr = original.to_array()
    assert len(arr) == N_THERMO

    recovered = ThermoParams.from_array(arr)
    assert recovered.compression_duration == original.compression_duration
    assert recovered.expansion_duration == original.expansion_duration
    assert recovered.intake_open_offset_from_bdc == original.intake_open_offset_from_bdc
    assert recovered.exhaust_duration_deg == original.exhaust_duration_deg


def test_gear_params_to_from_array():
    """Test GearParams array conversion."""
    original = GearParams(
        base_radius=35.0,
        pitch_coeffs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        face_width_mm=12.0,
    )

    arr = original.to_array()
    assert len(arr) == 9  # 1 + 7 coeffs + 1 face_width

    recovered = GearParams.from_array(arr)
    assert recovered.base_radius == original.base_radius
    np.testing.assert_array_almost_equal(recovered.pitch_coeffs, original.pitch_coeffs)
    assert recovered.face_width_mm == original.face_width_mm


def test_realworld_params_to_from_array():
    """Test RealWorldParams array conversion."""
    original = RealWorldParams(
        surface_finish_level=0.8,
        lube_mode_level=0.6,
        material_quality_level=0.4,
        coating_level=0.2,
    )

    arr = original.to_array()
    assert len(arr) == N_REALWORLD

    recovered = RealWorldParams.from_array(arr)
    assert recovered.surface_finish_level == original.surface_finish_level
    assert recovered.lube_mode_level == original.lube_mode_level
    assert recovered.material_quality_level == original.material_quality_level
    assert recovered.coating_level == original.coating_level


def test_decision_vector_layout():
    """Verify vector layout: thermo, then gear, then realworld segments."""
    x = mid_bounds_candidate()
    assert len(x) == N_TOTAL

    candidate = decode_candidate(x)

    # Thermo occupies x[0:N_THERMO]
    np.testing.assert_array_almost_equal(candidate.thermo.to_array(), x[0:N_THERMO])

    # Gear occupies x[N_THERMO:N_THERMO+N_GEAR]
    np.testing.assert_array_almost_equal(candidate.gear.to_array(), x[N_THERMO : N_THERMO + N_GEAR])

    # Realworld occupies x[N_THERMO+N_GEAR:N_TOTAL]
    np.testing.assert_array_almost_equal(
        candidate.realworld.to_array(), x[N_THERMO + N_GEAR : N_TOTAL]
    )

    # Face width bounds
    xl, xu = bounds()
    face_width_idx = N_THERMO + N_GEAR - 1
    assert xl[face_width_idx] == 4.0, "Face width lower bound should be 4.0 mm"
    assert xu[face_width_idx] == 14.0, (
        "Face width upper bound should be 14.0 mm (gear body z-height)"
    )

    # Realworld bounds
    rw_slice = slice(N_THERMO + N_GEAR, N_TOTAL)
    assert np.all(xl[rw_slice] == 0.0), "Realworld lower bounds should be 0.0"
    assert np.all(xu[rw_slice] == 1.0), "Realworld upper bounds should be 1.0"


def test_decode_legacy_vector_injects_default_timing() -> None:
    legacy = np.zeros(LEGACY_N_TOTAL, dtype=np.float64)
    legacy[:5] = np.array([60.0, 90.0, 15.0, 30.0, 1.0], dtype=np.float64)
    legacy[5:14] = np.array([40.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float64)
    legacy[-8:] = np.linspace(0.1, 0.8, 8, dtype=np.float64)

    candidate = decode_candidate(legacy)

    assert candidate.thermo.timing_legacy_injected is True
    assert candidate.thermo.timing_source == "legacy_default_profile"
    upgraded = upgrade_legacy_candidate_vector(legacy)
    assert upgraded.shape == (N_TOTAL,)
    np.testing.assert_allclose(upgraded[:5], legacy[:5])
    assert legacy_index_to_current(5) == 10
