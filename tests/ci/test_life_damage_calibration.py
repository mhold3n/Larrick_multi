"""Life-damage calibration and strict route-resolution checks."""

from __future__ import annotations

import pytest

from larrak2.realworld import life_damage


def test_sigma_ref_resolves_from_calibration_file() -> None:
    life_damage.invalidate_limit_stress_cache()
    ref = life_damage.get_sigma_ref_for_route("AISI_9310", strict_data=True)
    assert ref == pytest.approx(life_damage._SIGMA_REF_MPA)


def test_sigma_ref_missing_route_fails_when_strict(monkeypatch) -> None:
    life_damage.invalidate_limit_stress_cache()
    monkeypatch.setattr(
        life_damage,
        "_load_limit_stress_table",
        lambda strict_data=None: {"AISI_9310": 1500.0},
    )
    with pytest.raises(ValueError, match="missing"):
        life_damage.get_sigma_ref_for_route("NON_EXISTENT", strict_data=True)
