"""Tests for legacy artifact feature-index translation after thermo block expansion."""

from __future__ import annotations

import numpy as np

from larrak2.core.encoding import LEGACY_ENCODING_VERSION, N_TOTAL
from larrak2.surrogate.stack.runtime import feature_vector_from_inputs


def test_feature_vector_from_inputs_translates_legacy_indices() -> None:
    x_full = np.arange(N_TOTAL, dtype=np.float64)

    feats = feature_vector_from_inputs(
        ("x_005", "x_006", "rpm", "torque"),
        x_full,
        rpm=1800.0,
        torque=90.0,
        encoding_version=LEGACY_ENCODING_VERSION,
    )

    np.testing.assert_allclose(feats, np.array([10.0, 11.0, 1800.0, 90.0], dtype=np.float64))
