"""Fail-hard thermo behavior checks (no implicit fallback)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from larrak2.core.encoding import mid_bounds_candidate
from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext


def test_missing_openfoam_artifact_fails_hard_in_fidelity2(tmp_path: Path) -> None:
    x = mid_bounds_candidate()
    missing_model = tmp_path / "missing_openfoam_nn.pt"
    anchor_manifest = tmp_path / "anchors.json"
    anchor_manifest.write_text(
        json.dumps(
            {
                "version": "test",
                "validated_envelope": {
                    "rpm_min": 0.0,
                    "rpm_max": 1e9,
                    "torque_min": 0.0,
                    "torque_max": 1e9,
                },
                "thresholds": {},
                "anchors": [{"rpm": 2800, "torque": 140}],
            }
        ),
        encoding="utf-8",
    )

    ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=2,
        seed=17,
        openfoam_model_path=str(missing_model),
        thermo_anchor_manifest_path=str(anchor_manifest),
    )

    with pytest.raises(FileNotFoundError):
        evaluate_candidate(x, ctx)


def test_missing_thermo_constants_fails_hard(tmp_path: Path) -> None:
    x = mid_bounds_candidate()
    missing_constants = tmp_path / "missing_constants.json"

    ctx = EvalContext(
        rpm=2800.0,
        torque=140.0,
        fidelity=1,
        seed=17,
        thermo_constants_path=str(missing_constants),
    )

    with pytest.raises(FileNotFoundError):
        evaluate_candidate(x, ctx)
