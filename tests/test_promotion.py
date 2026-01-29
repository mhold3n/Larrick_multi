"""Tests for Phase 10: Multi-Fidelity Promotion."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.encoding import N_TOTAL
from larrak2.core.types import EvalContext
from larrak2.promote.manager import PromotionManager
from larrak2.promote.selectors import select_k_best_ref_dirs


@pytest.fixture
def temp_archive_dir(tmp_path):
    d = tmp_path / "promote_test"
    d.mkdir()
    return d


def test_select_k_best_ref_dirs_determinism():
    """Test that reference direction selection is deterministic."""
    # Synthetic Pareto front
    # 3 obj, 100 points
    rng = np.random.default_rng(42)
    F = rng.random((100, 3))
    # Make it non-dominated-ish (minimize)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    
    idx1 = select_k_best_ref_dirs(F, k=10)
    idx2 = select_k_best_ref_dirs(F, k=10)
    
    np.testing.assert_array_equal(idx1, idx2)
    assert len(idx1) == 10
    
    # Check bounds
    assert np.all(idx1 >= 0)
    assert np.all(idx1 < 100)


def test_promotion_manager_workflow(temp_archive_dir):
    """Test full promotion workflow."""
    pm = PromotionManager(temp_archive_dir)
    
    # 1. Create dummy population
    rng = np.random.default_rng(1)
    X = rng.random((20, N_TOTAL))
    F1 = rng.random((20, 3))
    # Make F1 look like pareto (sorted by first obj)
    F1 = F1[np.argsort(F1[:, 0])]
    G1 = np.zeros((20, 7)) # Feasible
    
    # 2. Ingest
    pm.ingest_population(X, F1, G1, fidelity=1)
    pm.save_archive()
    
    assert len(pm.archive) == 20
    
    # 3. Select for promotion
    # This calls select_k_best_ref_dirs internally
    promoted = pm.select_for_promotion(fidelity_source=1, n_promote=5)
    assert len(promoted) == 5
    
    # 4. Evaluate at Fidelity 2
    ctx_fid2 = EvalContext(rpm=3000, torque=200, fidelity=2, seed=1)
    
    # Mock evaluate_candidate to return feasible result
    from unittest.mock import patch
    from larrak2.core.types import EvalResult
    
    with patch("larrak2.promote.manager.evaluate_candidate") as mock_eval:
        # returns dummy result
        mock_eval.return_value = EvalResult(
            F=np.array([-0.5, 10.0, 50.0]),
            G=np.zeros(7), # feasible
            diag={"versions": {"surrogate_used": True}}
        )
        
        n_success = pm.promote_and_evaluate(promoted, ctx_fid2)
    
    assert n_success == 5
    
    # 5. Check archive content
    # Should have F_fid2 fields
    for x_hash in promoted:
        record = pm.archive[x_hash]
        assert "F_fid2" in record
        assert "versions_fid2" in record
        assert record["versions_fid2"]["surrogate_used"] is True
        
    # Check get_pareto_front(2)
    X2, F2, hashes2 = pm.get_pareto_front(fidelity=2)
    assert len(X2) > 0
    assert len(F2) == len(X2)


def test_evaluator_fidelity_2_structure():
    """Test that fidelity=2 returns corrected objectives."""
    ctx = EvalContext(rpm=3000, torque=200, fidelity=2, seed=42)
    x = np.random.rand(N_TOTAL)
    
    res = evaluate_candidate(x, ctx)
    
    # Check metadata
    assert "versions" in res.diag
    assert res.diag["versions"]["surrogate_used"] is True
    assert "delta_eff" in res.diag["versions"]
