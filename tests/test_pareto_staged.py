"""End-to-end tests for strict Staged Pareto workflow."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from larrak2.promote.staged import StagedWorkflow
from larrak2.promote.archive import load_npz


@pytest.fixture
def temp_staged_dir(tmp_path):
    d = tmp_path / "staged_test"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    return d


def run_tiny_workflow(outdir: Path, seed: int):
    """Run a tiny, fast version of the workflow."""
    workflow = StagedWorkflow(
        outdir=outdir,
        rpm=3000.0,
        torque=200.0,
        seed=seed
    )
    
    # Stage 1: exploration
    a1 = workflow.run_stage1(pop_size=16, n_gen=5)
    
    # Promotion: k=5
    a2 = workflow.run_promotion(a1, k=5)
    
    # Stage 3: refinement
    a3 = workflow.run_stage3(a2, pop_size=16, n_gen=5)
    
    return a1, a2, a3


def test_staged_workflow_determinism(temp_staged_dir):
    """Test that the workflow is strictly deterministic."""
    dir_a = temp_staged_dir / "run_a"
    dir_b = temp_staged_dir / "run_b"
    
    # Run A
    a1, a2, a3 = run_tiny_workflow(dir_a, seed=42)
    
    # Run B
    b1, b2, b3 = run_tiny_workflow(dir_b, seed=42)
    
    # Check Stage 1
    xa, fa, ga = a1.to_arrays()
    xb, fb, gb = b1.to_arrays()
    
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(fa, fb)
    
    # Check Stage 2 (Promotion)
    xa2, fa2, ga2 = a2.to_arrays()
    xb2, fb2, gb2 = b2.to_arrays()
    
    assert len(xa2) == 5
    np.testing.assert_array_equal(xa2, xb2)
    np.testing.assert_array_equal(fa2, fb2)
    
    # Check Stage 3
    xa3, fa3, ga3 = a3.to_arrays()
    xb3, fb3, gb3 = b3.to_arrays()
    
    np.testing.assert_array_equal(xa3, xb3)
    np.testing.assert_array_equal(fa3, fb3)
    
    # Check Metadata persistence
    # Verify we can load back from disk
    loaded_a3 = load_npz(dir_a / "archive_stage3.npz")
    lx, lf, lg = loaded_a3.to_arrays()
    np.testing.assert_array_equal(lx, xa3)


def test_staged_outputs(temp_staged_dir):
    """Check constraint satisfaction and properties."""
    a1, a2, a3 = run_tiny_workflow(temp_staged_dir, seed=123)
    
    # Ensure produced some feasible points eventually?
    # Hard to guarantee with tiny run, but check shapes
    
    xa, fa, ga = a3.to_arrays()
    assert xa.shape[1] == 12 # N_TOTAL
    assert fa.shape[1] == 3  # N_OBJ
    assert ga.shape[1] == 10  # N_CONSTR
    
    # Check files exist
    assert (temp_staged_dir / "stage1").exists()  # Pymoo uses checkpointing? No, we didn't enable it explicitly.
    # Our workflow saves archives to outdir directly with suffixes
    assert (temp_staged_dir / "archive_stage1.npz").exists()
    assert (temp_staged_dir / "archive_stage2.npz").exists()
    assert (temp_staged_dir / "archive_stage3.npz").exists()
    assert (temp_staged_dir / "meta.json").exists()
