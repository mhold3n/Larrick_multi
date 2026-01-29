"""Strict integration tests for Surrogate Engine."""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext
from larrak2.core.encoding import random_candidate
from larrak2.surrogate.features import get_gear_schema_v1
from larrak2.surrogate.inference import SurrogateEngine, get_surrogate_engine


class MockModel:
    def predict(self, x):
        # Legacy behavior: returns single value
        return np.array([0.5])


class MockEnsemble:
    def __init__(self, hash_str):
        self.schema_hash = hash_str
        
    def predict(self, x):
        # New behavior: returns (mean, std)
        return (np.array([0.7]), np.array([0.05]))


@pytest.fixture
def temp_model_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


def test_schema_stability():
    """Ensure schema hash is stable."""
    s1 = get_gear_schema_v1()
    s2 = get_gear_schema_v1()
    assert s1._hash == s2._hash
    assert len(s1._hash) == 16


def test_engine_load_legacy_dict(temp_model_dir, monkeypatch):
    """Test loading a legacy dict artifact."""
    
    # Create valid artifact
    schema = get_gear_schema_v1()
    artifact = {
        "model": MockModel(),
        "schema_hash": schema._hash,
        "meta": {}
    }
    
    p = temp_model_dir / "model_gear.pkl"
    with open(p, "wb") as f:
        pickle.dump(artifact, f)
        
    # Mock registry
    monkeypatch.setattr("larrak2.surrogate.registry.ModelRegistry.get_path", 
                        lambda self, key: p if key == "gear" else None)
                        
    engine = SurrogateEngine()
    assert "gear" in engine.models
    
    # Test predict (legacy model returnsscalar)
    x = random_candidate()
    d_eff, d_loss, meta = engine.predict_corrections(x)
    
    assert d_loss == 0.5
    assert "gear" in meta["surrogates_active"]
    assert meta["uncertainty"]["gear"] == 0.0


def test_engine_load_ensemble_object(temp_model_dir, monkeypatch):
    """Test loading a new Ensemble object artifact."""
    
    schema = get_gear_schema_v1()
    ensemble = MockEnsemble(schema._hash)
    
    p = temp_model_dir / "model_gear.pkl"
    with open(p, "wb") as f:
        pickle.dump(ensemble, f)
        
    monkeypatch.setattr("larrak2.surrogate.registry.ModelRegistry.get_path", 
                        lambda self, key: p if key == "gear" else None)
    
    engine = SurrogateEngine()
    assert "gear" in engine.models
    
    x = random_candidate()
    d_eff, d_loss, meta = engine.predict_corrections(x)
    
    assert d_loss == 0.7
    assert meta["uncertainty"]["gear"] == 0.05


def test_engine_reject_mismatch(temp_model_dir, monkeypatch):
    """Test rejection of model with wrong schema."""
    
    # Create artifact with WRONG hash
    artifact = {
        "model": MockModel(),
        "schema_hash": "badhash123",
        "meta": {}
    }
    
    p = temp_model_dir / "model_gear.pkl"
    with open(p, "wb") as f:
        pickle.dump(artifact, f)
        
    monkeypatch.setattr("larrak2.surrogate.registry.ModelRegistry.get_path", 
                        lambda self, key: p if key == "gear" else None)
    
    with pytest.warns(UserWarning, match="Schema Mismatch"):
        engine = SurrogateEngine()
        
    assert "gear" not in engine.models
    
    # Predict should return 0
    x = random_candidate()
    d_eff, d_loss, meta = engine.predict_corrections(x)
    assert d_loss == 0.0
    assert "gear" not in meta["surrogates_active"]


def test_evaluator_integration(monkeypatch):
    """Test evaluator calls engine."""
    
    # Mock global engine to return known values
    class MockEngine:
        def predict_corrections(self, x):
            return 0.1, 0.2, {"surrogates_active": ["mock"]}
            
    monkeypatch.setattr("larrak2.surrogate.inference._ENGINE", MockEngine())
    
    ctx = EvalContext(rpm=3000, torque=200, fidelity=2, seed=1)
    x = random_candidate()
    
    res = evaluate_candidate(x, ctx)
    
    # Check diagnostics
    versions = res.diag["versions"]
    assert versions["version_surrogate"] == "SurrogateEngine_v1"
    assert "mock" in versions["active_models"]
    
    # Check values applied
    # We can't easily check exact value without knowing base physics, 
    # but we know it calls the engine.
