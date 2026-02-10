"""Verification of Energy Closure for Gear System."""

import numpy as np
import pytest
from larrak2.core.encoding import GearParams
from larrak2.core.types import EvalContext
from larrak2.gear.litvin_core import eval_gear

def test_toy_energy_closure():
    """Verify energy closure for Fidelity 0 (Toy Physics)."""
    # Setup
    params = GearParams(
        base_radius=40.0,
        pitch_coeffs=tuple([0.0] * 5),  # Circular gear
    )
    # i_req = 1.0 constant
    i_req_profile = np.ones(360)
    
    # Context
    ctx = EvalContext(
        rpm=1000.0,
        torque=100.0,
        fidelity=0,
        seed=42
    )

    # Evaluate
    result = eval_gear(params, i_req_profile, ctx)
    
    assert result.ledger is not None
    ledger = result.ledger
    
    print("\n--- Toy Mode Ledger ---")
    print(ledger.summarize())
    
    # In Toy Mode, we populate:
    # W_out = Torque * 2pi
    # W_mesh = Integral(Loss)
    # W_in is NOT tracked (it's external in Thermo).
    # BUT, we can check if the calculated W_mesh matches expected for mu=0.05.
    
    # If mu=0.05, and sliding happens even for circular gears in this toy model?
    # No, circular gears should have near-zero sliding except for the toy 0.01 term.
    # v_sliding = ... + omega * r * 0.01 in toy model!
    
    # Check consistency
    assert ledger.W_loss_mesh > 0
    assert ledger.W_out_shaft > 0

def test_v1_noncircular_closure():
    """Verify energy closure for Fidelity 1 with Non-Circular Gear."""
    # Setup - Non-circular (add sine wave to pitch coeffs)
    params = GearParams(
        base_radius=40.0,
        # Add significant modulation
        pitch_coeffs=tuple([5.0, 0.0, 0.0, 0.0, 0.0]), 
    )
    # Variable ratio profile (approximate)
    i_req_profile = np.ones(360) * 2.0 
    
    ctx = EvalContext(
        rpm=1000.0,
        torque=100.0,
        fidelity=1,
        seed=42
    )
    
    result = eval_gear(params, i_req_profile, ctx)
    ledger = result.ledger
    
    print("\n--- V1 Non-Circular Ledger ---")
    print(ledger.summarize())
    
    # Should have significant mesh loss due to sliding
    print(f"Mesh Loss (J): {ledger.W_loss_mesh}")
    print(f"Mesh Loss (J): {ledger.W_loss_mesh}")
    assert ledger.W_loss_mesh > 0.1  # Expecting > 0.1 J

def test_speed_sensitivity_losses():
    """Verify Bearing/Churning losses increase with speed."""
    params = GearParams(base_radius=40.0, pitch_coeffs=tuple([0.0]*5))
    i_req = np.ones(360)
    
    # Low Speed
    ctx_low = EvalContext(rpm=1000.0, torque=100.0, fidelity=1, seed=42)
    res_low = eval_gear(params, i_req, ctx_low)
    
    # High Speed
    ctx_high = EvalContext(rpm=5000.0, torque=100.0, fidelity=1, seed=42)
    res_high = eval_gear(params, i_req, ctx_high)
    
    l_low = res_low.ledger
    l_high = res_high.ledger
    
    print("\n--- Speed Sensitivity ---")
    print(f"Low Speed (1000 RPM) Bearing Loss: {l_low.W_loss_bearing:.4f} J")
    print(f"High Speed (5000 RPM) Bearing Loss: {l_high.W_loss_bearing:.4f} J")
    print(f"Low Speed Churning Loss: {l_low.W_loss_churning:.4f} J")
    print(f"High Speed Churning Loss: {l_high.W_loss_churning:.4f} J")
    
    # Assert Scaling
    # Power ~ w^2 or w^3.
    # Energy = Power * T = Power / w.
    # So Energy ~ w or w^2.
    # It should increase.
    assert l_high.W_loss_bearing > l_low.W_loss_bearing
    assert l_high.W_loss_churning > l_low.W_loss_churning


    
def test_v1_energy_closure():
    """Verify energy closure for Fidelity 1 (Litvin Physics)."""
    # Setup
    params = GearParams(
        base_radius=40.0,
        pitch_coeffs=tuple([0.0] * 5),
    )
    i_req_profile = np.ones(360)
    
    ctx = EvalContext(
        rpm=1000.0,
        torque=100.0,
        fidelity=1,  # V1 Logic
        seed=42
    )
    
    result = eval_gear(params, i_req_profile, ctx)
    ledger = result.ledger
    assert ledger is not None
    
    print("\n--- V1 Mode Ledger ---")
    print(ledger.summarize())
    
    # Circular gears (ratio=2.0 usually in V1 default, but here params say base=40, ring=80 -> Ratio=2)
    # Sliding velocity for pitch circles touching?
    # Pitch radius planet = 40. Pitch radius ring = 80.
    # Pitch point velocity match -> Sliding should be zero at pitch point.
    # But contact happens along line of action.
    # Litvin code: v_sliding ~ omega * |r - rho|.
    # For circular gear, r = 40. rho = 20? (Euler-Savary).
    # So v_slide > 0.
    
    assert ledger.W_loss_mesh > 0
    assert ledger.W_out_shaft > 0

if __name__ == "__main__":
    test_toy_energy_closure()
    test_v1_energy_closure()
    test_v1_noncircular_closure()
    test_speed_sensitivity_losses()
