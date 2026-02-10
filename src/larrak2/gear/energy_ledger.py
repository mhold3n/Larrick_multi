"""Energy accounting for gear system verification.

Implements the "Energy Closure" check:
    W_in = W_out + Sum(Losses) + Delta_E_stored

This ensures the physics model is self-consistent and provides a ground truth
for verifying surrogate models.
"""

from __future__ import annotations

from dataclasses import dataclass



@dataclass
class EnergyLedger:
    """Tracks energy flow through the gear system per cycle."""

    # Work Terms (Joules) [Positive = Energy Added to System]
    W_in_piston: float = 0.0  # Work done by piston on gas (or vice versa, careful with sign logic)
    # Actually, W_in usually means Work INPUT to the gear system FROM the piston.
    # W_piston_gas is work done BY gas ON piston.
    # So W_in = W_piston_gas.

    # Output Work (Joules) [Positive = Useful Output]
    W_out_shaft: float = 0.0  # Useful work delivered to load

    # Loss Terms (Joules) [Positive = Energy Dissipated]
    W_loss_mesh: float = 0.0
    W_loss_bearing: float = 0.0
    W_loss_churning: float = 0.0
    W_loss_windage: float = 0.0  # Explicit aerodynamic drag distinct from churning

    # Actuation Work (Joules) [Net work done by/on VCR mechanism]
    W_actuation: float = 0.0

    # Stored Energy Change (Joules) [Final - Initial]
    # For a steady-state cycle, this should be zero.
    delta_E_stored: float = 0.0

    tolerance: float = 1e-3  # Joules

    def compute_closure_error(self) -> float:
        """Compute energy balance error (residual).

        Residual = W_in - (W_out + Losses + Actuation + Delta_E)
        """
        w_losses = (
            self.W_loss_mesh
            + self.W_loss_bearing
            + self.W_loss_churning
            + self.W_loss_windage
        )
        # Using W_in = W_out + Losses + ...
        # So Residual = W_in - W_out - Losses - Actuation - Delta_E
        # Note: W_actuation sign convention: Positive = Work DONE BY actuator ON system.
        # If Actuation adds energy, it's an input.
        # Let's align with: Inputs = Outputs + Storage
        # Inputs: W_in_piston, W_actuation (if positive)
        # Outputs: W_out_shaft, Losses
        # Storage: delta_E_stored

        # Actually, simpler ledger:
        # Sum(Inputs) - Sum(Outputs) - Sum(Accumulation) = 0
        
        # Let's assume W_actuation is NET work input by actuator.
        total_in = self.W_in_piston + self.W_actuation
        total_out = self.W_out_shaft + w_losses + self.delta_E_stored
        
        return total_in - total_out

    @property
    def is_closed(self) -> bool:
        """Check if energy balance is satisfied within tolerance."""
        return abs(self.compute_closure_error()) < self.tolerance

    @property
    def efficiency_mech(self) -> float:
        """Mechanical Efficiency (W_out / W_in).
        
        If W_in_piston is provided (measured), use it.
        Otherwise, infer W_in from equilibrium (W_out + Losses).
        """
        w_in = self.W_in_piston
        
        # If external input not provided, infer from output + losses (Backwards mode)
        if abs(w_in) < 1e-9:
             w_losses = (
                self.W_loss_mesh + self.W_loss_bearing + self.W_loss_churning + self.W_loss_windage
            )
             w_in = self.W_out_shaft + w_losses + self.delta_E_stored - self.W_actuation
             
        if abs(w_in) < 1e-9:
            return 0.0
            
        return self.W_out_shaft / w_in

    def summarize(self) -> str:
        return (
            f"W_in: {self.W_in_piston:.4f} | W_out: {self.W_out_shaft:.4f}\n"
            f"Losses -- Mesh: {self.W_loss_mesh:.4f}, Bear: {self.W_loss_bearing:.4f}, "
            f"Churn: {self.W_loss_churning:.4f}\n"
            f"Closure Err: {self.compute_closure_error():.4f} | Eff: {self.efficiency_mech:.4%}"
        )
