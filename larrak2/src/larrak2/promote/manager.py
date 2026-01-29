"""Promotion Orchestrator.

Manages the valid transition of candidates between fidelity stages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext
from larrak2.promote.archive import ArchiveBundle, ArchiveRecord
from larrak2.promote.selectors import select_strict_nsga3, select_hybrid


class PromotionManager:
    """Stateless manager for promotion logic."""

    def select_candidates(
        self,
        archive: ArchiveBundle,
        k: int,
        seed: int | None = None,
        strategy: str = "nsga3",
        **kwargs,
    ) -> list[int]:
        """Select k candidates from archive for promotion.
        
        Strategies:
            - 'nsga3': Strict Pareto (NDS -> RefDir -> Hash).
            - 'hybrid': Exploitation (NSGA-III) + Exploration (Max Uncertainty).
                        Requires 'ratio_explore' in kwargs (default 0.2).
        
        Args:
            archive: Source archive bundle.
            k: Number to select.
            seed: Unused for strict deterministic logic.
            strategy: Selection strategy.
            **kwargs: Extra args for selectors (e.g., ratio_explore).
                  
        Returns:
            Indices of selected records in the archive.
        """
        if not archive.records:
            return []
            
        X, F, G = archive.to_arrays()
        hashes = archive.get_hashes()
        
        # Filter for feasibility (Strict)
        feasible_mask = np.all(G <= 1e-6, axis=1)
        feasible_indices = np.where(feasible_mask)[0]
        
        if len(feasible_indices) == 0:
            print("Warning: No feasible candidates to promote.")
            return []
            
        F_feas = F[feasible_indices]
        hashes_feas = [hashes[i] for i in feasible_indices]
        
        # Extract Uncertainty Scores depending on strategy
        uncertainty_scores = []
        if strategy == "hybrid":
            for i in feasible_indices:
                rec = archive.records[i]
                # Path: diag -> versions -> uncertainty -> {model: val}
                unc_dict = rec.diag.get("versions", {}).get("uncertainty", {})
                if not isinstance(unc_dict, dict):
                    unc_dict = {}
                # Sum uncertainty across all models
                score = sum(float(v) for v in unc_dict.values())
                uncertainty_scores.append(score)
        
        uncertainty_scores = np.array(uncertainty_scores)
        
        # Dispatch
        if strategy == "hybrid":
            ratio = kwargs.get("ratio_explore", 0.2)
            selected_local_indices = select_hybrid(
                F_feas, 
                hashes_feas, 
                k, 
                uncertainty_scores, 
                ratio_explore=ratio
            )
        else:
            # Default to NSGA3
            selected_local_indices = select_strict_nsga3(F_feas, hashes_feas, k)
        
        # Map back to global archive indices
        return [feasible_indices[i] for i in selected_local_indices]

    def promote_candidates(
        self,
        archive: ArchiveBundle,
        indices: list[int],
        ctx_hi: EvalContext,
    ) -> ArchiveBundle:
        """Evaluate selected candidates at higher fidelity.
        
        Args:
            archive: Source archive.
            indices: Indices to promote.
            ctx_hi: Context for high-fidelity evaluation.
            
        Returns:
            New ArchiveBundle containing high-fidelity results.
        """
        new_bundle = ArchiveBundle()
        
        print(f"Promoting {len(indices)} candidates to Fidelity {ctx_hi.fidelity}...")
        
        # We process in order of indices (which came from selector)
        # Sort indices to match archive order or keep selection order?
        # Keep selection order (might imply priority).
        
        for idx in indices:
            src_rec = archive.records[idx]
            x = src_rec.x
            
            # Evaluate
            # (In a real system, this might be distributed/parallel)
            try:
                res = evaluate_candidate(x, ctx_hi)
                
                # specific checks? if res.G > 0, promotion "failed" to hold feasibility?
                # We store it regardless.
                
                new_rec = ArchiveRecord(
                    x=x,
                    f=res.F,
                    g=res.G,
                    fidelity=ctx_hi.fidelity,
                    seed=ctx_hi.seed,
                    diag=res.diag
                )
                new_bundle.add(new_rec)
                
            except Exception as e:
                print(f"Error promoting candidate {src_rec.x_hash}: {e}")
                
        return new_bundle
