"""Promotion Orchestrator.

Manages the valid transition of candidates between fidelity stages.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from larrak2.core.evaluator import evaluate_candidate
from larrak2.core.types import EvalContext
from larrak2.promote.archive import ArchiveBundle, ArchiveRecord
from larrak2.promote.selectors import (
    select_hybrid,
    select_k_best_ref_dirs,
    select_strict_nsga3,
)


class PromotionManager:
    """Manager for promotion logic (dict archive + bundle workflows)."""

    def __init__(self, root_dir: Path | None = None):
        self.root_dir = Path(root_dir) if root_dir else None
        # Hash -> record dict for lightweight workflows/tests
        self.archive: dict[str, dict] = {}

    @staticmethod
    def _hash_x(x: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()[:16]

    # --- Simple archive API used in tests ---
    def ingest_population(self, X: np.ndarray, F: np.ndarray, G: np.ndarray, fidelity: int):
        for x, f, g in zip(X, F, G):
            h = self._hash_x(x)
            rec = self.archive.setdefault(
                h,
                {
                    "x": np.asarray(x),
                },
            )
            rec[f"F_fid{fidelity}"] = np.asarray(f)
            rec[f"G_fid{fidelity}"] = np.asarray(g)
            rec["fidelity"] = fidelity

    def save_archive(self):
        # Optional persistence for compatibility; keep lightweight
        if not self.root_dir:
            return
        self.root_dir.mkdir(parents=True, exist_ok=True)
        # Save minimal NPZ
        hashes = list(self.archive.keys())
        X = np.array([self.archive[h]["x"] for h in hashes])
        # Pick lowest fidelity F present as baseline
        any_key = next(iter(self.archive.values()))
        fid_keys = [k for k in any_key.keys() if k.startswith("F_fid")]
        if fid_keys:
            F = np.array([self.archive[h][fid_keys[0]] for h in hashes])
        else:
            F = np.zeros((len(hashes), 0))
        np.savez_compressed(self.root_dir / "promotion_archive.npz", X=X, F=F, hashes=hashes)

    def select_for_promotion(self, fidelity_source: int, n_promote: int) -> list[str]:
        # Collect candidates with required fidelity data
        candidates = [
            (h, rec) for h, rec in self.archive.items() if f"F_fid{fidelity_source}" in rec
        ]
        if not candidates:
            return []
        hashes, recs = zip(*candidates)
        F = np.vstack([rec[f"F_fid{fidelity_source}"] for rec in recs])
        idx = select_k_best_ref_dirs(F, k=n_promote)
        return [hashes[i] for i in idx]

    def promote_and_evaluate(self, promoted_hashes: list[str], ctx_hi: EvalContext) -> int:
        n_success = 0
        for h in promoted_hashes:
            rec = self.archive[h]
            x = rec["x"]
            res = evaluate_candidate(x, ctx_hi)
            rec[f"F_fid{ctx_hi.fidelity}"] = res.F
            rec[f"G_fid{ctx_hi.fidelity}"] = res.G
            rec[f"versions_fid{ctx_hi.fidelity}"] = res.diag.get("versions", {})
            n_success += 1
        return n_success

    def get_pareto_front(self, fidelity: int):
        hashes = []
        X_list = []
        F_list = []
        for h, rec in self.archive.items():
            key = f"F_fid{fidelity}"
            if key not in rec:
                continue
            hashes.append(h)
            X_list.append(rec["x"])
            F_list.append(rec[key])
        if not hashes:
            return np.array([]), np.array([]), []
        return np.vstack(X_list), np.vstack(F_list), hashes

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

        uncertainty_arr = np.array(uncertainty_scores)

        # Dispatch
        if strategy == "hybrid":
            ratio = kwargs.get("ratio_explore", 0.2)
            selected_local_indices = select_hybrid(
                F_feas, hashes_feas, k, uncertainty_arr, ratio_explore=ratio
            )
        else:
            # Default to NSGA3
            selected_local_indices = select_strict_nsga3(F_feas, hashes_feas, k)

        # Map back to global archive indices
        return [int(feasible_indices[i]) for i in selected_local_indices]

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
                    x=x, f=res.F, g=res.G, fidelity=ctx_hi.fidelity, seed=ctx_hi.seed, diag=res.diag
                )
                new_bundle.add(new_rec)

            except Exception as e:
                print(f"Error promoting candidate {src_rec.x_hash}: {e}")

        return new_bundle
