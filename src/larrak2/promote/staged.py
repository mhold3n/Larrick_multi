"""Staged Workflow Orchestrator.

Encapsulates strict multi-stage workflow:
Stage 1: Exploration (Lo-Fi NSGA-III)
Stage 2: Promotion (Selection + Hi-Fi Evaluation)
Stage 3: Refinement (Hi-Fi NSGA-III seeded with Stage 2) (Optional)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from larrak2.adapters.pymoo_problem import ParetoProblem
from larrak2.core.types import EvalContext
from larrak2.promote.archive import ArchiveBundle, ArchiveRecord, save_meta, save_npz
from larrak2.promote.manager import PromotionManager


class StagedWorkflow:
    """Orchestrates strict staged optimization."""

    def __init__(self, outdir: Path, rpm: float, torque: float, seed: int):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.rpm = rpm
        self.torque = torque
        self.seed = seed
        self.pm = PromotionManager()

    def run_stage1(self, pop_size: int, n_gen: int) -> ArchiveBundle:
        """Stage 1: Exploration at Fidelity 1."""
        print("=== Stage 1: Exploration (Fidelity 1) ===")

        # Enforce strict determinism across repeated runs in-process.
        np.random.seed(self.seed)
        random.seed(self.seed)

        ctx = EvalContext(rpm=self.rpm, torque=self.torque, fidelity=1, seed=self.seed)
        problem = ParetoProblem(ctx)

        # NSGA-III Setup
        # Assuming 3 objectives for Larrak2
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

        # Ensure pop size >= ref dirs
        actual_pop = max(pop_size, len(ref_dirs))

        algorithm = NSGA3(
            pop_size=actual_pop,
            ref_dirs=ref_dirs,
            prob_neighbor_mating=0.7,
        )

        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", n_gen),
            seed=self.seed,
            verbose=True,
        )

        # Create Archive
        bundle = ArchiveBundle()
        X = res.pop.get("X")
        F = res.pop.get("F")
        G = res.pop.get("G")

        for i in range(len(X)):
            rec = ArchiveRecord(x=X[i], f=F[i], g=G[i], fidelity=1, seed=self.seed)
            bundle.add(rec)

        save_npz(
            self.outdir, bundle, "stage1", extra_arrays={"n_evals": np.array([problem.n_evals])}
        )
        save_meta(self.outdir, {"stage": 1, "pop": actual_pop, "gen": n_gen})
        (self.outdir / "stage1").mkdir(exist_ok=True)

        return bundle

    def run_promotion(self, archive: ArchiveBundle, k: int) -> ArchiveBundle:
        """Stage 2: Promotion to Fidelity 2."""
        print(f"=== Stage 2: Promotion ({k} candidates) ===")

        # 1. Select
        indices = self.pm.select_candidates(archive, k)

        # 2. Promote (Evaluate)
        # Context for Fidelity 2
        # Use same seed to ensure determinism of physics
        ctx_hi = EvalContext(rpm=self.rpm, torque=self.torque, fidelity=2, seed=self.seed)

        new_bundle = self.pm.promote_candidates(archive, indices, ctx_hi)

        save_npz(self.outdir, new_bundle, "stage2")
        (self.outdir / "stage2").mkdir(exist_ok=True)

        return new_bundle

    def run_stage3(self, seed_bundle: ArchiveBundle, pop_size: int, n_gen: int) -> ArchiveBundle:
        """Stage 3: Refinement at Fidelity 2."""
        # Typically we re-seed NSGA-III with the High-Fidelity points
        print("=== Stage 3: Refinement (Fidelity 2) ===")

        # Enforce strict determinism across repeated runs in-process.
        np.random.seed(self.seed)
        random.seed(self.seed)

        ctx = EvalContext(rpm=self.rpm, torque=self.torque, fidelity=2, seed=self.seed)
        problem = ParetoProblem(ctx)

        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        actual_pop = max(pop_size, len(ref_dirs))

        # Initial Population Injection
        # We need to format the seed bundle as an Initial Population for Pymoo
        # Pymoo allows sampling to be a population object

        X_seed, _, _ = seed_bundle.to_arrays()
        # Note: We don't necessarily trust the F/G from Stage 2 if we want the algorithm
        # to re-evaluate or if we want to ensure consistency.
        # But if we provide X, NSGA-III will evaluate them in the first generation.
        # This is safer.

        # Need to fill the rest of the population if seed < pop
        # NSGA3 will handle mixed initialization if we provide a Sampling object
        # that returns the matrix.

        # Custom Initialization
        # Manually construct X with seeds + random fill
        # Use legacy RandomState to avoid cross-test interference with default_rng.
        rng = np.random.RandomState(self.seed)
        X_init = rng.rand(actual_pop, problem.n_var)
        X_init = problem.xl + X_init * (problem.xu - problem.xl)

        n_seed = len(X_seed)
        if n_seed > 0:
            to_insert = min(n_seed, actual_pop)
            X_init[:to_insert] = X_seed[:to_insert]

        algorithm = NSGA3(
            pop_size=actual_pop,
            ref_dirs=ref_dirs,
            sampling=X_init,
            prob_neighbor_mating=0.7,
        )

        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", n_gen),
            seed=self.seed,
            verbose=True,
        )

        # Create Archive
        bundle = ArchiveBundle()
        X = res.pop.get("X")
        F = res.pop.get("F")
        G = res.pop.get("G")

        for i in range(len(X)):
            rec = ArchiveRecord(x=X[i], f=F[i], g=G[i], fidelity=2, seed=self.seed)
            bundle.add(rec)

        save_npz(
            self.outdir, bundle, "stage3", extra_arrays={"n_evals": np.array([problem.n_evals])}
        )
        (self.outdir / "stage3").mkdir(exist_ok=True)

        return bundle
