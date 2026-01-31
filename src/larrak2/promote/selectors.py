"""Strict selection logic for promotion.

Implements the user-specified selection strategy:
1. Non-dominated Sorting.
2. Reference Direction Diversity (NSGA-III style).
3. Hash-based tie-breaking for determinism.
"""

from __future__ import annotations

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions


def select_strict_nsga3(
    F: np.ndarray,
    hashes: list[str] | np.ndarray,
    k: int,
    ref_dirs: np.ndarray | None = None,
) -> np.ndarray:
    """Select k candidates using strict NSGA-III logic.

    Args:
        F: Objective values (N x M), minimization assumed.
        hashes: Candidate hashes (N) for tie-breaking.
        k: Number of candidates to select.
        ref_dirs: Optional reference directions. If None, generated.

    Returns:
        Indices of selected candidates.
    """
    n_points, n_obj = F.shape
    if k >= n_points:
        return np.arange(n_points)

    hashes = np.array(hashes)

    # 1. Non-dominated Sorting
    nds = NonDominatedSorting()
    fronts = nds.do(F)

    selected_indices = []

    # Add complete fronts until we overshoot
    for front in fronts:
        if len(selected_indices) + len(front) <= k:
            selected_indices.extend(front)
        else:
            # Critical front needs splitting
            n_needed = k - len(selected_indices)

            # Candidates to choose from (the critical front)
            candidates = np.array(front)

            # We assume "selected_indices" (from better fronts) are ALREADY selected
            # and they influence niching.
            # In standard NSGA-III, we consider all selected so far + critical front
            # to determine niching counts.

            chosen_from_front = _survive_ref_dirs(
                F,
                np.array(selected_indices, dtype=int),
                candidates,
                n_needed,
                hashes,
                n_obj,
                ref_dirs,
            )
            selected_indices.extend(chosen_from_front)
            break

    return np.array(selected_indices, dtype=int)


def select_k_best_ref_dirs(
    F: np.ndarray,
    k: int,
    ref_dirs: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministic selection of k candidates guided by reference directions.

    Simplified diversity-aware selector used by PromotionManager tests.
    """
    n_points, n_obj = F.shape
    if k >= n_points:
        return np.arange(n_points)

    # Generate ref dirs if not provided
    if ref_dirs is None:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=3)

    # Normalize objectives
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    denom = np.where((f_max - f_min) < 1e-9, 1.0, f_max - f_min)
    F_norm = (F - f_min) / denom

    # Associate each point to closest ref dir (perpendicular distance)
    refs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    dot_prod = F_norm @ refs_norm.T
    p_sq = np.sum(F_norm**2, axis=1)[:, None]
    perp_dist = np.sqrt(np.maximum(p_sq - dot_prod**2, 0.0))
    assoc = np.argmin(perp_dist, axis=1)

    # For each ref dir, pick the best (smallest norm) candidate
    selected = []
    for rd in range(len(ref_dirs)):
        candidates = np.where(assoc == rd)[0]
        if len(candidates) == 0:
            continue
        # Choose deterministically by total objective sum then index
        best_idx = min(candidates, key=lambda i: (np.sum(F_norm[i]), i))
        selected.append(best_idx)
        if len(selected) == k:
            break

    # If still short, fill deterministically by remaining lowest norm
    if len(selected) < k:
        remaining = [i for i in range(n_points) if i not in selected]
        remaining_sorted = sorted(remaining, key=lambda i: (np.sum(F_norm[i]), i))
        selected.extend(remaining_sorted[: k - len(selected)])

    return np.array(selected[:k], dtype=int)


def _survive_ref_dirs(
    F: np.ndarray,
    current_indices: np.ndarray,
    candidate_indices: np.ndarray,
    n_needed: int,
    hashes: np.ndarray,
    n_obj: int,
    ref_dirs: np.ndarray | None,
) -> np.ndarray:
    """Perform reference direction survival on the critical front."""

    # 1. Prepare Reference Directions
    if ref_dirs is None:
        # Heuristic: roughly matches pymoo default for small dims
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)

    # 2. Normalize Objectives
    # We consider ALL points involved (current + candidates) to define the specific range
    # Or typically ideally we track the ideal point globally.
    # Here using the local batch range is safer for promotion stability locally.

    all_indices = (
        np.concatenate([current_indices, candidate_indices])
        if len(current_indices) > 0
        else candidate_indices
    )
    F_all = F[all_indices]

    f_min = np.min(F_all, axis=0)
    f_max = np.max(F_all, axis=0)
    denom = f_max - f_min
    denom[denom < 1e-6] = 1.0

    # Normalize ALL considered points
    # But we only care about association for current and candidates

    # 3. Associate Members
    # For every point in (current + candidates), find closest ref dir
    # Niches: count how many CURRENTLY selected are in each niche

    # We need to map global indices to normalized F
    # Let's verify association for current set first to get niche counts

    F_norm_all = (F - f_min) / denom

    niche_counts = np.zeros(len(ref_dirs), dtype=int)

    # Pre-calculate distances for association
    # We only need association for 'current' and 'candidate'

    def associate(indices):
        if len(indices) == 0:
            return {}, {}

        vals = F_norm_all[indices]
        # Distance to lines defined by ref_dirs
        # Just use cosine similarity for speed/robustness as 'closeness' to direction
        # Dist to line is robust.

        # d(p, u) = || p - (p.u)u || if u is unit
        # p relative to ideal point (which is 0 after norm)

        # Normalize ref dirs
        refs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)

        # Calculate distances:
        # For each point p, dist to each ref r is perp distance
        # p_proj = (p . r) * r
        # dist = || p - p_proj ||

        # Matrix op:
        # Dot: (N, M) @ (M, R) -> (N, R)
        dot_prod = np.dot(vals, refs_norm.T)

        # We clamp dot product to be positive (direction match)
        # Actually ref dirs are in quadrant 1, points in quadrant 1.

        # Proj vectors: (N, R, 1) * (1, R, M) -> (N, R, M) ? Too big.
        # Just find argmax dot product / cosine for now?
        # Standard NSGA-III minimizes perpendicular distance.

        # Let's use simplified: closest Euclidean distance on the unit sphere (cosine sim)
        # This matches direction best.

        # Normalize P
        p_norm = np.linalg.norm(vals, axis=1, keepdims=True)
        p_norm[p_norm < 1e-9] = 1e-9
        p_unit = vals / p_norm

        cosine = np.dot(p_unit, refs_norm.T)  # (N, R)
        np.argmax(cosine, axis=1)  # (N,)
        np.linalg.norm(vals - (p_norm * p_unit), axis=1)  # wait, math.

        # Correct Perp Dist:
        # dist_sq = ||p||^2 - (p.u)^2  (if p.u > 0)
        p_sq = np.sum(vals**2, axis=1)[:, None]  # (N, 1)
        dot_sq = dot_prod**2  # (N, R)

        perp_dist_sq = p_sq - dot_sq
        perp_dist_sq[perp_dist_sq < 0] = 0
        perp_dist = np.sqrt(perp_dist_sq)

        # Associate to ref with MIN perp distance
        assoc_idx = np.argmin(perp_dist, axis=1)
        assoc_dist = perp_dist[np.arange(len(indices)), assoc_idx]

        return assoc_idx, assoc_dist

    # Count niches for already selected
    if len(current_indices) > 0:
        curr_assocs, _ = associate(current_indices)
        for r_idx in curr_assocs:
            niche_counts[r_idx] += 1

    # Assoicate candidates
    cand_assocs, cand_dists = associate(candidate_indices)

    # Group candidates by niche
    candidates_by_niche = {}
    for local_i, r_idx in enumerate(cand_assocs):
        if r_idx not in candidates_by_niche:
            candidates_by_niche[r_idx] = []
        candidates_by_niche[r_idx].append(local_i)

    final_selection = []

    # 4. Niching Loop
    # Until we fill k
    for _ in range(n_needed):
        # Find ref dirs with Min count
        # (Exclude those with no candidates available)
        valid_refs = [
            r
            for r in range(len(ref_dirs))
            if r in candidates_by_niche and len(candidates_by_niche[r]) > 0
        ]

        if not valid_refs:
            break  # No candidates left

        min_count = min(niche_counts[r] for r in valid_refs)
        best_refs = [r for r in valid_refs if niche_counts[r] == min_count]

        # Pick one ref dir (tie break arbitrary? standard usually random, we use deterministic)
        # Tie break by ref_dir index (implicit preference for low index directions)
        target_ref = best_refs[0]

        # Pick candidate from this ref dir
        # If count == 0 ("Sparse"), pick closest (min perp dist)
        # If count > 0, pick randomly? Or typically we prioritize convergence.
        # NSGA-III typically prioritizes closest perpendicular distance for population maintenance.

        # Get candidates in this niche
        c_local_indices = candidates_by_niche[target_ref]

        # Select best candidate
        # Primary key: Distance (Min)
        # Tie break: Hash

        best_cand_local = -1
        min_dist = 1e9
        best_hash = "z" * 64

        for idx_local in c_local_indices:
            d = cand_dists[idx_local]
            h = hashes[candidate_indices[idx_local]]

            # Tie breaking logic
            if d < min_dist - 1e-9:  # Distinctly better
                min_dist = d
                best_cand_local = idx_local
                best_hash = h
            elif abs(d - min_dist) <= 1e-9:  # Roughly equal
                if h < best_hash:  # Strict Hash Tie Break
                    min_dist = d
                    best_cand_local = idx_local
                    best_hash = h

        # Add to selection
        global_idx = candidate_indices[best_cand_local]
        final_selection.append(global_idx)

        # Update counts
        niche_counts[target_ref] += 1

        # Remove from pool
        c_local_indices.remove(best_cand_local)

    return np.array(final_selection, dtype=int)


def select_most_uncertain(
    hashes: list[str] | np.ndarray,
    k: int,
    uncertainty_scores: np.ndarray,
    candidates_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Select k candidates with highest uncertainty.

    Args:
        hashes: Candidate hashes (N).
        k: Number to select.
        uncertainty_scores: Uncertainty metric (N).
        candidates_indices: Optional subset of indices to consider.

    Returns:
        Indices of selected candidates.
    """
    hashes = np.array(hashes)
    n_points = len(hashes)

    if candidates_indices is None:
        candidates_indices = np.arange(n_points)

    if k >= len(candidates_indices):
        return candidates_indices

    # Filter subset
    subset_hashes = hashes[candidates_indices]
    subset_scores = uncertainty_scores[candidates_indices]

    # Sort by Score (Desc) -> Hash (Asc)
    items = []
    for i, local_idx in enumerate(candidates_indices):
        items.append((local_idx, subset_scores[i], subset_hashes[i]))

    # Sort: Higher score is better (-score), Lower hash is better (hash)
    items.sort(key=lambda x: (-x[1], x[2]))

    selected = [x[0] for x in items[:k]]
    return np.array(selected, dtype=int)


def select_hybrid(
    F: np.ndarray,
    hashes: list[str] | np.ndarray,
    k: int,
    uncertainty_scores: np.ndarray,
    ratio_explore: float = 0.2,
    ref_dirs: np.ndarray | None = None,
) -> np.ndarray:
    """Select k candidates using Exploration-Exploitation hybrid.

    Exploitation: Strict NSGA-III (Pareto).
    Exploration: Max Uncertainty.

    Args:
        F: Objectives.
        hashes: Hashes.
        k: Total to select.
        uncertainty_scores: Uncertainty scores.
        ratio_explore: Fraction of k allocated to exploration.
        ref_dirs: NSGA-III ref dirs.

    Returns:
        Indices of selected candidates.
    """
    n_points = F.shape[0]
    if k >= n_points:
        return np.arange(n_points)

    # 1. Determine Allocation
    k_explore = int(np.round(k * ratio_explore))
    k_exploit = k - k_explore

    # Safety: ensure valid counts
    if k_exploit < 0:
        k_exploit = 0
    if k_explore < 0:
        k_explore = 0

    # 2. Exploitation Phase
    exploit_indices = np.array([], dtype=int)
    if k_exploit > 0:
        exploit_indices = select_strict_nsga3(F, hashes, k_exploit, ref_dirs)

    # 3. Exploration Phase
    explore_indices = np.array([], dtype=int)
    if k_explore > 0:
        # Determine remaining candidates
        if len(exploit_indices) > 0:
            mask = np.ones(n_points, dtype=bool)
            mask[exploit_indices] = False
            remaining = np.where(mask)[0]
        else:
            remaining = np.arange(n_points)

        if len(remaining) > 0:
            # Select from remaining based on uncertainty
            explore_indices = select_most_uncertain(
                hashes,
                min(k_explore, len(remaining)),
                uncertainty_scores,
                candidates_indices=remaining,
            )

    # Combine
    combined = np.concatenate([exploit_indices, explore_indices])
    return np.unique(combined)  # Should be unique, but safe.
