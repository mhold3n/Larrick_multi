"""Strict archive management for multi-fidelity results.

Single source of truth for:
- Candidate storage (X, F, G)
- Fidelity metadata
- Version tracking
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from larrak2.core.constants import (
    MODEL_VERSION_THERMO_V1,
    MODEL_VERSION_GEAR_V1,
    N_THETA,
)

# Encoding version should ideally come from core.encoding
ENCODING_VERSION = "0.1"


@dataclass
class ArchiveRecord:
    """Single candidate record."""
    x: np.ndarray  # Decision vector
    f: np.ndarray  # Objectives
    g: np.ndarray  # Constraints
    fidelity: int
    seed: int
    x_hash: str = field(init=False)
    
    # Optional diagnostics/metadata
    diag: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Deterministic hash of decision vector (for tie-breaking/dedup)
        # Rounding to avoids float jitter issues if loaded from text, 
        # but here we rely on binary exactness or reasonable precision
        self.x_hash = hashlib.sha256(self.x.tobytes()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize mostly for debugging, actual storage via arrays."""
        return {
            "x": self.x.tolist(),
            "f": self.f.tolist(),
            "g": self.g.tolist(),
            "fidelity": self.fidelity,
            "seed": self.seed,
            "x_hash": self.x_hash,
            "diag": self.diag,
        }


@dataclass
class ArchiveBundle:
    """Container for a set of results."""
    records: list[ArchiveRecord] = field(default_factory=list)
    
    def add(self, record: ArchiveRecord):
        self.records.append(record)
        
    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract X, F, G arrays."""
        if not self.records:
            return np.array([]), np.array([]), np.array([])
            
        X = np.array([r.x for r in self.records])
        F = np.array([r.f for r in self.records])
        G = np.array([r.g for r in self.records])
        return X, F, G

    def get_hashes(self) -> list[str]:
        return [r.x_hash for r in self.records]

    def __len__(self):
        return len(self.records)


def save_meta(outdir: Path, extra_meta: dict[str, Any] = None):
    """Save rigorous metadata."""
    meta = {
        "timestamp": time.time(),
        "constants": {
            "N_THETA": N_THETA,
            "MODEL_VERSION_THERMO_V1": MODEL_VERSION_THERMO_V1,
            "MODEL_VERSION_GEAR_V1": MODEL_VERSION_GEAR_V1,
            "ENCODING_VERSION": ENCODING_VERSION,
        },
        # Placeholder / To be filled by caller
        "commit_hash": "unknown", 
    }
    if extra_meta:
        meta.update(extra_meta)
        
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def save_npz(outdir: Path, bundle: ArchiveBundle, stage_name: str, extra_arrays: dict = None):
    """Save bundle to numpy archive."""
    X, F, G = bundle.to_arrays()
    hashes = np.array(bundle.get_hashes())
    
    # Basic arrays
    data = {
        "X": X,
        "F": F,
        "G": G,
        "hashes": hashes,
    }
    
    # Store fidelities and seeds if mixed?
    # Usually a stage is uniform fidelity, but bundle might serve history.
    # We'll validly assume uniform for now or store arrays
    fids = np.array([r.fidelity for r in bundle.records])
    data["fidelity"] = fids
    
    if extra_arrays:
        data.update(extra_arrays)
        
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / f"archive_{stage_name}.npz", **data)


def load_npz(path: Path) -> ArchiveBundle:
    """Load bundle from numpy archive."""
    bundle = ArchiveBundle()
    if not path.exists():
        return bundle
        
    with np.load(path, allow_pickle=True) as data:
        X = data["X"]
        F = data["F"]
        G = data["G"]
        fids = data["fidelity"]
        
        # We might not save seed/diag in npz for bulk efficiency, 
        # or we need to serialize them separately or in object array.
        # For strict deterministic flows, we just need X, F, G.
        
        for i in range(len(X)):
            rec = ArchiveRecord(
                x=X[i],
                f=F[i],
                g=G[i],
                fidelity=int(fids[i]),
                seed=0, # Lost in simplified NPZ, trivial for analysis
            )
            bundle.add(rec)
            
    return bundle
