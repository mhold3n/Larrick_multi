"""Surrogate Model Registry.

Centralizes knowledge of where models live and which versions are active.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class ModelRegistry:
    def __init__(self, root_dir: str | Path = "models/surrogate_v1"):
        self.root_dir = Path(root_dir)
        
    def get_path(self, model_key: str) -> Optional[Path]:
        """Get path for a named model."""
        # Convention: key "gear" -> "model_gear.pkl"
        # or separate dirs?
        # Let's use separate files in root
        
        candidates = [
            self.root_dir / f"model_{model_key}.pkl",
            self.root_dir / f"{model_key}.pkl",
            self.root_dir / model_key / "model.pkl"
        ]
        
        for p in candidates:
            if p.exists():
                return p
        return None

    def list_models(self) -> list[str]:
        """List available models."""
        if not self.root_dir.exists():
            return []
        return [p.stem for p in self.root_dir.glob("*.pkl")]


# Global instance
_REGISTRY = ModelRegistry()

def get_model_path(key: str) -> Optional[Path]:
    return _REGISTRY.get_path(key)
