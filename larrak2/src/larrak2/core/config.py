"""Configuration management with pydantic and YAML support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ThermoConfig(BaseModel):
    """Thermodynamic model configuration."""

    n_points: int = Field(default=360, ge=36, le=3600)
    gamma: float = Field(default=1.4, ge=1.1, le=1.7)
    p_atm_kpa: float = Field(default=101.325, ge=50, le=200)
    t_atm_k: float = Field(default=300.0, ge=250, le=400)


class GearConfig(BaseModel):
    """Gear model configuration."""

    n_points: int = Field(default=360, ge=36, le=3600)
    friction_coeff: float = Field(default=0.05, ge=0.0, le=0.2)
    ratio_error_tol: float = Field(default=0.02, ge=0.001, le=0.1)


class OptimizationConfig(BaseModel):
    """Optimization settings."""

    pop_size: int = Field(default=64, ge=8, le=1000)
    n_gen: int = Field(default=100, ge=1, le=10000)
    seed: int = Field(default=42, ge=0)


class LarrakConfig(BaseModel):
    """Root configuration object."""

    thermo: ThermoConfig = Field(default_factory=ThermoConfig)
    gear: GearConfig = Field(default_factory=GearConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


def load_config(path: str | Path) -> LarrakConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        Parsed LarrakConfig object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return LarrakConfig.model_validate(data or {})


def save_config(config: LarrakConfig, path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(config.model_dump(), f, default_flow_style=False)


def default_config() -> LarrakConfig:
    """Return default configuration."""
    return LarrakConfig()


def merge_config(base: LarrakConfig, overrides: dict[str, Any]) -> LarrakConfig:
    """Merge overrides into base configuration.

    Args:
        base: Base configuration.
        overrides: Dictionary of override values.

    Returns:
        New configuration with overrides applied.
    """
    base_dict = base.model_dump()

    def deep_merge(d1: dict, d2: dict) -> dict:
        result = d1.copy()
        for k, v in d2.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    merged = deep_merge(base_dict, overrides)
    return LarrakConfig.model_validate(merged)
