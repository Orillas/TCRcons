"""Layered configuration system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class Config:
    """Resolved configuration for a tcrconsensus run."""

    io: dict = field(default_factory=dict)
    profiling: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    methods: dict = field(default_factory=dict)
    consensus: dict = field(default_factory=dict)
    refinement: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    reporting: dict = field(default_factory=dict)

    # Raw resolved config for provenance
    _raw: dict = field(default_factory=dict, repr=False)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(
    user_yaml: Optional[str] = None,
    preset: Optional[str] = None,
    package_dir: Optional[str] = None,
) -> Config:
    """Load layered configuration.

    Layer order (later overrides earlier):
    1. default.yaml
    2. preset (e.g. 'high_purity', 'noise_robust')
    3. user YAML file
    """
    # Default config
    if package_dir is None:
        package_dir = str(Path(__file__).parent.parent.parent / "configs")
    default_path = Path(package_dir) / "default.yaml"

    config: dict = {}
    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}

    # Preset overrides — subset of default config
    presets = {
        "high_purity": {
            "consensus": {
                "balanced": {"threshold": 0.5},
            },
            "selection": {
                "default_objective": "high_purity",
            },
        },
        "noise_robust": {
            "consensus": {
                "balanced": {"threshold": 0.4},
            },
        },
        "fast_screening": {
            "selection": {
                "max_methods": 2,
            },
        },
    }
    if preset and preset in presets:
        config = _deep_merge(config, presets[preset])

    # User overrides
    if user_yaml:
        user_path = Path(user_yaml)
        if user_path.exists():
            with open(user_path) as f:
                user_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, user_config)

    return Config(
        io=config.get("io", {}),
        profiling=config.get("profiling", {}),
        selection=config.get("selection", {}),
        methods=config.get("methods", {}),
        consensus=config.get("consensus", {}),
        refinement=config.get("refinement", {}),
        evaluation=config.get("evaluation", {}),
        reporting=config.get("reporting", {}),
        _raw=config,
    )


def config_to_dict(cfg: Config) -> dict:
    """Serialize config back to dict for persistence."""
    return cfg._raw.copy()
