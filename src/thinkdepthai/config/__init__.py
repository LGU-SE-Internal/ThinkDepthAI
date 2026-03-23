"""Configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .schema import AgentConfig, ModelConfig


def _resolve_env_vars(data: Any) -> Any:
    """Resolve ${oc.env:VAR} and ${VAR} patterns in YAML values."""
    if isinstance(data, str):
        # Handle Hydra-style ${oc.env:VAR} and ${oc.env:VAR,default}
        import re

        def _replace_oc_env(match):
            var_expr = match.group(1)
            if "," in var_expr:
                var_name, default = var_expr.split(",", 1)
                return os.getenv(var_name.strip(), default.strip())
            return os.getenv(var_expr.strip(), "")

        data = re.sub(r"\$\{oc\.env:([^}]+)\}", _replace_oc_env, data)

        # Handle simple ${VAR} and ${VAR:default}
        def _replace_simple(match):
            var_expr = match.group(1)
            if ":" in var_expr:
                var_name, default = var_expr.split(":", 1)
                return os.getenv(var_name.strip(), default.strip())
            return os.getenv(var_expr.strip(), "")

        data = re.sub(r"\$\{([^}]+)\}", _replace_simple, data)
        return data
    elif isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML config file with env var resolution."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return _resolve_env_vars(data)


def _get_default_config_dir() -> Path:
    """Get the default config directory (project root config/)."""
    # Try to find config/ relative to cwd first, then relative to package
    cwd_config = Path.cwd() / "config"
    if cwd_config.exists():
        return cwd_config
    return Path(__file__).parent.parent.parent.parent / "config"


def load_agent_config(config_path: str | Path | None = None) -> AgentConfig:
    """Load an AgentConfig from a YAML file.

    If no path given, loads the default config/agent/base.yaml.
    Also merges model config if referenced.
    """
    config_dir = _get_default_config_dir()

    if config_path is None:
        config_path = config_dir / "agent" / "base.yaml"
    elif isinstance(config_path, str) and not Path(config_path).is_absolute():
        config_path = config_dir / "agent" / config_path
        if not Path(config_path).exists() and not str(config_path).endswith(".yaml"):
            config_path = Path(str(config_path) + ".yaml")

    agent_data = load_yaml_config(config_path)

    # Remove Hydra-specific keys
    agent_data.pop("defaults", None)

    # Load and merge model config if not inline
    if "model" not in agent_data or not agent_data.get("model"):
        model_path = config_dir / "model" / "base.yaml"
        if model_path.exists():
            model_data = load_yaml_config(model_path)
            agent_data["model"] = model_data

    return AgentConfig(**agent_data)


def load_model_config(config_path: str | Path | None = None) -> ModelConfig:
    """Load a ModelConfig from a YAML file."""
    config_dir = _get_default_config_dir()
    if config_path is None:
        config_path = config_dir / "model" / "base.yaml"
    data = load_yaml_config(config_path)
    return ModelConfig(**data)
