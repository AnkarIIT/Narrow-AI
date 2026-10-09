"""Configuration utilities for NarrowAI."""
from __future__ import annotations
import os
from typing import Any


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file or environment variables.

    Falls back gracefully when PyYAML is not installed.
    """
    config: dict[str, Any] = {}

    # Environment variable overrides
    env_map = {
        "OPENAI_API_KEY": "api.openai_key",
        "OLLAMA_URL": "api.ollama_url",
        "LOG_LEVEL": "logging.level",
    }
    for env_key, config_path in env_map.items():
        val = os.environ.get(env_key)
        if val:
            _set_nested(config, config_path, val)

    # File config (optional)
    if path and os.path.exists(path):
        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            _deep_merge(config, file_config)
        except ImportError:
            pass

    return config


def _set_nested(d: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
