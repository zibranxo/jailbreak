"""Configuration loader."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _merge_env(config: dict[str, Any], prefix: str = "SAFETY_") -> None:
    for section_key, section_val in config.items():
        if isinstance(section_val, dict):
            for key, val in section_val.items():
                env_key = f"{prefix}{section_key.upper()}_{key.upper()}"
                if env_key in os.environ:
                    env_val = os.environ[env_key]
                    if isinstance(val, bool):
                        config[section_key][key] = env_val.lower() in ("true", "1", "yes")
                    elif isinstance(val, int):
                        config[section_key][key] = int(env_val)
                    elif isinstance(val, float):
                        config[section_key][key] = float(env_val)
                    elif isinstance(val, list) and isinstance(env_val, str):
                        config[section_key][key] = [x.strip() for x in env_val.split(",") if x.strip()]
                    else:
                        config[section_key][key] = env_val
        else:
            env_key = f"{prefix}{section_key.upper()}"
            if env_key in os.environ:
                config[section_key] = os.environ[env_key]

def load_config(path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        path: Path to config.yaml. Defaults to config.yaml in project root.

    Returns:
        Nested dict of configuration values.
    """
    # Walk up from this file to find project root
    search = Path(__file__).resolve().parent.parent.parent
    if path is None:
        path = str(search / "config.yaml")
        
    # Load .env file
    env_path = search / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    with open(path) as f:
        config = yaml.safe_load(f)
        
    _merge_env(config)
    return config


def resolve_path(config: dict, key: str) -> Path:
    """Resolve a path from config relative to project root."""
    path_str = config.get("paths", {}).get(key, "")
    return Path(__file__).resolve().parent.parent.parent / path_str
