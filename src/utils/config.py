"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).resolve().parents[2]
