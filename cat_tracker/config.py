"""
Configuration loader for cat-tracker.

config.yaml in the project root is the single source of truth.
"""

import os
import yaml


_DEFAULT_YAML = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))


def load_config(path=None):
    resolved = path or _DEFAULT_YAML
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"[CONFIG] Config file not found: {resolved}")
    print(f"[CONFIG] Loaded from {resolved}")
    with open(resolved, 'r') as f:
        return yaml.safe_load(f) or {}
