"""
Configuration loader for cat-tracker.

Loads settings from a YAML file and merges with built-in defaults
so the system works out of the box with no config file present.
"""

import os
import yaml


DEFAULTS = {
    'camera': {
        'width': 640,
        'height': 480,
        'fps': 20.0,
    },
    'servo': {
        'enabled': True,
        'pan_channel': 0,
        'tilt_channel': 1,
        'pan_center': 60,
        'tilt_center': 90,
        'pan_min': 0,
        'pan_max': 180,
        'tilt_min': 60,
        'tilt_max': 120,
        'deadzone': 50,
        'max_step': 5,
        'patrol_step': 0.6,
    },
    'detection': {
        'model_path': 'yolo11s.onnx',
        'confidence_threshold': 0.15,
        'iou_threshold': 0.4,
    },
    'tracking': {
        'max_missed': 15,
        'min_hits': 3,
        'iou_threshold': 0.3,
    },
    'identification': {
        'profile_path': 'cat_profiles.json',
        'bins_h': 30,
        'bins_s': 32,
        'bins_v': 32,
        'hsv_weights': [0.7, 0.2, 0.1],
        'min_saturation': 20,
        'min_value': 20,
    },
    'logging': {
        'position_log_path': 'occupancy_log.csv',
    },
}


def _deep_merge(base, override):
    """Recursively merge *override* into a copy of *base*."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path=None):
    """
    Load configuration from a YAML file, merged over built-in defaults.

    Args:
        path: Path to a YAML config file. If *None* or the file does not
              exist, only built-in defaults are returned.

    Returns:
        dict with all configuration sections populated.
    """
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}
        return _deep_merge(DEFAULTS, user_cfg)

    return DEFAULTS.copy()
