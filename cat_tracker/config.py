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
        'fps': 15.0,
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
        'camera_hfov': 66,
        'camera_vfov': 49,
    },
    'detection': {
        'model_path': 'yolo11s.onnx',
        'confidence_threshold': 0.15,
        'iou_threshold': 0.4,
    },
    'tracking': {
        'max_missed': 45,
        'min_hits': 3,
        'iou_threshold': 0.3,
        'look_ahead': 2,
        'inference_every': 3,
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


_DEFAULT_YAML = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))


def load_config(path=None):
    """
    Load configuration from a YAML file, merged over built-in defaults.

    Looks for config.yaml in the project root automatically if no path given.
    """
    resolved = path or _DEFAULT_YAML
    if resolved and os.path.exists(resolved):
        print(f"[CONFIG] Loaded from {resolved}")
        with open(resolved, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}
        return _deep_merge(DEFAULTS, user_cfg)

    print(f"[CONFIG] No config file found at {resolved}, using defaults")
    return DEFAULTS.copy()
