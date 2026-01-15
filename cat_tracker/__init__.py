"""
Cat Tracker - Multi-cat detection, tracking, and identification.
"""

from .multi_tracker import MultiTracker
from .tracker import Track
from .prefix_colors import ColorHistogramExtractor, ColorHistogramIdentifier
from .detection import (
    load_yolo_model,
    parse_yolo_output,
    preprocess_frame,
    TRACK_COLORS,
    CAT_CLASS_ID,
    DEFAULT_MODEL_PATH,
)

__all__ = [
    'MultiTracker',
    'Track',
    'ColorHistogramExtractor',
    'ColorHistogramIdentifier',
    'load_yolo_model',
    'parse_yolo_output',
    'preprocess_frame',
    'TRACK_COLORS',
    'CAT_CLASS_ID',
    'DEFAULT_MODEL_PATH',
]
