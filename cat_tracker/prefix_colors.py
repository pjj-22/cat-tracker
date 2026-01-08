"""
Prefix colors to compute the area color.
Used to determine histogram of color different to standard.

This implementation replaces full prefix-sum histograms with
Rgeion of interest based masked Hue Saturation Value histograms for speed and memory efficiency.
"""

import cv2
import numpy as np
import json
import os
from cat_tracker.utils import clamp_bbox_to_image


class ColorHistogramExtractor:
    """
    Extract HSV color histograms from rectangular image regions.

    Histograms are normalized and masked to suppress background,
    shadows, and low-saturation pixels.
    """

    def __init__(
        self,
        bins_h=30,
        bins_s=32,
        bins_v=32,
        min_saturation=20,
        min_value=20
    ):
        """
        Initialize histogram extractor.

        Args:
            bins_h: Number of bins for Hue (0-180)
            bins_s: Number of bins for Saturation (0-255)
            bins_v: Number of bins for Value (0-255)
            min_saturation: Ignore pixels below this saturation
            min_value: Ignore pixels below this brightness
        """
        self.bins_h = bins_h
        self.bins_s = bins_s
        self.bins_v = bins_v
        self.min_saturation = min_saturation
        self.min_value = min_value

    def extract(self, frame_rgb, bbox):
        """
        Compute normalized HSV histograms for a bounding box.

        Args:
            frame_rgb: RGB image
            bbox: (x1, y1, x2, y2)

        Returns:
            hist_h, hist_s, hist_v:
                Normalized histograms for HSV channels,
                or (None, None, None) if region is invalid
        """
        x1, y1, x2, y2 = clamp_bbox_to_image(bbox, w, h)

        roi = frame_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None, None

        # Convert ROI to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        # Mask out background, shadows, and low-color pixels
        mask = (
            (hsv[:, :, 1] >= self.min_saturation) &
            (hsv[:, :, 2] >= self.min_value)
        ).astype(np.uint8)

        # Require minimum number of valid pixels
        if np.count_nonzero(mask) < 10:
            return None, None, None

        # Compute histograms for each channel
        hist_h = cv2.calcHist(
            [hsv], [0], mask,
            [self.bins_h], [0, 180]
        )
        hist_s = cv2.calcHist(
            [hsv], [1], mask,
            [self.bins_s], [0, 256]
        )
        hist_v = cv2.calcHist(
            [hsv], [2], mask,
            [self.bins_v], [0, 256]
        )

        # Flatten and normalize histograms
        hist_h = hist_h.flatten().astype(np.float32)
        hist_s = hist_s.flatten().astype(np.float32)
        hist_v = hist_v.flatten().astype(np.float32)

        hist_h /= np.sum(hist_h)
        hist_s /= np.sum(hist_s)
        hist_v /= np.sum(hist_v)

        return hist_h, hist_s, hist_v


class ColorHistogramIdentifier:
    """
    Identify cats by comparing HSV color histograms
    to previously learned color profiles.
    """

    def __init__(self, profile_path="cat_profiles.json"):
        """
        Load learned cat color profiles from disk.

        Args:
            profile_path: Path to JSON file containing profiles
        """
        self.profile_path = profile_path
        self.profiles = {}

        if os.path.exists(profile_path):
            self.load_profiles()

    def load_profiles(self):
        """
        Load histogram profiles from JSON file.
        """
        with open(self.profile_path, "r") as f:
            data = json.load(f)

        for cat_name, profile_data in data.items():
            self.profiles[cat_name] = {
                'hist_h': np.array(profile_data['hist_h'], dtype=np.float32),
                'hist_s': np.array(profile_data['hist_s'], dtype=np.float32),
                'hist_v': np.array(profile_data['hist_v'], dtype=np.float32),
                'sample_count': profile_data.get('sample_count', 1)
            }

    def save_profiles(self):
        """
        Save learned histogram profiles to disk.
        """
        data = {}
        for cat_name, profile in self.profiles.items():
            data[cat_name] = {
                'hist_h': profile['hist_h'].tolist(),
                'hist_s': profile['hist_s'].tolist(),
                'hist_v': profile['hist_v'].tolist(),
                'sample_count': profile['sample_count']
            }

        with open(self.profile_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_training_sample(self, cat_name, hist_h, hist_s, hist_v):
        """
        Add a training sample for a cat.

        If the cat already exists, this updates the profile
        using a running average.

        Args:
            cat_name: Name of the cat
            hist_h, hist_s, hist_v: Normalized HSV histograms
        """
        if cat_name not in self.profiles:
            self.profiles[cat_name] = {
                'hist_h': hist_h.copy(),
                'hist_s': hist_s.copy(),
                'hist_v': hist_v.copy(),
                'sample_count': 1
            }
            return

        profile = self.profiles[cat_name]
        n = profile['sample_count']

        profile['hist_h'] = (profile['hist_h'] * n + hist_h) / (n + 1)
        profile['hist_s'] = (profile['hist_s'] * n + hist_s) / (n + 1)
        profile['hist_v'] = (profile['hist_v'] * n + hist_v) / (n + 1)
        profile['sample_count'] = n + 1

    def identify(self, hist_h, hist_s, hist_v, threshold=0.35):
        """
        Identify cat by comparing histograms to learned profiles.

        Args:
            hist_h, hist_s, hist_v: HSV histograms
            threshold: Maximum allowed Bhattacharyya distance

        Returns:
            cat_name: Best matching cat name or "Unknown"
            confidence: Confidence score (0–1)
            debug_info: Distances to each profile
        """
        if not self.profiles:
            return "Unknown", 0.0, {}

        distances = {}
        for cat_name, profile in self.profiles.items():
            distance = self._bhattacharyya_distance(
                hist_h, hist_s, hist_v,
                profile['hist_h'],
                profile['hist_s'],
                profile['hist_v']
            )
            distances[cat_name] = distance

        best_cat = min(distances, key=distances.get)
        best_distance = distances[best_cat]

        if best_distance > threshold:
            return "Unknown", 1.0 - best_distance, distances

        confidence = max(0.0, 1.0 - best_distance)
        return best_cat, confidence, distances

    @staticmethod
    def _bhattacharyya_distance(h1_h, h1_s, h1_v, h2_h, h2_s, h2_v):
        """
        Compute Bhattacharyya distance between two HSV histograms.

        Returns:
            distance: 0 = identical, 1 = completely different
        """
        bc_h = np.sum(np.sqrt(h1_h * h2_h))
        bc_s = np.sum(np.sqrt(h1_s * h2_s))
        bc_v = np.sum(np.sqrt(h1_v * h2_v))

        bc = (bc_h + bc_s + bc_v) / 3.0
        bc = np.clip(bc, 0.0, 1.0)

        return np.sqrt(1.0 - bc)
