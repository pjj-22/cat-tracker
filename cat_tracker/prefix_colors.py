"""
Prefix colors to compute the area color.
Used to determine histogram of color different to standard.

This implementation replaces full prefix-sum histograms with
Region of interest based masked Hue Saturation Value histograms for speed and memory efficiency.
"""

import cv2
import numpy as np
import json
import os
from cat_tracker.utils import clamp_bbox_to_image


# WHY HSV INSTEAD OF RGB
#
# RGB mixes color and brightness together. An orange cat in bright light and
# the same cat in shadow have very different RGB values but similar Hue values.
# HSV separates them:
#   H (Hue)        0-180 in OpenCV. The actual color: orange, grey, black, etc.
#   S (Saturation) 0-255. How vivid the color is. A grey cat has low saturation.
#   V (Value)      0-255. Brightness. Same cat, dim room vs bright room.
#
# We weight hue at 70%, saturation at 20%, value at 10% in the distance formula.
# Hue is the most stable across lighting changes. Value shifts the most.

# BHATTACHARYYA DISTANCE
#
# Given two normalized histograms p and q (same length, sum to 1):
#
#   Bhattacharyya coefficient BC = sum(sqrt(p[i] * q[i]))
#
# BC is 1 if the distributions are identical, 0 if they share no bins.
# We convert to a distance: d = sqrt(1 - BC), so 0 = identical, 1 = different.
#
# We do this separately for H, S, V and then combine with weights:
#   BC_total = 0.7 * BC_h + 0.2 * BC_s + 0.1 * BC_v
#   distance = sqrt(1 - BC_total)
#
# The cat with the lowest distance is the best match. We always return a name
# (never "no match") because if profiles exist there's always a closest one.
# confidence = 1.0 - distance, so a perfect match is confidence 1.0.



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
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = clamp_bbox_to_image(bbox, w, h)

        roi = frame_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None, None

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        # Mask out background, shadows, and low-color pixels
        mask = (
            (hsv[:, :, 1] >= self.min_saturation) &
            (hsv[:, :, 2] >= self.min_value)
        ).astype(np.uint8)

        # Require minimum number of valid pixels
        if np.count_nonzero(mask) < 10:
            return None, None, None

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

        # Normalize with safety check for division by zero
        sum_h = np.sum(hist_h)
        sum_s = np.sum(hist_s)
        sum_v = np.sum(hist_v)

        hist_h = hist_h / sum_h if sum_h > 0 else np.zeros_like(hist_h)
        hist_s = hist_s / sum_s if sum_s > 0 else np.zeros_like(hist_s)
        hist_v = hist_v / sum_v if sum_v > 0 else np.zeros_like(hist_v)

        return hist_h, hist_s, hist_v


class ColorHistogramIdentifier:
    """
    Identify cats by comparing HSV color histograms
    to previously learned color profiles.
    """

    def __init__(self, profile_path="cat_profiles.json", hsv_weights=None):
        """
        Load learned cat color profiles from disk.

        Args:
            profile_path: Path to JSON file containing profiles
            hsv_weights: (weight_h, weight_s, weight_v) for Bhattacharyya
                         distance. Defaults to (0.7, 0.2, 0.1).
        """
        self.profile_path = profile_path
        self.hsv_weights = tuple(hsv_weights) if hsv_weights else (0.7, 0.2, 0.1)
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
                'sample_count': profile_data.get('sample_count', 1),
                'sources': profile_data.get('sources', [])
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
                'sample_count': profile['sample_count'],
                'sources': profile.get('sources', [])
            }

        with open(self.profile_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_training_sample(self, cat_name, hist_h, hist_s, hist_v, source_path=None):
        """
        Add a training sample for a cat.

        If the cat already exists, this updates the profile
        using a running average. Tracks source paths to prevent
        counting the same image twice.

        Args:
            cat_name: Name of the cat
            hist_h, hist_s, hist_v: Normalized HSV histograms
            source_path: Optional path to source image (for deduplication)

        Returns:
            True if sample was added, False if skipped (duplicate)
        """
        if cat_name not in self.profiles:
            self.profiles[cat_name] = {
                'hist_h': hist_h.copy(),
                'hist_s': hist_s.copy(),
                'hist_v': hist_v.copy(),
                'sample_count': 1,
                'sources': [source_path] if source_path else []
            }
            return True

        profile = self.profiles[cat_name]

        # Skip if this source was already processed
        if source_path and source_path in profile.get('sources', []):
            return False

        n = profile['sample_count']

        profile['hist_h'] = (profile['hist_h'] * n + hist_h) / (n + 1)
        profile['hist_s'] = (profile['hist_s'] * n + hist_s) / (n + 1)
        profile['hist_v'] = (profile['hist_v'] * n + hist_v) / (n + 1)
        profile['sample_count'] = n + 1

        if source_path:
            if 'sources' not in profile:
                profile['sources'] = []
            profile['sources'].append(source_path)

        return True

    def identify(self, hist_h, hist_s, hist_v):
        """
        Identify cat by comparing histograms to learned profiles.
        Always returns the best matching cat if profiles exist.

        Args:
            hist_h, hist_s, hist_v: HSV histograms

        Returns:
            cat_name: Best matching cat name or "Unknown" if no profiles
            confidence: Confidence score (0-1)
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
        confidence = max(0.0, 1.0 - best_distance)

        return best_cat, confidence, distances

    def identify_exclusive(self, histograms):
        """
        Identify multiple cats with exclusive assignment so no two tracks share a name.

        Uses the Hungarian algorithm to find the global minimum-cost assignment when
        multiple tracks need identification simultaneously. Prevents both tracks from
        grabbing the same cat name when profiles are ambiguous.

        Args:
            histograms: list of (hist_h, hist_s, hist_v) tuples, one per track

        Returns:
            list of (cat_name, confidence) in the same order as input
        """
        from scipy.optimize import linear_sum_assignment

        if not self.profiles or not histograms:
            return [("Unknown", 0.0)] * len(histograms)

        if len(histograms) == 1 or len(self.profiles) == 1:
            results = []
            for h, s, v in histograms:
                name, conf, _ = self.identify(h, s, v)
                results.append((name, conf))
            return results

        cat_names = list(self.profiles.keys())
        n_tracks = len(histograms)
        n_cats = len(cat_names)

        cost = np.zeros((n_tracks, n_cats), dtype=np.float32)
        for i, (h, s, v) in enumerate(histograms):
            for j, name in enumerate(cat_names):
                p = self.profiles[name]
                cost[i, j] = self._bhattacharyya_distance(
                    h, s, v, p['hist_h'], p['hist_s'], p['hist_v']
                )

        track_idx, cat_idx = linear_sum_assignment(cost)

        results = [("Unknown", 0.0)] * n_tracks
        for ti, ci in zip(track_idx, cat_idx):
            results[ti] = (cat_names[ci], max(0.0, 1.0 - cost[ti, ci]))

        return results

    def _bhattacharyya_distance(self, h1_h, h1_s, h1_v, h2_h, h2_s, h2_v):
        """
        Compute Bhattacharyya distance between two HSV histograms.

        Uses per-channel weights stored in self.hsv_weights.

        Returns:
            distance: 0 = identical, 1 = completely different
        """
        bc_h = np.sum(np.sqrt(h1_h * h2_h))
        bc_s = np.sum(np.sqrt(h1_s * h2_s))
        bc_v = np.sum(np.sqrt(h1_v * h2_v))

        w_h, w_s, w_v = self.hsv_weights
        bc = w_h * bc_h + w_s * bc_s + w_v * bc_v
        bc = np.clip(bc, 0.0, 1.0)

        return np.sqrt(1.0 - bc)
