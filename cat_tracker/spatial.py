"""
Spatial tracking utilities for position logging, camera calibration,
and zone analysis.
"""

import csv
import json
import os
from datetime import datetime

import cv2
import numpy as np


class PositionLogger:
    """Log cat pixel positions to CSV. Supports context manager protocol."""

    def __init__(self, filepath='occupancy_log.csv'):
        self.filepath = filepath
        self._file = None
        self.writer = None
        self._open()

    def _open(self):
        self._file = open(self.filepath, 'a', newline='')
        self.writer = csv.writer(self._file)

        # Write header if new file
        self._file.seek(0, 2)
        if self._file.tell() == 0:
            self.writer.writerow(['timestamp', 'cat_name', 'pixel_x', 'pixel_y', 'width', 'height'])

    def log(self, cat_name, pixel_x, pixel_y, width, height):
        """Log a position entry with bounding box dimensions."""
        timestamp = datetime.now().isoformat()
        self.writer.writerow([timestamp, cat_name, pixel_x, pixel_y, width, height])
        self._file.flush()

    def close(self):
        """Close the log file."""
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class Calibration:
    """
    Camera-to-floor homography calibration.

    Wraps an OpenCV homography matrix that maps pixel coordinates
    to real-world floor coordinates (in meters).
    """

    def __init__(self, calibration_path='calibration.json'):
        self.calibration_path = calibration_path
        self.homography = None

        if os.path.exists(calibration_path):
            self.load()

    def save(self, points_pixel, points_real):
        """
        Compute and save a homography from pixel↔real point pairs.

        Args:
            points_pixel: List of [x, y] pixel coordinates (≥ 4 points)
            points_real: List of [x, y] real-world coordinates in meters
        """
        pts_pixel = np.array(points_pixel, dtype=np.float32)
        pts_real = np.array(points_real, dtype=np.float32)

        H, _ = cv2.findHomography(pts_pixel, pts_real, method=cv2.RANSAC)
        self.homography = H

        data = {
            'homography': H.tolist(),
            'points_pixel': pts_pixel.tolist(),
            'points_real': pts_real.tolist(),
        }
        with open(self.calibration_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load a previously saved homography from disk."""
        with open(self.calibration_path, 'r') as f:
            data = json.load(f)

        self.homography = np.array(data['homography'], dtype=np.float64)

    def pixel_to_floor(self, px, py):
        """
        Convert a single pixel coordinate to floor coordinates.

        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate

        Returns:
            (floor_x, floor_y) in meters

        Raises:
            RuntimeError: If no homography has been loaded/computed.
        """
        if self.homography is None:
            raise RuntimeError("No calibration loaded. Run save() or load() first.")

        pt = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography)
        floor_x, floor_y = transformed[0, 0]
        return float(floor_x), float(floor_y)


class ZoneAnalyzer:
    """
    Analyze cat position data against user-defined room zones.

    Zones are axis-aligned rectangles defined in meters (matching
    calibration output). If no calibration exists, works directly
    with whatever coordinate system is in the CSV data.
    """

    def __init__(self, zones_path='zones.json', calibration_path='calibration.json'):
        """
        Load zone definitions and optional calibration.

        Args:
            zones_path: Path to JSON file defining named rectangular zones.
            calibration_path: Path to calibration JSON (optional).
        """
        self.zones = {}
        self.calibration = None

        if os.path.exists(zones_path):
            with open(zones_path, 'r') as f:
                self.zones = json.load(f)

        if os.path.exists(calibration_path):
            self.calibration = Calibration(calibration_path)

    def analyze(self, df, cat_name=None):
        """
        Compute percentage of time spent in each zone.

        Args:
            df: pandas DataFrame with columns 'cat_name', 'pixel_x', 'pixel_y'.
            cat_name: If given, filter to this cat only.

        Returns:
            dict of zone_name → percentage (0-100). Includes an "Other"
            entry for time outside all defined zones.
        """
        if cat_name is not None:
            df = df[df['cat_name'] == cat_name]

        total = len(df)
        if total == 0:
            return {}

        # Convert pixel positions to floor coordinates if calibration exists
        if self.calibration is not None and self.calibration.homography is not None:
            coords = np.column_stack([df['pixel_x'].values, df['pixel_y'].values])
            pts = coords.reshape(-1, 1, 2).astype(np.float32)
            transformed = cv2.perspectiveTransform(pts, self.calibration.homography)
            xs = transformed[:, 0, 0]
            ys = transformed[:, 0, 1]
        else:
            xs = df['pixel_x'].values.astype(float)
            ys = df['pixel_y'].values.astype(float)

        zone_counts = {}
        accounted = np.zeros(total, dtype=bool)

        for zone_name, zone_def in self.zones.items():
            inside = (
                (xs >= zone_def['x1']) & (xs <= zone_def['x2']) &
                (ys >= zone_def['y1']) & (ys <= zone_def['y2'])
            )
            zone_counts[zone_name] = float(np.sum(inside))
            accounted |= inside

        zone_counts['Other'] = float(np.sum(~accounted))

        results = {}
        for name, count in zone_counts.items():
            results[name] = (count / total) * 100.0

        return results
