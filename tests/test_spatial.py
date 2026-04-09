"""Tests for spatial module: PositionLogger, Calibration, ZoneAnalyzer."""

import csv
import json
import os
import tempfile

import numpy as np
import pandas as pd

from cat_tracker.spatial import PositionLogger, Calibration, ZoneAnalyzer


class TestPositionLogger:
    def test_writes_header_and_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            with PositionLogger(path) as logger:
                logger.log("Honey", 100.0, 200.0, 50.0, 60.0)
                logger.log("Mochi", 300.0, 400.0, 40.0, 45.0)

            with open(path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[0] == ['timestamp', 'cat_name', 'pixel_x', 'pixel_y', 'width', 'height']
            assert len(rows) == 3
            assert rows[1][1] == "Honey"
            assert rows[2][1] == "Mochi"
        finally:
            os.unlink(path)

    def test_context_manager_closes_file(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            logger = PositionLogger(path)
            logger.__enter__()
            logger.__exit__(None, None, None)
            assert logger._file.closed
        finally:
            os.unlink(path)


class TestCalibration:
    def _temp_path(self, suffix=".json"):
        """Return a temp path that does NOT yet exist on disk."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        os.unlink(path)
        return path

    def _make_identity_calibration(self, path):
        """Create a calibration where pixel coords ≈ floor coords (identity)."""
        pixels = [[0, 0], [100, 0], [100, 100], [0, 100]]
        real = [[0, 0], [1, 0], [1, 1], [0, 1]]
        cal = Calibration(calibration_path=path)
        cal.save(pixels, real)
        return cal

    def test_save_creates_file(self):
        path = self._temp_path()
        try:
            self._make_identity_calibration(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert 'homography' in data
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_roundtrip(self):
        path = self._temp_path()
        try:
            cal1 = self._make_identity_calibration(path)
            cal2 = Calibration(calibration_path=path)
            np.testing.assert_allclose(cal1.homography, cal2.homography, atol=1e-6)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_pixel_to_floor(self):
        path = self._temp_path()
        try:
            cal = self._make_identity_calibration(path)
            fx, fy = cal.pixel_to_floor(50, 50)
            assert abs(fx - 0.5) < 0.05
            assert abs(fy - 0.5) < 0.05
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_pixel_to_floor_no_calibration_raises(self):
        cal = Calibration(calibration_path="/tmp/does_not_exist.json")
        try:
            cal.pixel_to_floor(0, 0)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass


class TestZoneAnalyzer:
    def _make_zones_file(self, path):
        zones = {
            "Kitchen": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            "Bedroom": {"x1": 200, "y1": 200, "x2": 300, "y2": 300},
        }
        with open(path, 'w') as f:
            json.dump(zones, f)

    def test_analyze_basic(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            zones_path = f.name

        try:
            self._make_zones_file(zones_path)
            analyzer = ZoneAnalyzer(
                zones_path=zones_path,
                calibration_path="/tmp/does_not_exist.json",
            )

            df = pd.DataFrame({
                'cat_name': ['Honey'] * 4,
                'pixel_x': [50, 50, 250, 500],
                'pixel_y': [50, 50, 250, 500],
            })

            results = analyzer.analyze(df, cat_name='Honey')
            assert results['Kitchen'] == 50.0
            assert results['Bedroom'] == 25.0
            assert results['Other'] == 25.0
        finally:
            os.unlink(zones_path)

    def test_analyze_empty_df(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            zones_path = f.name

        try:
            self._make_zones_file(zones_path)
            analyzer = ZoneAnalyzer(
                zones_path=zones_path,
                calibration_path="/tmp/does_not_exist.json",
            )
            df = pd.DataFrame({'cat_name': [], 'pixel_x': [], 'pixel_y': []})
            results = analyzer.analyze(df)
            assert results == {}
        finally:
            os.unlink(zones_path)

    def test_analyze_filters_by_cat(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            zones_path = f.name

        try:
            self._make_zones_file(zones_path)
            analyzer = ZoneAnalyzer(
                zones_path=zones_path,
                calibration_path="/tmp/does_not_exist.json",
            )

            df = pd.DataFrame({
                'cat_name': ['Honey', 'Mochi', 'Honey'],
                'pixel_x': [50, 250, 250],
                'pixel_y': [50, 250, 250],
            })

            results = analyzer.analyze(df, cat_name='Honey')
            # Should only consider Honey's 2 rows
            assert abs(results['Kitchen'] - 50.0) < 0.01
            assert abs(results['Bedroom'] - 50.0) < 0.01
        finally:
            os.unlink(zones_path)
