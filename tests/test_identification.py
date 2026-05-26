"""Tests for ColorHistogramExtractor and ColorHistogramIdentifier."""

import os
import json
import tempfile

import numpy as np
from cat_tracker.prefix_colors import ColorHistogramExtractor, ColorHistogramIdentifier


class TestColorHistogramExtractor:
    def test_extract_returns_histograms(self):
        # Create a solid-color image (orange-ish in RGB)
        frame = np.full((100, 100, 3), [200, 120, 50], dtype=np.uint8)
        extractor = ColorHistogramExtractor()
        h, s, v = extractor.extract(frame, (10, 10, 90, 90))
        assert h is not None
        assert s is not None
        assert v is not None
        assert h.shape == (30,)
        assert s.shape == (32,)
        assert v.shape == (32,)

    def test_extract_normalized(self):
        frame = np.full((100, 100, 3), [200, 120, 50], dtype=np.uint8)
        extractor = ColorHistogramExtractor()
        h, s, v = extractor.extract(frame, (10, 10, 90, 90))
        if h is not None:
            assert abs(np.sum(h) - 1.0) < 1e-5
            assert abs(np.sum(s) - 1.0) < 1e-5
            assert abs(np.sum(v) - 1.0) < 1e-5

    def test_extract_empty_roi_returns_none(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        extractor = ColorHistogramExtractor()
        h, s, v = extractor.extract(frame, (50, 50, 50, 50))  # zero-area
        assert h is None

    def test_extract_low_saturation_returns_none(self):
        # Gray image, all pixels should be masked out
        frame = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
        extractor = ColorHistogramExtractor(min_saturation=20, min_value=20)
        h, s, v = extractor.extract(frame, (0, 0, 100, 100))
        assert h is None


class TestColorHistogramIdentifier:
    def test_identify_unknown_when_empty(self):
        identifier = ColorHistogramIdentifier(profile_path="/tmp/nonexistent.json")
        h = np.ones(30, dtype=np.float32) / 30
        s = np.ones(32, dtype=np.float32) / 32
        v = np.ones(32, dtype=np.float32) / 32
        name, conf, _ = identifier.identify(h, s, v)
        assert name == "Unknown"
        assert conf == 0.0

    def test_add_training_sample_and_identify(self):
        identifier = ColorHistogramIdentifier(profile_path="/tmp/nonexistent.json")
        h = np.random.dirichlet(np.ones(30)).astype(np.float32)
        s = np.random.dirichlet(np.ones(32)).astype(np.float32)
        v = np.random.dirichlet(np.ones(32)).astype(np.float32)

        identifier.add_training_sample("Honey", h, s, v)
        name, conf, _ = identifier.identify(h, s, v)
        assert name == "Honey"
        assert conf > 0.5

    def test_duplicate_source_skipped(self):
        identifier = ColorHistogramIdentifier(profile_path="/tmp/nonexistent.json")
        h = np.ones(30, dtype=np.float32) / 30
        s = np.ones(32, dtype=np.float32) / 32
        v = np.ones(32, dtype=np.float32) / 32

        assert identifier.add_training_sample("Cat", h, s, v, source_path="img1.jpg")
        assert not identifier.add_training_sample("Cat", h, s, v, source_path="img1.jpg")

    def test_save_load_roundtrip(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(path)

        try:
            id1 = ColorHistogramIdentifier(profile_path=path)
            h = np.random.dirichlet(np.ones(30)).astype(np.float32)
            s = np.random.dirichlet(np.ones(32)).astype(np.float32)
            v = np.random.dirichlet(np.ones(32)).astype(np.float32)
            id1.add_training_sample("Mochi", h, s, v)
            id1.save_profiles()

            id2 = ColorHistogramIdentifier(profile_path=path)
            assert "Mochi" in id2.profiles
            np.testing.assert_allclose(
                id2.profiles["Mochi"]["hist_h"], h, atol=1e-5
            )
        finally:
            os.unlink(path)

    def test_hsv_weights_customizable(self):
        identifier = ColorHistogramIdentifier(
            profile_path="/tmp/nonexistent.json",
            hsv_weights=[1.0, 0.0, 0.0],
        )
        assert identifier.hsv_weights == (1.0, 0.0, 0.0)
