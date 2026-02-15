"""Tests for cat_tracker.utils."""

import numpy as np
from cat_tracker.utils import (
    iou,
    euclidean_distance,
    bbox_to_pixel_xyxy,
    clamp_bbox_to_image,
    bbox_to_xyxy,
    xyxy_to_bbox,
)


def test_iou_identical_boxes():
    box = np.array([100, 100, 50, 50])
    assert iou(box, box) == 1.0


def test_iou_no_overlap():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([100, 100, 10, 10])
    assert iou(box1, box2) == 0.0


def test_iou_partial_overlap():
    box1 = np.array([10, 10, 20, 20])  # spans 0-20, 0-20
    box2 = np.array([20, 10, 20, 20])  # spans 10-30, 0-20
    score = iou(box1, box2)
    assert 0.0 < score < 1.0


def test_iou_zero_area():
    box1 = np.array([0, 0, 0, 0])
    box2 = np.array([0, 0, 0, 0])
    assert iou(box1, box2) == 0


def test_euclidean_distance_same_point():
    box = np.array([50, 50, 10, 10])
    assert euclidean_distance(box, box) == 0.0


def test_euclidean_distance():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([3, 4, 10, 10])
    assert abs(euclidean_distance(box1, box2) - 5.0) < 1e-6


def test_bbox_to_pixel_xyxy_center():
    # Box in model coords centered at (320, 240) with size 100x100
    # Model is 640x480, frame is 640x480 → 1:1 mapping
    x1, y1, x2, y2 = bbox_to_pixel_xyxy(
        np.array([320, 240, 100, 100]), 640, 480, 640, 480
    )
    assert x1 == 270
    assert y1 == 190
    assert x2 == 370
    assert y2 == 290


def test_bbox_to_pixel_xyxy_scaling():
    # Model 640x640 → frame 1280x960 (2x)
    x1, y1, x2, y2 = bbox_to_pixel_xyxy(
        np.array([320, 320, 100, 100]), 640, 640, 1280, 960
    )
    # Center should be at 640, 480 in pixel space; size 200x150
    assert x1 == int(640 - 100)
    assert x2 == int(640 + 100)


def test_clamp_bbox_to_image():
    x1, y1, x2, y2 = clamp_bbox_to_image((-10, -10, 700, 500), 640, 480)
    assert x1 == 0
    assert y1 == 0
    assert x2 == 640
    assert y2 == 480


def test_clamp_bbox_inside():
    x1, y1, x2, y2 = clamp_bbox_to_image((10, 10, 100, 100), 640, 480)
    assert (x1, y1, x2, y2) == (10, 10, 100, 100)


def test_bbox_to_xyxy_roundtrip():
    original = np.array([50, 60, 30, 40])
    xyxy = bbox_to_xyxy(original)
    recovered = xyxy_to_bbox(xyxy)
    np.testing.assert_allclose(recovered, original)
