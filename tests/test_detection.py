"""Tests for cat_tracker.detection (preprocessing and output parsing)."""

import numpy as np
from cat_tracker.detection import preprocess_frame, parse_yolo_output, CAT_CLASS_ID


def test_preprocess_frame_shape():
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = preprocess_frame(frame, 640, 640)
    assert result.shape == (1, 3, 640, 640)


def test_preprocess_frame_dtype():
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = preprocess_frame(frame, 640, 640)
    assert result.dtype == np.float32


def test_preprocess_frame_range():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    result = preprocess_frame(frame, 640, 640)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def _make_yolo_output(boxes, class_id, confidence):
    """Create a synthetic YOLO output array.

    YOLO output shape for a single image: (1, 4+num_classes, num_detections).
    ``parse_yolo_output`` transposes to (num_detections, 4+num_classes).
    """
    num_classes = 80
    num_dets = len(boxes)
    output = np.zeros((1, 4 + num_classes, num_dets), dtype=np.float32)
    for i, (x, y, w, h) in enumerate(boxes):
        output[0, 0, i] = x
        output[0, 1, i] = y
        output[0, 2, i] = w
        output[0, 3, i] = h
        output[0, 4 + class_id, i] = confidence
    return output


def test_parse_yolo_output_single_cat():
    output = _make_yolo_output(
        boxes=[(320, 240, 100, 80)],
        class_id=CAT_CLASS_ID,
        confidence=0.9,
    )
    dets = parse_yolo_output(output, conf_threshold=0.15, iou_threshold=0.4)
    assert len(dets) == 1
    assert dets[0]['confidence'] >= 0.15
    np.testing.assert_allclose(dets[0]['box'][:2], [320, 240])


def test_parse_yolo_output_no_cat():
    output = _make_yolo_output(
        boxes=[(320, 240, 100, 80)],
        class_id=0,  # person, not cat
        confidence=0.9,
    )
    dets = parse_yolo_output(output, conf_threshold=0.15, iou_threshold=0.4)
    assert len(dets) == 0


def test_parse_yolo_output_below_threshold():
    output = _make_yolo_output(
        boxes=[(320, 240, 100, 80)],
        class_id=CAT_CLASS_ID,
        confidence=0.05,
    )
    dets = parse_yolo_output(output, conf_threshold=0.15, iou_threshold=0.4)
    assert len(dets) == 0


def test_parse_yolo_output_empty():
    output = np.zeros((1, 84, 0), dtype=np.float32)
    dets = parse_yolo_output(output)
    assert dets == []
