"""
YOLO detection utilities and shared constants.
"""

import os
import numpy as np
import cv2
import onnxruntime as ort


# Colors for different track IDs (up to 10 cats)
TRACK_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
    (0, 128, 255),  # Light Blue
    (128, 255, 0),  # Lime
]

# YOLO class ID for cats
CAT_CLASS_ID = 15

# Default model path
DEFAULT_MODEL_PATH = "yolov8n.onnx"


def load_yolo_model(model_path=DEFAULT_MODEL_PATH):
    """
    Load YOLO ONNX model with existence check.

    Args:
        model_path: Path to the ONNX model file

    Returns:
        tuple: (session, input_name, model_height, model_width)

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}\n\n"
            f"Please download the YOLOv8 ONNX model:\n"
            f"  1. Install ultralytics: pip install ultralytics\n"
            f"  2. Export model: yolo export model=yolov8s.pt format=onnx\n"
            f"  3. Move yolov8s.onnx to this directory\n\n"
            f"Or download directly from Ultralytics."
        )

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    model_h, model_w = input_shape[2], input_shape[3]

    return session, input_name, model_h, model_w


def parse_yolo_output(output, conf_threshold=0.2, iou_threshold=0.4):
    """
    Parse YOLOv8 ONNX output and return cat detections.
    """
    output = output[0].T
    boxes, boxes_tl, scores = [], [], []

    for detection in output:
        box = detection[:4]
        class_scores = detection[4:]

        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if class_id == CAT_CLASS_ID and confidence > conf_threshold:
            x, y, w, h = box
            boxes.append(box)
            boxes_tl.append([x - w / 2, y - h / 2, w, h])
            scores.append(float(confidence))

    if len(boxes) == 0:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes_tl,
        scores,
        conf_threshold,
        iou_threshold,
    )

    if len(indices) > 0:
        indices = indices.flatten()

    detections = []
    for i in indices:
        detections.append({
            'box': np.array(boxes[i]),
            'confidence': scores[i]
        })

    return detections



def preprocess_frame(frame, model_w, model_h):
    """
    Preprocess frame for YOLO inference.

    Args:
        frame: RGB image (numpy array)
        model_w: Model input width
        model_h: Model input height

    Returns:
        Preprocessed input tensor ready for inference
    """
    resized = cv2.resize(frame, (model_w, model_h))
    input_data = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data
