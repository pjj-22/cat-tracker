"""
Utility functions for tracking.
"""

import numpy as np


def bbox_to_xyxy(bbox):
    """
    Convert bbox from [x_center, y_center, w, h] to [x1, y1, x2, y2].
    YOLOv8 and Kalman filters use center format, but OpenCV's
    cv2.rectangle() needs corner format (top-left, bottom-right).
    """
    x, y, w, h = bbox
    return np.array([x - w/2, y - h/2, x + w/2, y + h/2])


def xyxy_to_bbox(xyxy):
    """
    If we receive corner-format boxes (e.g., from some detection tools),
    we need to convert them to center format for YOLOv8/Kalman compatibility.

    Convert bbox from [x1, y1, x2, y2] to [x_center, y_center, w, h].
    """
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    return np.array([x1 + w/2, y1 + h/2, w, h])


def iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes in [x_center, y_center, w, h] format
        
    Returns:
        IoU score (0 to 1)
    """
    box1 = bbox_to_xyxy(bbox1)
    box2 = bbox_to_xyxy(bbox2)
    
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    return intersection / union


def euclidean_distance(bbox1, bbox2):
    """
    Calculate Euclidean distance between centers of two bounding boxes.
    """
    return np.linalg.norm(bbox1[:2] - bbox2[:2])


def clamp_bbox_to_image(bbox, img_width, img_height):
    """
    Clamp bounding box coordinates to image boundaries.
    
    Args:
        bbox: [x1, y1, x2, y2] in corner format
        img_width: width
        img_height: height
        
    Returns:
        Clamped bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, min(x1, img_width - 1))
    x2 = max(0, min(x2, img_width))
    y1 = max(0, min(y1, img_height - 1))
    y2 = max(0, min(y2, img_height))
    
    return x1, y1, x2, y2