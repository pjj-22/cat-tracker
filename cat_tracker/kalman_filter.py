"""
Kalman Filter for tracking cat bounding boxes.
Predicts where a cat will be in the next frame based on its motion history.

TODO: rewrite in C++ extension for Pi performance
"""

import numpy as np
from filterpy.kalman import KalmanFilter


class BBoxKalmanFilter:
    """
    Kalman Filter for bounding box tracking.
    
    State vector: [x, y, w, h, vx, vy, vw, vh]
    - x, y: center coordinates
    - w, h: width and height
    - vx, vy: velocity in x and y
    - vw, vh: velocity of width and height (usually ~0 for cats)
    """
    
    def __init__(self, bbox):
        """
        Initialize Kalman filter with initial bounding box.
        
        Args:
            bbox: [x_center, y_center, width, height]
        """
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        self.kf.R *= 10.0

        # vw/vh constrained — box size doesn't change much between frames
        self.kf.Q[6:8, 6:8] *= 0.01

        self.kf.P[4:6, 4:6] *= 1000.0
        self.kf.P[6:8, 6:8] *= 100.0

        self.kf.x[:4] = bbox.reshape(4, 1)
        
    def _clamp_dimensions(self):
        self.kf.x[2] = max(10.0, self.kf.x[2].item())
        self.kf.x[3] = max(10.0, self.kf.x[3].item())

    def predict(self):
        self.kf.predict()
        self._clamp_dimensions()
        return self.kf.x[:4].flatten()

    def update(self, bbox):
        self.kf.update(bbox.reshape(4, 1))
        self._clamp_dimensions()
    
    def get_state(self):
        """Get current state estimate."""
        return self.kf.x[:4].flatten()

    def get_velocity(self):
        """Get current velocity estimate [vx, vy] in model-space pixels/frame."""
        return self.kf.x[4:6].flatten()