"""
Kalman Filter for tracking cat bounding boxes.
Predicts where a cat will be in the next frame based on its motion history.
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
        
        # Measurement uncertainty (detection might be slightly off)
        self.kf.R *= 10.0
        
        # Process uncertainty (how much we trust the motion model)
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initial state
        self.kf.x[:4] = bbox.reshape(4, 1)
        
    def predict(self):
        """
        Predict next position based on current state.
        
        Returns:
            Predicted bbox: [x_center, y_center, width, height]
        """
        self.kf.predict()
        return self.kf.x[:4].flatten()
    
    def update(self, bbox):
        """
        Update filter with new measurement.
        
        Args:
            bbox: Detected bounding box [x_center, y_center, width, height]
        """
        self.kf.update(bbox.reshape(4, 1))
    
    def get_state(self):
        """Get current state estimate."""
        return self.kf.x[:4].flatten()