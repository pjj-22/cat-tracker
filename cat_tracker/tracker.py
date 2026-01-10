"""
Track class representing a single cat being tracked across frames.
"""

from .kalman_filter import BBoxKalmanFilter


class Track:
    """
    A single track representing one cat.
    """
    
    _next_id = 1
    
    def __init__(self, bbox, confidence):
        """
        Initialize a new track.
        
        Args:
            bbox: Initial bounding box [x_center, y_center, width, height]
            confidence: Detection confidence score
        """
        self.id = Track._next_id
        Track._next_id += 1
        
        # Kalman filter for motion prediction
        self.kf = BBoxKalmanFilter(bbox)
        
        # Current state
        self.bbox = bbox
        self.predicted_bbox = bbox
        self.confidence = confidence
        
        # Track management
        self.hits = 1  # Number of times this track was matched
        self.missed_frames = 0  # Consecutive frames without detection
        self.age = 0  # Total frames this track has existed
        
    def predict(self):
        """Predict next position."""
        self.predicted_bbox = self.kf.predict()
        self.age += 1
        return self.predicted_bbox
    
    def update(self, bbox, confidence):
        """
        Update track with new detection.
        
        Args:
            bbox: New detected bounding box
            confidence: Detection confidence
        """
        self.kf.update(bbox)
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.missed_frames = 0
    
    def mark_missed(self):
        """Mark this track as not detected in current frame."""
        self.missed_frames += 1
        # Use predicted position as current position
        self.bbox = self.predicted_bbox
    
    def is_confirmed(self):
        """
        Check if track is confirmed (not a false detection).
        Requires multiple hits and tolerates brief occlusions.
        """
        return self.hits >= 3 and self.missed_frames < 3
    
    def should_delete(self, max_missed=10):
        """Check if track should be deleted (lost for too long)."""
        return self.missed_frames > max_missed