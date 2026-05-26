"""
Track class representing a single cat being tracked across frames.
"""

from .kalman_filter import BBoxKalmanFilter


class Track:
    """
    A single track representing one cat.
    """

    def __init__(self, bbox, confidence, track_id=1, min_hits=3):
        self.id = track_id
        self._min_hits = min_hits

        self.kf = BBoxKalmanFilter(bbox)

        self.bbox = bbox
        self.predicted_bbox = bbox
        self.confidence = confidence

        self.hits = 1             # number of times this track was matched
        self.missed_frames = 0    # consecutive frames without a detection
        self.age = 0              # total frames this track has existed
        self.frames_since_identified = 0
        self.name = "Unknown"
        self.name_confidence = 0.0

    def predict(self):
        """Predict next position."""
        self.predicted_bbox = self.kf.predict()
        self.bbox = self.predicted_bbox
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
        self.bbox = self.kf.get_state()  # smoothed posterior, not raw detection
        self.confidence = confidence
        self.hits += 1
        self.missed_frames = 0
        self.frames_since_identified += 1

    def mark_missed(self):
        self.missed_frames += 1
        self.kf.kf.x[6] = 0.0  # freeze box size, no detection to correct drift
        self.kf.kf.x[7] = 0.0
        self.bbox = self.predicted_bbox

    def is_confirmed(self):
        return self.hits >= self._min_hits

    @property
    def velocity(self):
        """Current velocity estimate [vx, vy] in model-space pixels/frame."""
        return self.kf.get_velocity()

    def should_delete(self, max_missed=10):
        """Check if track should be deleted (lost for too long)."""
        return self.missed_frames > max_missed
