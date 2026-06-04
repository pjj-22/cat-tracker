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
        self.name = "Unknown"
        self.name_confidence = 0.0
        self._candidate_name = "Unknown"
        self._candidate_streak = 0
        self.hist_h = None
        self.hist_s = None
        self.hist_v = None

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

    def mark_missed(self):
        self.missed_frames += 1
        self.kf.kf.x[4] *= 0.5  # damp velocity so the box doesn't fly off
        self.kf.kf.x[5] *= 0.5
        self.kf.kf.x[6] = 0.0
        self.kf.kf.x[7] = 0.0
        self.bbox = self.predicted_bbox

    def is_confirmed(self):
        return self.hits >= self._min_hits

    @property
    def velocity(self):
        """Current velocity estimate [vx, vy] in model-space pixels/frame."""
        return self.kf.get_velocity()

    def update_hist(self, hist_h, hist_s, hist_v, alpha=0.25):
        if self.hist_h is None:
            self.hist_h = hist_h.copy()
            self.hist_s = hist_s.copy()
            self.hist_v = hist_v.copy()
        else:
            self.hist_h = alpha * hist_h + (1.0 - alpha) * self.hist_h
            self.hist_s = alpha * hist_s + (1.0 - alpha) * self.hist_s
            self.hist_v = alpha * hist_v + (1.0 - alpha) * self.hist_v

    def should_delete(self, max_missed=10):
        """Check if track should be deleted (lost for too long)."""
        return self.missed_frames > max_missed
