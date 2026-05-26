"""
Multi-object tracker that manages multiple cat tracks.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .tracker import Track
from .utils import iou, euclidean_distance


class MultiTracker:
    """
    Manages multiple tracks and performs data association.
    """

    def __init__(self, max_missed=10, min_hits=3, iou_threshold=0.3,
                 model_w=320, model_h=320):
        """
        Initialize multi-tracker.

        Args:
            max_missed: Maximum frames a track can be missed before deletion
            min_hits: Minimum hits before track is confirmed
            iou_threshold: Minimum IoU for matching detection to track
            model_w, model_h: YOLO input dimensions (used for distance normalization)
        """
        self.tracks = []
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._img_diagonal = np.sqrt(model_w ** 2 + model_h ** 2)
        # Gate pairs further apart than half the frame width — won't be the same cat
        self._max_match_dist = model_w * 0.5

    def update(self, detections):
        """
        Update tracks with new detections.

        Args:
            detections: List of detections, each is dict with 'box' and 'confidence'

        Returns:
            List of confirmed tracks
        """
        for track in self.tracks:
            track.predict()

        if len(detections) > 0 and len(self.tracks) > 0:
            matches, unmatched_dets, unmatched_tracks = self._match(detections)
        else:
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                detections[det_idx]['box'],
                detections[det_idx]['confidence']
            )

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                detections[det_idx]['box'],
                detections[det_idx]['confidence'],
                track_id=self._alloc_id(),
                min_hits=self.min_hits,
            )
            self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if not t.should_delete(self.max_missed)]
        self._deduplicate()

        return [t for t in self.tracks if t.is_confirmed()]

    def _deduplicate(self):
        """Drop ghost tracks that are duplicates of a better-established track.

        Two confirmed tracks with the same coat name are always the same cat;
        one is a ghost. Unidentified tracks are left alone until coat matching
        runs. Only runs on YOLO frames so we compare against fresh detections.
        """
        confirmed = [t for t in self.tracks if t.is_confirmed()]
        to_delete = set()

        for i in range(len(confirmed)):
            for j in range(i + 1, len(confirmed)):
                a, b = confirmed[i], confirmed[j]
                if a.name == "Unknown" or a.name != b.name:
                    continue
                if a.missed_frames == 0 and b.missed_frames > 0:
                    to_delete.add(b.id)
                elif b.missed_frames == 0 and a.missed_frames > 0:
                    to_delete.add(a.id)
                else:
                    to_delete.add(b.id if a.hits >= b.hits else a.id)

        if to_delete:
            self.tracks = [t for t in self.tracks if t.id not in to_delete]

    def compensate_camera_motion(self, dx, dy):
        """Shift Kalman state to account for camera pan/tilt.

        When the servo moves, every tracked object shifts by the same pixel
        delta in the opposite direction. Without this, the predicted position
        stays at the old pixel location while YOLO detects the cat at the new
        one, so IoU drops to ~0, Hungarian fails to match, and a ghost duplicate
        track is created.

        Sign convention (matches servo.py auto_follow):
          dx > 0  -> camera panned right -> cat shifts left  (x decreases)
          dy > 0  -> camera tilted up    -> cat shifts down  (y increases)
        """
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            track.kf.kf.x[0] -= dx
            track.kf.kf.x[1] += dy
            track.kf.kf.x[4] = 0.0  # velocity is meaningless after a pan
            track.kf.kf.x[5] = 0.0
            track.kf.kf.P[0, 0] += 500.0
            track.kf.kf.P[1, 1] += 500.0
            track.kf.kf.P[4, 4] += 200.0 
            track.kf.kf.P[5, 5] += 200.0

    def predict_only(self):
        """Advance Kalman predictions without processing detections.

        Used on frames where YOLO inference is skipped; tracks move forward
        without being marked missed or matched against new detections.
        """
        for track in self.tracks:
            track.predict()
        return [t for t in self.tracks if t.is_confirmed()]

    def _alloc_id(self):
        id = self._next_id
        self._next_id += 1
        return id

    def _match(self, detections):
        """
        Match detections to existing tracks using Hungarian algorithm.

        Returns:
            matches: List of (track_idx, detection_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost_matrix = np.zeros((n_tracks, n_dets))
        iou_matrix = np.zeros((n_tracks, n_dets))

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_score = iou(track.predicted_bbox, det['box'])
                center_dist = euclidean_distance(track.predicted_bbox, det['box'])
                iou_matrix[i, j] = iou_score

                # Gate pairs too far apart
                if center_dist > self._max_match_dist:
                    cost_matrix[i, j] = 1e6
                else:
                    normalized_dist = center_dist / self._img_diagonal
                    cost_matrix[i, j] = 0.7 * (1 - iou_score) + 0.3 * normalized_dist

        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_dets = list(range(n_dets))
        unmatched_tracks = list(range(n_tracks))

        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] >= 1e6:
                continue
            if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                matches.append((track_idx, det_idx))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(track_idx)

        return matches, unmatched_dets, unmatched_tracks
